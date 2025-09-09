# tripletgen/summarize_and_qa.py
from pathlib import Path
import re, torch
from typing import Optional, Dict, Any, List
from rouge_score import rouge_scorer
import os

# --- HF helper to auto-download checkpoint if missing ---
try:
    from tripletgen.hf_utils import ensure_latest_checkpoint
except Exception:
    def ensure_latest_checkpoint(out_dir: str = "checkpoints_clean", token: Optional[str] = None):
        # no-op fallback when hf_utils unavailable
        return None

# --- heuristics ---
YEAR_RE = re.compile(r'\b(19|20)\d{2}\b')
PROPER_RE = re.compile(r'\b([A-Z][a-z]{2,})\b')
def extract_years_list(text): return set(m.group(0) for m in YEAR_RE.finditer(text))
def extract_propers_smarter(text): return set(m.group(1) for m in PROPER_RE.finditer(text))
def hallucination_penalty(article, candidate):
    pen = 0.0; reasons=[]
    art_years = extract_years_list(article); cand_years = extract_years_list(candidate)
    novel_years = [y for y in cand_years if y not in art_years]
    if novel_years:
        pen += 1.2 * len(novel_years); reasons.append(f"novel_years:{','.join(novel_years)}")
    art_propers = extract_propers_smarter(article); cand_propers = extract_propers_smarter(candidate)
    novel_propers = cand_propers - art_propers
    if len(novel_propers) >= 2:
        pen += 0.5 * (len(novel_propers)-1); reasons.append(f"novel_entities:{','.join(list(novel_propers)[:6])}")
    elif len(novel_propers) == 1:
        pen += 0.12; reasons.append(f"novel_entity:{list(novel_propers)[0]}")
    return pen, reasons

scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
def combined_score(article, reference, cand, rerank_score):
    r1 = 0.0
    if reference:
        r1 = scorer.score(reference, cand)["rouge1"].fmeasure
    pen, reasons = hallucination_penalty(article, cand)
    new = float(rerank_score) + 0.6 * float(r1) - pen
    return new, {"rouge1": r1, "pen": pen, "reasons": reasons}

# --- models container (lazy load) ---
class Models:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer_gen = None
        self.model_gen = None
        self.tok_r = None
        self.reranker = None
        self.tok_nli = None
        self.model_nli = None
        self.qa_pipeline = None

MODELS = Models()

# minimal reranker head
import torch.nn as nn
class RerankerModelWrap(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.encoder = base_model
        h = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(nn.Linear(h, h//2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h//2, 1))
    def forward(self, batch):
        out = self.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], return_dict=True)
        cls = out.last_hidden_state[:,0,:]
        score = self.classifier(cls).squeeze(-1)
        return score

def load_models(checkpoints_clean: str = "checkpoints_clean", load_qa: bool = True):
    """Lazy load models into MODELS and return it."""
    if MODELS.tokenizer_gen is not None:
        return MODELS
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, AutoTokenizer as AT, AutoModelForSequenceClassification, pipeline
    # Ensure we have a checkpoint available locally — attempt HF download if missing
    HF_TOKEN = os.environ.get("HF_TOKEN")
    try:
        ck_dir = Path(checkpoints_clean)
        if not ck_dir.exists() or not any(ck_dir.glob("reranker_ckpt_best_*.pt")):
            print("[summarize_and_qa] No local reranker checkpoint detected. Attempting HF download...")
            ensure_latest_checkpoint(out_dir=checkpoints_clean, token=HF_TOKEN)
    except Exception as e:
        print("[summarize_and_qa] HF download attempt failed:", e)

    MODELS.tokenizer_gen = AutoTokenizer.from_pretrained("facebook/bart-large-cnn", use_fast=False)
    MODELS.model_gen = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    try: MODELS.model_gen.to(MODELS.device)
    except: pass
    MODELS.model_gen.eval()

    MODELS.tok_r = AT.from_pretrained("roberta-base")
    base_enc = AutoModel.from_pretrained("roberta-base")
    MODELS.reranker = RerankerModelWrap(base_enc)
    # try to load reranker ckpt
    ck_dir = Path(checkpoints_clean)
    if ck_dir.exists():
        cands = sorted(ck_dir.glob("reranker_ckpt_best_*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not cands:
            cands = sorted(ck_dir.glob("reranker_ckpt_resumed_*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
        if cands:
            try:
                blob = torch.load(str(cands[0]), map_location="cpu")
                state = blob.get("model_state", blob)
                st = MODELS.reranker.state_dict()
                for k,v in state.items():
                    if k in st and v.shape == st[k].shape:
                        st[k] = v
                MODELS.reranker.load_state_dict(st)
                print(f"[summarize_and_qa] Loaded reranker weights from {cands[0]}")
            except Exception as e:
                print("[summarize_and_qa] Failed to load reranker checkpoint:", e)
    try: MODELS.reranker.to(MODELS.device)
    except: pass
    MODELS.reranker.eval()

    MODELS.tok_nli = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    MODELS.model_nli = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
    try: MODELS.model_nli.to(MODELS.device)
    except: pass
    MODELS.model_nli.eval()

    if load_qa:
        try:
            MODELS.qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad",
                                         tokenizer="distilbert-base-cased-distilled-squad", device=0 if MODELS.device.type=="cuda" else -1)
        except Exception:
            MODELS.qa_pipeline = None
    return MODELS

# reranker scoring helper
@torch.no_grad()
def reranker_score_batch(article: str, candidates: List[str], max_len: int = 384):
    mc = MODELS
    enc = mc.tok_r([article]*len(candidates), candidates, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    enc = {k:v.to(mc.device) for k,v in enc.items()}
    scores = mc.reranker(enc).detach().cpu().tolist()
    return list(zip(candidates, scores))

# main wrapper function
def summarize_and_qa(article: str, question: Optional[str] = None,
                     K: int = 6, num_beams: int = 6, max_new_tokens: int = 96,
                     entail_threshold: float = 0.8, checkpoints_clean: str = "checkpoints_clean") -> Dict[str, Any]:
    mc = load_models(checkpoints_clean=checkpoints_clean, load_qa=True)
    prefix = article
    try:
        toks = mc.tokenizer_gen(prefix, return_tensors="pt", truncation=False)
        if toks["input_ids"].shape[1] > 1024:
            prefix = prefix[:4000]
    except Exception:
        prefix = prefix[:4000]

    enc = mc.tokenizer_gen([prefix], truncation=True, padding=True, return_tensors="pt", max_length=1024)
    enc = {k:v.to(mc.device) for k,v in enc.items()}
    with torch.no_grad():
        outs = mc.model_gen.generate(**enc,
                                     max_new_tokens=max_new_tokens,
                                     num_beams=max(num_beams, K),
                                     num_return_sequences=K,
                                     early_stopping=True,
                                     no_repeat_ngram_size=3,
                                     trust_remote_code=True)
    cands = mc.tokenizer_gen.batch_decode(outs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    try:
        rer_scores = reranker_score_batch(prefix, cands)
    except Exception:
        rer_scores = [(c, 0.0) for c in cands]

    ranked = []
    for cand, s in rer_scores:
        new, meta = combined_score(prefix, "", cand, s)
        ranked.append({"candidate": cand, "rerank_score": float(s), "combined": float(new), **meta})
    ranked = sorted(ranked, key=lambda x: x["combined"], reverse=True)
    top = ranked[0]
    entail_prob = None
    try:
        enc_nli = mc.tok_nli(prefix[:4096], top["candidate"][:1024], truncation=True, padding=True, return_tensors="pt")
        enc_nli = {k:v.to(mc.device) for k,v in enc_nli.items()}
        with torch.no_grad():
            out = mc.model_nli(**enc_nli)
            probs = torch.softmax(out.logits, dim=-1)[0].cpu().numpy()
            entail_prob = float(probs[2])
    except Exception:
        try:
            enc_nli = mc.tok_nli(prefix[:4096], top["candidate"][:1024], truncation=True, padding=True, return_tensors="pt")
            enc_cpu = {k:v.to("cpu") for k,v in enc_nli.items()}
            mc.model_nli.to("cpu")
            with torch.no_grad():
                out = mc.model_nli(**enc_cpu)
                probs = torch.softmax(out.logits, dim=-1)[0].cpu().numpy()
                entail_prob = float(probs[2])
            mc.model_nli.to(mc.device)
        except Exception:
            entail_prob = None

    fallback_used = False; fallback = None
    if entail_prob is None or entail_prob < entail_threshold:
        fallback_used = True
        sents = re.split(r'(?<=[.!?])\s+', article.strip())
        def jaccard_sim(a,b):
            sa = set(re.findall(r"\w+", a.lower())); sb = set(re.findall(r"\w+", b.lower()))
            if not sa or not sb: return 0.0
            return len(sa & sb) / len(sa | sb)
        scores = [(i, jaccard_sim(top["candidate"], s)) for i,s in enumerate(sents)]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        top_idxs = [i for i,sc in scores[:3]]
        top_sents = [sents[i] for i in top_idxs if i < len(sents)]
        extractive_summary = " ".join(top_sents)
        try:
            rer_fb = reranker_score_batch(prefix, [extractive_summary])[0][1]
        except Exception:
            rer_fb = 0.0
        entail_fb = None
        try:
            enc_nli = mc.tok_nli(prefix[:4096], extractive_summary[:1024], truncation=True, padding=True, return_tensors="pt")
            enc_nli = {k:v.to(mc.device) for k,v in enc_nli.items()}
            with torch.no_grad():
                out = mc.model_nli(**enc_nli)
                probs = torch.softmax(out.logits, dim=-1)[0].cpu().numpy()
                entail_fb = float(probs[2])
        except Exception:
            entail_fb = None
        fallback = {"extractive_summary": extractive_summary, "rerank_score": float(rer_fb), "entail_prob": entail_fb}

    qa_answer = None
    if question and mc.qa_pipeline is not None:
        try:
            qa_res = mc.qa_pipeline(question=question, context=article, top_k=1)
            if isinstance(qa_res, list) and qa_res: qa_answer = qa_res[0]
            elif isinstance(qa_res, dict): qa_answer = qa_res
        except Exception:
            qa_answer = None

    out = {"generative_top3": ranked[:3], "generative_top1_entail": entail_prob,
           "fallback_used": fallback_used, "fallback": fallback, "qa_answer": qa_answer}
    return out
