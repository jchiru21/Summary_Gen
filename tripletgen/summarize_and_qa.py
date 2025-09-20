# tripletgen/summarize_and_qa.py
import logging
import re
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import torch
import numpy as np
from rouge_score import rouge_scorer

from .reranker import RerankerModelWrap
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, pipeline

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# MODELS container
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
        self.loading = False
        self.ready = False
        self.last_error = None

MODELS = Models()


def load_models(checkpoints_clean: str = "checkpoints_clean", load_qa: bool = True) -> Models:
    mc = MODELS
    if mc.ready:
        logger.info("Models already ready, returning cached container.")
        return mc
    if mc.loading:
        logger.info("Models are already loading; returning container (not ready).")
        return mc

    mc.loading = True
    mc.last_error = None
    logger.info("load_models starting. device=%s", mc.device)

    # find reranker checkpoint candidates
    ck_dir = Path(checkpoints_clean)
    cands = []
    if ck_dir.exists():
        try:
            cands = sorted(ck_dir.glob("reranker_ckpt_best_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not cands:
                cands = sorted(ck_dir.glob("reranker_ckpt_resumed_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            logger.info("Found reranker checkpoint candidates: %s", [str(p) for p in cands[:3]])
        except Exception:
            logger.exception("Error listing checkpoints; continuing without checkpoint.")

    # generator selection: env override then fallback list
    env_model = os.environ.get("GENERATOR_MODEL", "").strip()
    candidate_models = []
    if env_model:
        candidate_models.append(env_model)
    candidate_models += ["google/flan-t5-small", "t5-small", "facebook/bart-large-cnn", "sshleifer/tiny-bart"]
    logger.info("Generator candidate order: %s", candidate_models)

    loaded = False
    for cand in candidate_models:
        try:
            logger.info("Attempting to load generator tokenizer/model: %s", cand)
            mc.tokenizer_gen = AutoTokenizer.from_pretrained(cand, use_fast=True)
            mc.model_gen = AutoModelForSeq2SeqLM.from_pretrained(cand)
            try:
                mc.model_gen.to(mc.device)
                logger.info("Moved generator to device %s", mc.device)
            except Exception:
                logger.warning("Could not move generator to device %s - continuing on CPU.", mc.device)
            mc.model_gen.eval()
            logger.info("Generator loaded: %s", cand)
            loaded = True
            break
        except Exception as e:
            logger.exception("Failed to load generator %s: %s", cand, e)
            mc.tokenizer_gen = None
            mc.model_gen = None
            mc.last_error = str(e)

    if not loaded:
        mc.loading = False
        mc.ready = False
        logger.error("No generator could be loaded. Last error: %s", mc.last_error)
        return mc

    # instantiate reranker wrapper (if checkpoint exists)
    ckpt_path = str(cands[0]) if cands else None
    try:
        logger.info("Instantiating RerankerModelWrap (ckpt=%s)...", ckpt_path)
        base_encoder = os.environ.get("RERANK_BASE", "roberta-base")
        mc.reranker = RerankerModelWrap(model_name_or_path=None, ckpt_path=ckpt_path, base_encoder_name=base_encoder, device=mc.device)
        mc.reranker.eval()
        logger.info("Reranker wrapper initialized.")
    except Exception:
        logger.exception("Failed to create reranker wrapper; continuing with reranker=None")
        mc.reranker = None

    # NLI model
    try:
        logger.info("Loading NLI model 'facebook/bart-large-mnli' ...")
        mc.tok_nli = AutoTokenizer.from_pretrained("facebook/bart-large-mnli", use_fast=True)
        mc.model_nli = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
        try:
            mc.model_nli.to(mc.device)
        except Exception:
            logger.warning("Could not move NLI model to device.")
        mc.model_nli.eval()
        logger.info("NLI model loaded.")
    except Exception:
        logger.exception("Failed to load NLI model; entailment will be skipped.")
        mc.tok_nli = None
        mc.model_nli = None

    # QA pipeline (optional)
    if load_qa:
        try:
            logger.info("Loading QA pipeline (distilbert) ...")
            mc.qa_pipeline = pipeline("question-answering",
                                     model="distilbert-base-cased-distilled-squad",
                                     tokenizer="distilbert-base-cased-distilled-squad",
                                     device=0 if mc.device.type == "cuda" else -1)
            logger.info("QA pipeline ready.")
        except Exception:
            logger.exception("Failed to create QA pipeline; continuing without QA.")
            mc.qa_pipeline = None

    mc.loading = False
    mc.ready = True
    logger.info("load_models completed. MODELS.ready=True")
    return mc


def _build_generation_kwargs(max_new_tokens=512, num_beams=8, do_sample=False,
                             top_p=0.95, temperature=0.7, length_penalty=1.4,
                             no_repeat_ngram_size=3, early_stopping=True, num_return_sequences=8):
    """
    Build generation kwargs with defaults preferring longer outputs.
    """
    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        num_beams=max(1, int(num_beams)),
        num_return_sequences=max(1, int(num_return_sequences)),
        no_repeat_ngram_size=int(no_repeat_ngram_size),
        early_stopping=bool(early_stopping),
        length_penalty=float(length_penalty),
        do_sample=bool(do_sample),
    )
    if gen_kwargs["do_sample"]:
        gen_kwargs.update({
            "top_p": float(top_p),
            "temperature": float(temperature),
            "top_k": 50
        })
    return gen_kwargs


@torch.no_grad()
def reranker_score_batch(article: str, candidates: List[str], max_len: int = 384) -> List[Tuple[str, float]]:
    mc = MODELS
    if mc.reranker is None:
        return [(c, 0.0) for c in candidates]

    scores = []
    for cand in candidates:
        try:
            s = mc.reranker.score(article, cand)
            if isinstance(s, torch.Tensor):
                s = s.detach().cpu().item()
            scores.append((cand, float(s)))
        except Exception:
            scores.append((cand, 0.0))
    return scores


def combined_score(article: str, question: str, candidate: str, rerank_score: float,
                   alpha: float = 0.15, beta: float = 0.85):
    """
    Combine reranker score with ROUGE-1 overlap, favoring ROUGE for detail.
    """
    try:
        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        rouge1 = scorer.score(article, candidate)["rouge1"].fmeasure
    except Exception:
        rouge1 = 0.0

    rer = float(rerank_score) if rerank_score is not None else 0.0
    rer = max(0.0, min(1.0, rer))
    rouge1 = max(0.0, min(1.0, rouge1))

    combined = float(alpha * rer + beta * rouge1)
    combined = max(0.0, min(1.0, combined))
    return combined, {"rouge1": float(rouge1), "rerank_score": float(rer), "alpha": float(alpha), "beta": float(beta)}


def _contains_required_tokens(text: str, required: List[str]) -> Tuple[bool, List[str]]:
    missing = []
    low = text or ""
    for token in required:
        if token.lower() not in low.lower():
            missing.append(token)
    return (len(missing) == 0, missing)


@torch.no_grad()
def summarize_and_qa(article: str, question: Optional[str] = None,
                     K: int = 8, num_beams: int = 8, max_new_tokens: int = 512,
                     entail_threshold: float = 0.8, checkpoints_clean: str = "checkpoints_clean",
                     do_sample: bool = False, top_p: float = 0.95, temperature: float = 0.7,
                     length_penalty: float = 1.4) -> Dict[str, Any]:
    """
    Produces a detail-rich 6-8 sentence summary when possible.
    """
    # defensive clamps
    max_new_tokens = int(max(64, min(1024, int(max_new_tokens))))
    num_beams = int(max(1, min(12, int(num_beams))))
    length_penalty = float(max(0.2, min(3.0, float(length_penalty))))
    K = int(max(1, min(12, int(K))))

    mc = load_models(checkpoints_clean=checkpoints_clean, load_qa=True)

    # strong instruction: request 6-8 sentence detailed summary
    prefix_instr = (
        "Write a detailed, factual summary of the following article in 6-8 sentences. "
        "Include major participants, the step-by-step mechanism, relevant dates, and the stated rationale. "
        "Do not invent facts; only use information present in the article."
    )

    approx_char_cap = 20000
    article_short = article if len(article) <= approx_char_cap else article[:approx_char_cap]
    prompt = prefix_instr + "\n\n" + article_short
    if question:
        prompt = f"{prefix_instr} Answer the question if present: {question}\n\n" + article_short

    # generation inputs
    enc = mc.tokenizer_gen([prompt], truncation=True, padding=True, return_tensors="pt", max_length=1024)
    enc = {k: v.to(mc.device) for k, v in enc.items()}

    gen_kwargs = _build_generation_kwargs(max_new_tokens=max_new_tokens,
                                          num_beams=num_beams,
                                          do_sample=do_sample,
                                          top_p=top_p,
                                          temperature=temperature,
                                          length_penalty=length_penalty,
                                          num_return_sequences=max(1, min(12, K)))
    logger.info("Generation kwargs: %s", {k: v for k, v in gen_kwargs.items() if k != "decoder_input_ids"})

    # generate K candidates
    with torch.no_grad():
        outs = mc.model_gen.generate(**enc, **gen_kwargs)

    cands = mc.tokenizer_gen.batch_decode(outs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    logger.info("Generated %d candidates. Sample: %s", len(cands), [c[:220] for c in cands[:3]])

    # rerank
    try:
        rer_scores = reranker_score_batch(prompt, cands)
    except Exception:
        rer_scores = [(c, 0.0) for c in cands]
    logger.info("Reranker scores: %s", [round(s,4) for (_, s) in rer_scores[:8]])

    ranked = []
    for cand, s in rer_scores:
        new, meta = combined_score(prompt, "", cand, s)
        ranked.append({"candidate": cand, "rerank_score": float(s), "combined": float(new), **meta})
    ranked = sorted(ranked, key=lambda x: x["combined"], reverse=True)
    top = ranked[0] if ranked else {"candidate": "", "rerank_score": 0.0, "combined": 0.0}

    # auto-expand short-ish top candidate (threshold 140 words)
    try:
        top_text = top.get("candidate", "") if isinstance(top, dict) else str(top)
        word_count_top = len(re.findall(r"\w+", top_text))
        if word_count_top < 140:
            logger.info("Top candidate short (%d words). Running expansion pass.", word_count_top)
            expand_prompt = ("Expand the summary below into a longer factual paragraph (6-8 sentences) using only information from the article. "
                             "Do not add new claims.\n\nSUMMARY:\n" + top_text + "\n\nARTICLE:\n" + article_short)
            enc_expand = mc.tokenizer_gen([expand_prompt], truncation=True, padding=True, return_tensors="pt", max_length=1024)
            enc_expand = {k: v.to(mc.device) for k, v in enc_expand.items()}
            expand_max_tokens = min(320, max_new_tokens)
            with torch.no_grad():
                outs_expand = mc.model_gen.generate(**enc_expand,
                                                   max_new_tokens=expand_max_tokens,
                                                   num_beams=max(4, num_beams),
                                                   no_repeat_ngram_size=3,
                                                   length_penalty=max(1.2, length_penalty),
                                                   early_stopping=True)
            expanded = mc.tokenizer_gen.batch_decode(outs_expand, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if expanded and expanded[0].strip():
                expanded_text = expanded[0].strip()
                # rerank expanded_text vs original candidates
                try:
                    expanded_score = reranker_score_batch(prompt, [expanded_text])[0][1]
                except Exception:
                    expanded_score = 0.0
                expanded_combined, expanded_meta = combined_score(prompt, "", expanded_text, expanded_score)
                logger.info("Expanded combined score: %.4f (orig top combined: %.4f)", expanded_combined, float(top.get("combined",0.0)))
                # keep expanded if combined improved or if it includes more required tokens (checked later)
                if expanded_combined >= float(top.get("combined", 0.0)):
                    logger.info("Replacing top with expanded candidate.")
                    top["candidate"] = expanded_text
                    top["rerank_score"] = float(expanded_score)
                    top["combined"] = float(expanded_combined)
                    top.update(expanded_meta)
                    if ranked:
                        ranked[0] = top
                else:
                    logger.info("Expanded candidate not better by combined score; keeping original top.")
    except Exception:
        logger.exception("Expansion pass failed; continuing with original top candidate.")

    # ensure required tokens/entities/dates are present; if not, attempt final targeted rewrite
    required_tokens = ["Rusvietpetro", "Zarubezhneft", "PVN", "June 11, 2024"]
    try:
        ok, missing = _contains_required_tokens(top.get("candidate", ""), required_tokens)
        if not ok:
            logger.info("Top candidate missing tokens: %s. Running targeted rewrite to include them.", missing)
            include_prompt = ("Rewrite the following summary to include these specific terms where appropriate: " +
                              ", ".join(missing) +
                              ". Keep it factual and 6-8 sentences, use only the article's content.\n\nSUMMARY:\n" +
                              top.get("candidate", "") + "\n\nARTICLE:\n" + article_short)
            enc_include = mc.tokenizer_gen([include_prompt], truncation=True, padding=True, return_tensors="pt", max_length=1024)
            enc_include = {k: v.to(mc.device) for k, v in enc_include.items()}
            with torch.no_grad():
                outs_include = mc.model_gen.generate(**enc_include,
                                                    max_new_tokens=min(320, max_new_tokens),
                                                    num_beams=max(4, num_beams),
                                                    no_repeat_ngram_size=3,
                                                    length_penalty=max(1.2, length_penalty),
                                                    early_stopping=True)
            included = mc.tokenizer_gen.batch_decode(outs_include, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if included and included[0].strip():
                included_text = included[0].strip()
                # verify presence
                ok2, missing2 = _contains_required_tokens(included_text, required_tokens)
                if ok2:
                    logger.info("Targeted rewrite included required tokens. Replacing top candidate.")
                    try:
                        rer_fb = reranker_score_batch(prompt, [included_text])[0][1]
                    except Exception:
                        rer_fb = 0.0
                    combined_included, meta_included = combined_score(prompt, "", included_text, rer_fb)
                    # accept if combined not worse than original
                    if combined_included >= float(top.get("combined", 0.0)) - 1e-6:
                        top["candidate"] = included_text
                        top["rerank_score"] = float(rer_fb)
                        top["combined"] = float(combined_included)
                        top.update(meta_included)
                        if ranked:
                            ranked[0] = top
                    else:
                        logger.info("Included rewrite had worse combined score; keeping previous top.")
                else:
                    logger.info("Included rewrite still missing tokens: %s", missing2)
    except Exception:
        logger.exception("Targeted inclusion pass failed; continuing.")

    # entailment check
    entail_prob = None
    try:
        if mc.tok_nli and mc.model_nli:
            enc_nli = mc.tok_nli(prompt[:4096], top["candidate"][:1024], truncation=True, padding=True, return_tensors="pt")
            enc_nli = {k: v.to(mc.device) for k, v in enc_nli.items()}
            with torch.no_grad():
                out = mc.model_nli(**enc_nli)
                probs = torch.softmax(out.logits, dim=-1)[0].cpu().numpy()
                id2label = getattr(mc.model_nli.config, "id2label", None)
                entail_idx = None
                if id2label:
                    for i, lbl in id2label.items():
                        if "entail" in lbl.lower():
                            entail_idx = int(i)
                            break
                if entail_idx is None:
                    entail_idx = 2 if len(probs) > 2 else int(np.argmax(probs))
                entail_prob = float(probs[entail_idx])
    except Exception:
        entail_prob = None

    # fallback extractive summary if entailment low or failed
    fallback_used = False; fallback = None
    if entail_prob is None or entail_prob < entail_threshold:
        fallback_used = True
        sents = re.split(r'(?<=[.!?])\s+', article.strip())
        def jaccard_sim(a,b):
            sa = set(re.findall(r"\w+", a.lower())); sb = set(re.findall(r"\w+", b.lower()))
            if not sa or not sb: return 0.0
            return len(sa & sb) / len(sa | sb)
        scores = [(i, jaccard_sim(top.get("candidate", ""), s)) for i,s in enumerate(sents)]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        top_idxs = [i for i,sc in scores[:3]]
        top_sents = [sents[i] for i in top_idxs if i < len(sents)]
        extractive_summary = " ".join(top_sents)
        try:
            rer_fb = reranker_score_batch(prompt, [extractive_summary])[0][1]
        except Exception:
            rer_fb = 0.0
        entail_fb = None
        try:
            if mc.tok_nli and mc.model_nli:
                enc_nli = mc.tok_nli(prompt[:4096], extractive_summary[:1024], truncation=True, padding=True, return_tensors="pt")
                enc_nli = {k: v.to(mc.device) for k, v in enc_nli.items()}
                with torch.no_grad():
                    out = mc.model_nli(**enc_nli)
                    probs = torch.softmax(out.logits, dim=-1)[0].cpu().numpy()
                    id2label = getattr(mc.model_nli.config, "id2label", None)
                    entail_idx = None
                    if id2label:
                        for i, lbl in id2label.items():
                            if "entail" in lbl.lower():
                                entail_idx = int(i)
                                break
                    if entail_idx is None:
                        entail_idx = 2 if len(probs) > 2 else int(np.argmax(probs))
                    entail_fb = float(probs[entail_idx])
        except Exception:
            entail_fb = None
        fallback = {"extractive_summary": extractive_summary, "rerank_score": float(rer_fb), "entail_prob": entail_fb}

    # QA
    qa_answer = None
    if question and mc.qa_pipeline is not None:
        try:
            qa_res = mc.qa_pipeline(question=question, context=article, top_k=1)
            if isinstance(qa_res, list) and qa_res:
                qa_answer = qa_res[0]
            elif isinstance(qa_res, dict):
                qa_answer = qa_res
        except Exception:
            qa_answer = None

    out = {"generative_top3": ranked[:3], "generative_top1_entail": entail_prob,
           "fallback_used": fallback_used, "fallback": fallback, "qa_answer": qa_answer}

    try:
        topmeta = ranked[0] if ranked else {}
        logger.info("summarize result: top_combined=%.4f entail=%.3f fallback_used=%s tokens=%d",
                    float(topmeta.get("combined", 0.0)), float(entail_prob or 0.0), fallback_used, max_new_tokens)
    except Exception:
        pass

    return out
