# tripletgen/reranker.py
import math
import logging
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class _NotebookHead(nn.Module):
    """
    Matches your notebook's small MLP head:
      Linear(h, h//2) -> ReLU -> Dropout(0.1) -> Linear(h//2, 1)
    Input: pooled CLS vector.
    Output: single logit (float).
    """
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        mid = max(64, hidden_size // 2)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, 1)
        )

    def forward(self, pooled):
        # pooled: (B, H)
        return self.net(pooled).squeeze(-1)  # -> (B,)


class CrossEncoderNotebook(nn.Module):
    """
    Cross-encoder built like the notebook:
      - encoder (AutoModel, e.g., roberta-base)
      - CLS pooling: last_hidden_state[:,0,:]
      - small notebook head producing a single logit
    """
    def __init__(self, base_model_name: str = "roberta-base", device: Optional[torch.device] = None):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = getattr(self.encoder.config, "hidden_size", 768)
        self.head = _NotebookHead(hidden_size)
        self.to(self.device)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # CLS pooling using token 0 (not pooler_output)
        cls_vec = out.last_hidden_state[:, 0, :]  # (B, H)
        logits = self.head(cls_vec)  # (B,)
        return logits


class RerankerModelWrap:
    """
    Adapter that matches your notebook's reranker semantics.

    Behavior:
    - If model_name_or_path is a HF seq-class model, loads AutoModelForSequenceClassification and tokenizer.
    - Else, if ckpt_path is a .pt state_dict (as saved in your notebook), instantiate CrossEncoderNotebook(base_encoder)
      and try to smart-load weights (matching keys / shapes).
    - Otherwise, instantiate a fresh CrossEncoderNotebook(base_encoder).
    - .score(article, candidate) returns a float in [0,1] (sigmoid of logit or softmax-based for seq-class).
    """
    def __init__(self, model_name_or_path: Optional[str] = None, ckpt_path: Optional[str] = None,
                 base_encoder_name: str = "roberta-base", device: Optional[torch.device] = None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name_or_path = model_name_or_path
        self.ckpt_path = ckpt_path
        self.base_encoder_name = base_encoder_name
        self.tokenizer = None
        self.model = None
        self._init()

    def _init(self):
        # Strategy 1: try a HF AutoModelForSequenceClassification
        if self.model_name_or_path:
            try:
                logger.info("Reranker: trying HF AutoModelForSequenceClassification: %s", self.model_name_or_path)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path)
                self.model.to(self.device)
                self.model.eval()
                logger.info("Reranker: loaded HF seq-class model.")
                return
            except Exception:
                logger.exception("Reranker: loading HF seq-class failed, will try checkpoint or notebook style.")

        # Strategy 2: if ckpt_path provided and is .pt -> try to load state_dict into CrossEncoderNotebook
        if self.ckpt_path and str(self.ckpt_path).endswith(".pt"):
            try:
                logger.info("Reranker: attempting to load state_dict from %s", self.ckpt_path)
                sd = torch.load(self.ckpt_path, map_location="cpu")
                # Support wrapped objs: {'state_dict': {...}} or plain state_dict
                if isinstance(sd, dict) and ("state_dict" in sd or "model_state" in sd or "model" in sd):
                    if "state_dict" in sd:
                        state = sd["state_dict"]
                    elif "model_state" in sd:
                        state = sd["model_state"]
                    elif "model" in sd:
                        state = sd["model"]
                    else:
                        state = sd
                else:
                    state = sd

                # instantiate fresh notebook cross-encoder
                model = CrossEncoderNotebook(base_model_name=self.base_encoder_name, device=self.device)
                model_sd = model.state_dict()

                # smart key matching: keep only matching keys with same shape
                loadable = {}
                loaded_keys = 0
                skipped_keys = 0
                for k, v in state.items():
                    if k in model_sd and isinstance(v, torch.Tensor) and v.shape == model_sd[k].shape:
                        loadable[k] = v
                        loaded_keys += 1
                    else:
                        skipped_keys += 1

                if loaded_keys > 0:
                    model_sd.update(loadable)
                    model.load_state_dict(model_sd)
                    logger.info("Reranker: loaded %d matching keys from checkpoint, skipped %d keys", loaded_keys, skipped_keys)
                else:
                    logger.warning("Reranker: no matching keys found in checkpoint; using fresh model (skipped %d keys)", skipped_keys)
                model.to(self.device)
                model.eval()
                self.model = model
                # tokenizer from base encoder
                self.tokenizer = AutoTokenizer.from_pretrained(self.base_encoder_name, use_fast=True)
                return
            except Exception:
                logger.exception("Reranker: failed to instantiate CrossEncoder from checkpoint; falling back.")

        # Strategy 3: try ckpt_path as HF dir for seq-class (if not .pt)
        if self.ckpt_path:
            try:
                logger.info("Reranker: trying HF dir from ckpt_path: %s", self.ckpt_path)
                self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt_path, use_fast=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.ckpt_path)
                self.model.to(self.device)
                self.model.eval()
                logger.info("Reranker: loaded HF seq-class from ckpt_path.")
                return
            except Exception:
                logger.exception("Reranker: ckpt_path not a HF seq-class dir; will fallback.")

        # Strategy 4: fresh CrossEncoderNotebook from base encoder
        try:
            logger.info("Reranker: instantiating fresh CrossEncoderNotebook using base encoder %s", self.base_encoder_name)
            model = CrossEncoderNotebook(base_model_name=self.base_encoder_name, device=self.device)
            model.to(self.device)
            model.eval()
            self.model = model
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_encoder_name, use_fast=True)
            logger.info("Reranker: fresh CrossEncoderNotebook ready.")
            return
        except Exception:
            logger.exception("Reranker: failed to instantiate CrossEncoderNotebook. Reranker disabled.")
            self.model = None
            self.tokenizer = None

    def to(self, device: Union[torch.device, str]):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        if self.model is not None:
            try:
                self.model.to(device)
            except Exception:
                logger.exception("Reranker: failed to move model to device %s", device)

    def eval(self):
        if self.model is not None:
            try:
                self.model.eval()
            except Exception:
                pass

    @torch.no_grad()
    def score(self, a: str, b: str) -> float:
        """
        Score pair (a,b) -> probability in [0,1].
        Tokenizes with tokenizer and passes through model.
        If model is CrossEncoderNotebook (outputs logits), use sigmoid(logit).
        If model is AutoModelForSequenceClassification, use softmax and return max prob.
        If no model/tokenizer available -> lexical Jaccard fallback.
        """
        if self.model is None or self.tokenizer is None:
            # lexical fallback (precision-like)
            a_tokens = set(t.lower() for t in str(a).split())
            b_tokens = set(t.lower() for t in str(b).split())
            if not a_tokens:
                return 0.0
            ov = len(a_tokens & b_tokens) / max(1, len(a_tokens))
            return float(max(0.0, min(1.0, ov)))

        try:
            inputs = self.tokenizer(a, b, truncation=True, padding=True, return_tensors="pt", max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model(**inputs)
            # if model returns logits (CrossEncoderNotebook returns logits tensor)
            if isinstance(out, torch.Tensor):
                logit = out.detach().cpu()
                prob = torch.sigmoid(logit).numpy()
                return float(prob[0]) if hasattr(prob, '__len__') else float(prob)
            # if it's AutoModelForSequenceClassification -> out is ModelOutput with logits
            elif hasattr(out, "logits"):
                logits = out.logits.detach().cpu()
                probs = torch.softmax(logits, dim=-1).numpy()[0]
                return float(max(probs))
            else:
                # unexpected return type; fallback to lexical
                logger.warning("Reranker: unexpected model output type %s; using lexical fallback", type(out))
                a_tokens = set(t.lower() for t in str(a).split())
                b_tokens = set(t.lower() for t in str(b).split())
                if not a_tokens:
                    return 0.0
                ov = len(a_tokens & b_tokens) / max(1, len(a_tokens))
                return float(max(0.0, min(1.0, ov)))
        except Exception:
            logger.exception("Reranker: scoring failed at runtime; using lexical fallback.")
            a_tokens = set(t.lower() for t in str(a).split())
            b_tokens = set(t.lower() for t in str(b).split())
            if not a_tokens:
                return 0.0
            ov = len(a_tokens & b_tokens) / max(1, len(a_tokens))
            return float(max(0.0, min(1.0, ov)))
