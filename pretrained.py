"""
pretrained.py  –  English-to-Marathi translation backend with two model tiers.

  FAST   : Helsinki-NLP/opus-mt-en-mr    – ~300 MB, loads in seconds.
  ACCURATE: facebook/nllb-200-distilled-600M – ~2.4 GB, much better Marathi quality.

The caller selects which tier to use; PretrainedTranslator accepts a `model_tier`
argument ("fast" | "accurate") so the Streamlit UI can let the user choose.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# ──────────────────────────────────────────────────────────────
# Model identifiers
# ──────────────────────────────────────────────────────────────
NLLB_MODEL_NAME  = "facebook/nllb-200-distilled-600M"
OPUS_MODEL_NAME  = "Helsinki-NLP/opus-mt-en-mr"

NLLB_SRC_LANG = "eng_Latn"
NLLB_TGT_LANG = "mar_Deva"


# ──────────────────────────────────────────────────────────────
# Text pre/post-processing helpers
# ──────────────────────────────────────────────────────────────

def _preprocess_input(text: str) -> str:
    """Normalize unicode, collapse whitespace, fix punctuation spacing."""
    text = unicodedata.normalize("NFC", text.strip())
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", text)
    return text.strip()


def _postprocess_output(text: str) -> str:
    """Clean Devanagari output: NFC, collapse spaces, fix danda spacing."""
    text = unicodedata.normalize("NFC", text.strip())
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s([।,!?])", r"\1", text)
    return text.strip()


# ──────────────────────────────────────────────────────────────
# Fast Opus-MT backend  (default – loads in seconds)
# ──────────────────────────────────────────────────────────────

class OpusMTTranslator:
    """
    Lightweight MarianMT model (Helsinki-NLP/opus-mt-en-mr).
    ~300 MB, loads in a few seconds, good baseline quality.
    """
    model_label = "Opus-MT en→mr (Helsinki-NLP)"
    model_name  = OPUS_MODEL_NAME

    def __init__(self) -> None:
        print(f"[pretrained] Loading Opus-MT: {OPUS_MODEL_NAME} …")
        self.tokenizer = AutoTokenizer.from_pretrained(OPUS_MODEL_NAME)
        self.model     = AutoModelForSeq2SeqLM.from_pretrained(OPUS_MODEL_NAME)
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"[pretrained] Opus-MT ready on {self.device}.")

    @torch.inference_mode()
    def translate(self, sentence: str, max_length: int = 256) -> str:
        sentence = _preprocess_input(sentence)
        if not sentence:
            return ""
        encoded = self.tokenizer(
            sentence, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        generated = self.model.generate(
            **encoded,
            num_beams=5,
            length_penalty=1.1,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            early_stopping=True,
            max_new_tokens=max_length,
        )
        return _postprocess_output(
            self.tokenizer.decode(generated[0], skip_special_tokens=True)
        )

    def translate_with_tokens(
        self, sentence: str, max_length: int = 256
    ) -> Tuple[str, List[str]]:
        t = self.translate(sentence, max_length)
        return t, t.split()


# ──────────────────────────────────────────────────────────────
# Accurate NLLB-200 backend  (high quality, larger download)
# ──────────────────────────────────────────────────────────────

class NLLBTranslator:
    """
    Meta's NLLB-200-distilled-600M.
    ~2.4 GB, significantly better Marathi morphology and accuracy.
    """
    model_label = "NLLB-200 Distilled 600M (Meta AI)"
    model_name  = NLLB_MODEL_NAME

    def __init__(self) -> None:
        print(f"[pretrained] Loading NLLB-200: {NLLB_MODEL_NAME} …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            NLLB_MODEL_NAME, src_lang=NLLB_SRC_LANG
        )
        self.model  = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self._mar_bos_id = self.tokenizer.convert_tokens_to_ids(NLLB_TGT_LANG)
        print(f"[pretrained] NLLB-200 ready on {self.device}.")

    @torch.inference_mode()
    def translate(self, sentence: str, max_length: int = 256) -> str:
        sentence = _preprocess_input(sentence)
        if not sentence:
            return ""
        encoded = self.tokenizer(
            sentence, return_tensors="pt",
            truncation=True, max_length=512, padding=True,
        ).to(self.device)
        generated = self.model.generate(
            **encoded,
            forced_bos_token_id=self._mar_bos_id,
            num_beams=6,
            length_penalty=1.2,
            no_repeat_ngram_size=4,
            repetition_penalty=1.3,
            early_stopping=True,
            max_new_tokens=max_length,
        )
        return _postprocess_output(
            self.tokenizer.decode(generated[0], skip_special_tokens=True)
        )

    def translate_with_tokens(
        self, sentence: str, max_length: int = 256
    ) -> Tuple[str, List[str]]:
        t = self.translate(sentence, max_length)
        return t, t.split()


# ──────────────────────────────────────────────────────────────
# Public interface
# ──────────────────────────────────────────────────────────────

class PretrainedTranslator:
    """
    Public translation interface.

    Parameters
    ----------
    model_tier : "fast" | "accurate"
        "fast"     → Opus-MT  (~300 MB, loads in seconds)
        "accurate" → NLLB-200 (~2.4 GB, much better Marathi quality)
    """

    def __init__(self, model_tier: str = "fast") -> None:
        if model_tier == "accurate":
            try:
                self._backend = NLLBTranslator()
            except Exception as exc:
                print(f"[pretrained] NLLB failed ({exc}). Falling back to Opus-MT.")
                self._backend = OpusMTTranslator()
        else:
            self._backend = OpusMTTranslator()

        self.model_name  = self._backend.model_name
        self.model_label = self._backend.model_label

    def translate(self, sentence: str, max_length: int = 256) -> str:
        return self._backend.translate(sentence, max_length=max_length)

    def translate_with_tokens(
        self, sentence: str, max_length: int = 256
    ) -> Tuple[str, List[str]]:
        return self._backend.translate_with_tokens(sentence, max_length=max_length)
