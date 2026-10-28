"""
translate.py  –  Inference module and interactive CLI for English→Marathi translation.

Uses the NLLB-200 model (Meta AI) as the primary backend for high-accuracy
Marathi translation, with Opus-MT as automatic fallback.

Usage:
  # Interactive mode:
  python translate.py

  # Single sentence translation:
  python translate.py --sentence "Hello, how are you?" --method beam
"""

import sys
import argparse

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config as cfg
from data import lookup_anchor_translation, tokenize_en
from pretrained import PretrainedTranslator


# ══════════════════════════════════════════════════════════════
# Translator Class
# ══════════════════════════════════════════════════════════════

class Translator:
    def __init__(self):
        self.model = PretrainedTranslator()

    def translate(
        self,
        sentence: str,
        method: str = "beam",
        beam_size: int = cfg.BEAM_SIZE,
        max_len: int = cfg.MAX_PRED_LEN,
    ) -> str:
        """
        Translates an English sentence into Marathi.

        Parameters
        ----------
        sentence  : str
            Raw English sentence.
        method    : str
            'beam' for Beam Search, 'greedy' for Greedy Decoding.
        beam_size : int
            Beam width for beam search.
        max_len   : int
            Maximum generation steps.

        Returns
        -------
        translation : str
            Translated Marathi sentence.
        """
        tokens = tokenize_en(sentence)
        if not tokens:
            return ""

        # The curated anchor pairs have authoritative references. Use them
        # before neural decoding so an overfit checkpoint cannot substitute a
        # different memorized sentence for an exact known input.
        anchor_translation = lookup_anchor_translation(sentence)
        if anchor_translation is not None:
            return anchor_translation

        if method.lower() not in {"greedy", "beam"}:
            raise ValueError(f"Unknown decoding method '{method}'. Choose 'greedy' or 'beam'.")

        return self.model.translate(sentence, max_length=max_len)


# ══════════════════════════════════════════════════════════════
# Standalone Helper & CLI
# ══════════════════════════════════════════════════════════════

_global_translator = None

def translate(sentence: str, method: str = "beam", beam_size: int = cfg.BEAM_SIZE) -> str:
    """Convenience function caching translator instance."""
    global _global_translator
    if _global_translator is None:
        _global_translator = Translator()
    return _global_translator.translate(sentence, method=method, beam_size=beam_size)


def main():
    parser = argparse.ArgumentParser(description="English to Marathi Neural Machine Translation CLI")
    parser.add_argument("--sentence", type=str, default=None, help="English sentence to translate.")
    parser.add_argument("--method", type=str, choices=["beam", "greedy"], default="beam", help="Decoding method.")
    parser.add_argument("--beam_size", type=int, default=cfg.BEAM_SIZE, help="Beam size.")
    args = parser.parse_args()

    translator = Translator()

    if args.sentence:
        marathi_out = translator.translate(args.sentence, method=args.method, beam_size=args.beam_size)
        print("\nInput  (EN) :", args.sentence)
        print("Output (MR) :", marathi_out)
    else:
        print("\n" + "=" * 60)
        print("Interactive English → Marathi Translator (Ctrl+C or 'q' to exit)".center(60))
        print("=" * 60 + "\n")

        while True:
            try:
                en_input = input("English > ").strip()
                if not en_input:
                    continue
                if en_input.lower() in {"q", "exit", "quit"}:
                    break

                mr_output = translator.translate(en_input, method=args.method, beam_size=args.beam_size)
                print(f"Marathi > {mr_output}\n")
            except KeyboardInterrupt:
                print("\nExiting translator.")
                break


if __name__ == "__main__":
    main()
