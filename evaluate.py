"""
evaluate.py  –  Test set BLEU evaluation, loss/perplexity curve plotting,
and attention heatmap visualization.
"""

import os
import sys
import json

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import sacrebleu
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import config as cfg
from data import get_dataloaders, Vocabulary, normalize_english, tokenize_en
from model import build_model, Seq2Seq


# ══════════════════════════════════════════════════════════════
# 1. Model Loading
# ══════════════════════════════════════════════════════════════

def load_best_model(src_vocab: Vocabulary, tgt_vocab: Vocabulary) -> Seq2Seq:
    """Load model architecture and restore saved checkpoint weights."""
    model = build_model(len(src_vocab), len(tgt_vocab))
    if not os.path.exists(cfg.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found at {cfg.BEST_MODEL_PATH}. Please run train.py first."
        )

    checkpoint = torch.load(cfg.BEST_MODEL_PATH, map_location=cfg.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[evaluate] Loaded best model checkpoint from {cfg.BEST_MODEL_PATH} (Epoch {checkpoint.get('epoch', 0) + 1})")
    return model


# ══════════════════════════════════════════════════════════════
# 2. BLEU Score Calculation
# ══════════════════════════════════════════════════════════════

def calculate_bleu(
    model: Seq2Seq,
    test_loader,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    use_beam: bool = False,
    max_samples: int = 500,
) -> sacrebleu.BLEU:
    """
    Computes SacreBLEU score across the test set.
    """
    model.eval()
    hypotheses = []
    references = []

    count = 0
    print(f"[evaluate] Generating translations for BLEU calculation (beam={use_beam}, max_samples={max_samples})...")

    with torch.no_grad():
        for src_batch, tgt_batch, lengths, masks in test_loader:
            batch_size = src_batch.size(1)

            for b in range(batch_size):
                if count >= max_samples:
                    break

                # Extract single source and target
                src_seq = src_batch[:, b : b + 1].to(cfg.DEVICE)
                src_len = lengths[b : b + 1]
                src_mask = masks[b : b + 1, :].to(cfg.DEVICE)

                if use_beam:
                    pred_tokens = model.beam_search(
                        src_seq, src_len, src_mask, beam_size=cfg.BEAM_SIZE
                    )
                else:
                    pred_tokens, _ = model.greedy_decode(src_seq, src_len, src_mask)

                hyp_str = tgt_vocab.decode(pred_tokens, remove_special=True)

                # Ground truth target: strip <sos>, <eos>, <pad>
                ref_indices = tgt_batch[:, b].tolist()
                ref_str = tgt_vocab.decode(ref_indices, remove_special=True)

                hypotheses.append(hyp_str)
                references.append(ref_str)
                count += 1

            if count >= max_samples:
                break

    # sacrebleu expects references as a list of reference streams: [[ref1, ref2, ...]]
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    print("\n" + "─" * 50)
    print(f"SacreBLEU Result: {bleu.score:.2f}")
    print(f"Details: {bleu}")
    print("─" * 50)

    # Print a few examples
    print("\nSample Translations from Test Set:")
    for i in range(min(3, len(hypotheses))):
        print(f"Ref:  {references[i]}")
        print(f"Hyp:  {hypotheses[i]}")
        print()

    return bleu


# ══════════════════════════════════════════════════════════════
# 3. Training Curves Plotting
# ══════════════════════════════════════════════════════════════

def plot_training_curves(history_path: str = cfg.TRAIN_HISTORY):
    """Plots and saves loss and perplexity curves."""
    if not os.path.exists(history_path):
        print(f"[evaluate] No history file found at {history_path}. Skipping curve plots.")
        return

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-o", label="Validation Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Perplexity curve
    ax2.plot(epochs, history["train_ppl"], "b-o", label="Train PPL")
    ax2.plot(epochs, history["val_ppl"], "r-o", label="Validation PPL")
    ax2.set_title("Perplexity Curves")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Perplexity")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(cfg.PLOT_DIR, "loss_perplexity_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[evaluate] Training curves saved to {out_path}")


# ══════════════════════════════════════════════════════════════
# 4. Attention Heatmap Visualisation
# ══════════════════════════════════════════════════════════════

def plot_attention_map(
    src_tokens: list,
    pred_tokens: list,
    attention_matrix: torch.Tensor,
    save_path: str,
):
    """
    Renders an attention heatmap matrix between source and generated tokens.
    attention_matrix: (tgt_len, src_len)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Convert to numpy
    attn_np = attention_matrix.detach().cpu().numpy()

    cax = ax.matshow(attn_np, cmap="viridis")
    fig.colorbar(cax)

    # Set up labels
    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(pred_tokens)))

    ax.set_xticklabels(src_tokens, rotation=45, ha="left", fontname="DejaVu Sans")
    ax.set_yticklabels(pred_tokens, fontname="DejaVu Sans")

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.title("Attention Heatmap", pad=20)
    plt.xlabel("Source (English)")
    plt.ylabel("Target (Marathi)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def generate_attention_heatmaps(
    model: Seq2Seq,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    sample_sentences: list = None,
):
    """Generates heatmaps for a set of sample input sentences."""
    if sample_sentences is None:
        sample_sentences = [
            "Hello, how are you?",
            "She goes to school every day.",
            "I love programming in Python.",
        ]

    model.eval()
    for idx, sentence in enumerate(sample_sentences):
        tokens = tokenize_en(sentence)
        if not tokens:
            continue

        src_indices = [src_vocab.token2idx.get(t, cfg.UNK_IDX) for t in tokens]
        src_tensor = torch.tensor(src_indices, dtype=torch.long, device=cfg.DEVICE).unsqueeze(1)
        src_len = torch.tensor([len(src_indices)], dtype=torch.long)
        src_mask = (src_tensor == cfg.PAD_IDX).T

        pred_token_ids, attn_weights = model.greedy_decode(
            src_tensor, src_len, src_mask, max_len=cfg.MAX_PRED_LEN
        )

        pred_tokens = [tgt_vocab.idx2token.get(t, cfg.UNK_TOKEN) for t in pred_token_ids]
        # attn_weights: (pred_len, src_len)
        attn_matrix = attn_weights[: len(pred_tokens), : len(tokens)]

        save_path = os.path.join(cfg.PLOT_DIR, f"attention_heatmap_{idx + 1}.png")
        plot_attention_map(tokens, pred_tokens, attn_matrix, save_path)
        print(f"[evaluate] Attention heatmap saved to {save_path}")


# ══════════════════════════════════════════════════════════════
# Main Evaluation Entrypoint
# ══════════════════════════════════════════════════════════════

def evaluate():
    _, _, test_loader, src_vocab, tgt_vocab = get_dataloaders()
    model = load_best_model(src_vocab, tgt_vocab)

    # 1. Plot loss & perplexity curves
    plot_training_curves()

    # 2. Attention heatmaps
    generate_attention_heatmaps(model, src_vocab, tgt_vocab)

    # 3. Calculate SacreBLEU on test set
    calculate_bleu(model, test_loader, src_vocab, tgt_vocab, use_beam=False)


if __name__ == "__main__":
    evaluate()
