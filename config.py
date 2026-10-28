"""
config.py  –  Central configuration for the English→Marathi NMT system.
All hyperparameters and file-paths live here so every other module
imports from a single source of truth.
"""

import os
import torch

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR    = os.path.join(BASE_DIR, "logs")
PLOT_DIR   = os.path.join(BASE_DIR, "plots")

for d in [DATA_DIR, MODEL_DIR, LOG_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────
# Dataset  (IIT Bombay English–Marathi corpus)
# ─────────────────────────────────────────────
# We use the CFILT / IIT-Bombay parallel corpus hosted on OPUS / HuggingFace.
# The data.py downloader fetches it automatically.
DATASET_NAME    = "cfilt/iitb-english-marathi"   # HuggingFace dataset id
SRC_LANG        = "en"
TGT_LANG        = "mr"

# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────
MAX_SEQ_LEN     = 50          # discard sentence pairs longer than this
MIN_FREQ        = 1           # include common words even with lower frequency
MAX_DATASET_SIZE = 25000      # clean randomly sampled pairs + conversational anchors
TRAIN_RATIO     = 0.80
VALID_RATIO     = 0.10
# remaining 0.10 → test

# ─────────────────────────────────────────────
# Special tokens
# ─────────────────────────────────────────────
PAD_TOKEN  = "<pad>"
SOS_TOKEN  = "<sos>"
EOS_TOKEN  = "<eos>"
UNK_TOKEN  = "<unk>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# ─────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────
EMB_DIM        = 256
HID_DIM        = 512
N_LAYERS       = 2
ENC_DROPOUT    = 0.3
DEC_DROPOUT    = 0.3
ATTENTION_TYPE = "bahdanau"   # "bahdanau" | "luong"

# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
BATCH_SIZE           = 64
N_EPOCHS             = 12
LEARNING_RATE        = 1e-3
WEIGHT_DECAY         = 1e-5
CLIP_GRAD            = 1.0          # gradient clipping max-norm
TEACHER_FORCING_INIT = 1.0          # starting teacher-forcing ratio
TEACHER_FORCING_MIN  = 0.5          # floor after decay
TEACHER_FORCING_DECAY= 0.04         # subtracted per epoch
PATIENCE             = 4            # early-stopping patience (epochs)

# LR scheduler  – ReduceLROnPlateau
LR_FACTOR   = 0.5
LR_PATIENCE = 2

# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
BEAM_SIZE    = 8     # wider beam → better quality (used by translate.py CLI)
MAX_PRED_LEN = 256   # max generated tokens during inference (NLLB can handle longer)

# ─────────────────────────────────────────────
# Hardware
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# Checkpoint filenames
# ─────────────────────────────────────────────
BEST_MODEL_PATH  = os.path.join(MODEL_DIR, "best_model.pt")
VOCAB_SRC_PATH   = os.path.join(DATA_DIR,  "vocab_src.pkl")
VOCAB_TGT_PATH   = os.path.join(DATA_DIR,  "vocab_tgt.pkl")
TRAIN_HISTORY    = os.path.join(LOG_DIR,   "train_history.json")
