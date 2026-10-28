# Neural Machine Translation (English → Marathi)

A Streamlit English-to-Marathi translator using **Meta's NLLB-200-distilled-600M** as the primary backend for high-accuracy Marathi translation, with Helsinki-NLP Opus-MT as automatic fallback. The original LSTM training code is retained for experimentation and architecture study.

---

## 📁 Project Structure

```
nmt_lstm/
├── config.py          # Central hyperparameters, tokens, device, and file paths
├── data.py            # Dataset loader (IITB corpus), text preprocessor, vocabularies & DataLoader
├── model.py           # Stacked BiLSTM Encoder, Bahdanau/Luong Attention, Decoder, Seq2Seq
├── train.py           # Training loop, LR scheduler, gradient clipping, early stopping & logs
├── evaluate.py        # SacreBLEU evaluation, loss/perplexity curve plots, attention heatmaps
├── translate.py       # Inference class, translate() function, and interactive CLI
├── pretrained.py      # Pretrained OPUS-MT translation backend
├── requirements.txt   # Python package dependencies
└── README.md          # Setup guide, run instructions, and improvement suggestions
```

---

## ⚙️ Architecture & Features

- **Encoder**: 2-layer stacked bidirectional LSTM (`hidden_size=512`, `embedding_size=256`, `dropout=0.3`), packed sequences for variable length.
- **Attention**: Bahdanau (additive) attention with masking to zero out `<pad>` tokens (switchable to Luong attention in [config.py](file:///c:/Users/DEEP/OneDrive/Documents/Machine%20Learning%20Projects/Deep%20Learning/LStm%20college%20Project/nmt_lstm/config.py)).
- **Decoder**: 2-layer stacked LSTM receiving previous embedding and attention context vector, with multi-layer skip projection to target vocabulary.
- **Teacher Forcing**: Starts at 1.0 and decays per epoch down to a minimum ratio of 0.5.
- **Decoding Options**:
  - Beam Search decoding (`beam_size=5`)
  - Greedy decoding
- **Training Enhancements**:
  - Adam optimizer + `ReduceLROnPlateau` learning rate scheduler.
  - Gradient norm clipping at 1.0 to prevent exploding gradients.
  - Cross-Entropy loss ignoring `<pad>` index.
  - Model checkpointing on lowest validation loss.
  - Early stopping with configurable patience.
- **Evaluation**:
  - Standardized corpus-level BLEU score calculation via `sacrebleu`.
  - Loss and perplexity tracking plotted to disk.
  - Attention matrix heatmap visualization between English and Marathi tokens.
- **High-Accuracy Inference**:
  - Primary model: `facebook/nllb-200-distilled-600M` (Meta AI NLLB-200)
  - Fallback model: `Helsinki-NLP/opus-mt-en-mr` (MarianMT)
  - 8-beam search with length penalty and repetition penalty
  - Smart pre/post-processing (unicode normalization, punctuation cleanup)

---

## 🚀 Installation & Setup

### 1. Clone & Navigate
```bash
cd nmt_lstm
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 Running the Pipeline

### 1. Train the Model
Fetches the IIT Bombay English–Marathi dataset (via Hugging Face Datasets), builds vocabularies, splits data into 80/10/10, and runs the training loop:
```bash
python train.py
```
*Note: If Hugging Face is unreachable or runs offline, a built-in multilingual fallback corpus is automatically used so training can proceed without errors.*

### 2. Evaluate & Plot Results
Evaluates the best checkpoint against the test set with SacreBLEU, exports loss/perplexity curves to `plots/loss_perplexity_curves.png`, and renders attention heatmaps:
```bash
python evaluate.py
```

### 3. Translate (CLI / Interactive)

The first run downloads **`facebook/nllb-200-distilled-600M`** (~2.4 GB) from Hugging Face.
An internet connection is required for the first run; later runs use the local Hugging Face cache.
If NLLB fails to load, `Helsinki-NLP/opus-mt-en-mr` is used automatically as fallback.

**Interactive CLI:**
```bash
python translate.py
```

**Single Sentence Translation:**
```bash
python translate.py --sentence "The weather is very nice today." --method beam --beam_size 5
```

Exact phrases from the curated conversational training anchors use their
authoritative Marathi reference translation before neural decoding. This avoids
the compact checkpoint replacing a known phrase with another memorized phrase;
unseen inputs continue through the LSTM decoder.

---

### 4. Streamlit Web Application

Launch the interactive UI to translate sentences and visualize attention heatmaps in real time:

```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your web browser.

---

## 💡 Notes on Improving Results

1. **Subword Tokenization (BPE / SentencePiece)**:
   - Word-level tokenization can suffer from high out-of-vocabulary (OOV) rates, especially in morphologically rich languages like Marathi. Using SentencePiece with a 16k–32k vocabulary merges subwords and drastically cuts down `<unk>` tokens.
2. **Transformer Architecture**:
   - Replacing recurrence with multi-head self-attention removes sequential bottlenecks, scales better to long sentences, and is faster to train in parallel.
3. **Pre-trained Embeddings & Models**:
   - Initializing embeddings with pre-trained fastText or fine-tuning multilingual encoders (e.g., IndicBERT, mBART-50, or NLLB-200) significantly improves rare-word handling.
4. **Back-Translation**:
   - Synthesizing parallel data from abundant monolingual Marathi text (translating Marathi → English with an inverse model and adding to training) increases model robustness and BLEU scores.
5. **Length Penalty in Beam Search**:
   - Applying length normalization $\frac{(5 + |Y|)^\alpha}{(5 + 1)^\alpha}$ prevents beam search from biasing towards shorter sequences.
