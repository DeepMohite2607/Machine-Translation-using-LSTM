"""
model.py  –  Encoder, Attention, Decoder, and Seq2Seq wrapper.

Architecture
────────────
  Encoder  :  Embedding → 2-layer stacked LSTM → encoder outputs + hidden
  Attention:  Bahdanau (additive) attention  OR  Luong (dot/general) attention
  Decoder  :  Embedding → Attention → 2-layer stacked LSTM → Linear projection
  Seq2Seq  :  Encoder + Decoder glued together with teacher forcing

All sub-modules are documented inline.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


# ══════════════════════════════════════════════════════════════
# Encoder
# ══════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """
    Stacked bidirectional LSTM encoder.

    Input  : (src_len, batch)  integer indices
    Output :
        encoder_outputs : (src_len, batch, hid_dim)  – all time-step outputs
        hidden          : (n_layers, batch, hid_dim)  – last-step hidden state
        cell            : (n_layers, batch, hid_dim)  – last-step cell state

    Because the LSTM is bidirectional, the raw outputs and states have
    2 × hid_dim / 2 dimensions. We project them down to hid_dim with a
    linear layer so the decoder does not need to know about bidirectionality.
    """

    def __init__(
        self,
        input_dim:   int,        # source vocabulary size
        emb_dim:     int = cfg.EMB_DIM,
        hid_dim:     int = cfg.HID_DIM,
        n_layers:    int = cfg.N_LAYERS,
        dropout:     float = cfg.ENC_DROPOUT,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            input_dim, emb_dim, padding_idx=cfg.PAD_IDX
        )

        # Bidirectional to capture context from both directions.
        # We use half the hidden size so that after concatenating fwd+bwd
        # the dimension matches hid_dim expected by the decoder.
        self.rnn = nn.LSTM(
            emb_dim,
            hid_dim // 2,
            num_layers     = n_layers,
            dropout        = dropout if n_layers > 1 else 0.0,
            bidirectional  = True,
            batch_first    = False,   # (seq, batch, feature)
        )

        self.dropout = nn.Dropout(dropout)

        # Project concatenated bidirectional hidden/cell → hid_dim
        self.fc_hidden = nn.Linear(hid_dim, hid_dim)
        self.fc_cell   = nn.Linear(hid_dim, hid_dim)

    def forward(
        self,
        src: torch.Tensor,         # (src_len, batch)
        src_lengths: torch.Tensor, # (batch,)
    ):
        embedded = self.dropout(self.embedding(src))   # (src_len, B, emb)

        # PackedSequence → handles variable-length inputs efficiently
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), enforce_sorted=True
        )
        packed_out, (hidden, cell) = self.rnn(packed)
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out)
        # encoder_outputs: (src_len, B, hid_dim)   [fwd||bwd concatenated]
        # hidden:          (n_layers*2, B, hid_dim//2)

        # ── Merge bidirectional layers ──────────────────────────────
        # hidden has shape (num_directions * n_layers, B, hid//2)
        # Reshape to (n_layers, 2, B, hid//2), concatenate on dim=1 → (n_layers, B, hid)
        n_layers = hidden.size(0) // 2
        hidden = hidden.view(n_layers, 2, hidden.size(1), hidden.size(2))
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)  # (n_layers, B, hid)
        cell   = cell.view(n_layers, 2, cell.size(1), cell.size(2))
        cell   = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)

        # Pass through linear projection + tanh for a better initialisation
        hidden = torch.tanh(self.fc_hidden(hidden))
        cell   = torch.tanh(self.fc_cell(cell))

        return encoder_outputs, hidden, cell


# ══════════════════════════════════════════════════════════════
# Attention mechanisms
# ══════════════════════════════════════════════════════════════

class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau) attention.

        score(s, h) = v^T · tanh(W_1·s + W_2·h)

    s : decoder hidden state  (batch, hid_dim)
    h : encoder output        (src_len, batch, hid_dim)

    Returns
    -------
    context : (batch, hid_dim)  – weighted sum of encoder outputs
    weights : (batch, src_len)  – attention distribution (for visualisation)
    """

    def __init__(self, hid_dim: int = cfg.HID_DIM):
        super().__init__()
        self.W_decoder  = nn.Linear(hid_dim, hid_dim, bias=False)
        self.W_encoder  = nn.Linear(hid_dim, hid_dim, bias=False)
        self.v          = nn.Linear(hid_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden:   torch.Tensor,  # (batch, hid_dim)
        encoder_outputs:  torch.Tensor,  # (src_len, batch, hid_dim)
        src_mask:         torch.Tensor,  # (batch, src_len)  – True where PAD
    ):
        src_len = encoder_outputs.size(0)

        # Broadcast decoder hidden over all source positions
        s = self.W_decoder(decoder_hidden)          # (B, hid)
        s = s.unsqueeze(1).repeat(1, src_len, 1)    # (B, src_len, hid)

        h = self.W_encoder(encoder_outputs)          # (src_len, B, hid)
        h = h.permute(1, 0, 2)                       # (B, src_len, hid)

        energy = torch.tanh(s + h)                  # (B, src_len, hid)
        scores = self.v(energy).squeeze(2)           # (B, src_len)

        # Mask out PAD positions with a very negative number
        scores = scores.masked_fill(src_mask, -1e10)

        weights = F.softmax(scores, dim=1)           # (B, src_len)

        # Weighted sum of encoder outputs
        context = torch.bmm(
            weights.unsqueeze(1),                    # (B, 1, src_len)
            encoder_outputs.permute(1, 0, 2),        # (B, src_len, hid)
        ).squeeze(1)                                 # (B, hid)

        return context, weights


class LuongAttention(nn.Module):
    """
    Multiplicative (Luong) attention – general version.

        score(s, h) = s^T · W · h

    Returns context and weights analogous to BahdanauAttention.
    """

    def __init__(self, hid_dim: int = cfg.HID_DIM):
        super().__init__()
        self.W = nn.Linear(hid_dim, hid_dim, bias=False)

    def forward(
        self,
        decoder_hidden:  torch.Tensor,  # (B, hid)
        encoder_outputs: torch.Tensor,  # (src_len, B, hid)
        src_mask:        torch.Tensor,  # (B, src_len)
    ):
        # Wh: (src_len, B, hid)
        Wh = self.W(encoder_outputs)
        # scores = s · (Wh)^T  →  (B, src_len)
        scores = torch.bmm(
            decoder_hidden.unsqueeze(1),    # (B, 1, hid)
            Wh.permute(1, 2, 0),            # (B, hid, src_len)
        ).squeeze(1)                        # (B, src_len)

        scores = scores.masked_fill(src_mask, -1e10)
        weights = F.softmax(scores, dim=1)

        context = torch.bmm(
            weights.unsqueeze(1),
            encoder_outputs.permute(1, 0, 2),
        ).squeeze(1)

        return context, weights


def build_attention(attention_type: str = cfg.ATTENTION_TYPE) -> nn.Module:
    """Factory that selects the attention implementation from config."""
    if attention_type == "bahdanau":
        return BahdanauAttention()
    elif attention_type == "luong":
        return LuongAttention()
    else:
        raise ValueError(f"Unknown attention type: {attention_type!r}")


# ══════════════════════════════════════════════════════════════
# Decoder (one-step-at-a-time)
# ══════════════════════════════════════════════════════════════

class Decoder(nn.Module):
    """
    Single-step attention decoder.

    At each decoding step:
      1. Embed the input token.
      2. Compute attention context from encoder outputs.
      3. Concatenate embedding + context → LSTM input.
      4. Run the stacked LSTM for one time step.
      5. Project LSTM output → target vocabulary logits.

    Input  : one token  (batch,)  plus hidden / cell / encoder_outputs
    Output : logits (batch, tgt_vocab_size), updated hidden & cell, weights
    """

    def __init__(
        self,
        output_dim:  int,         # target vocabulary size
        emb_dim:     int = cfg.EMB_DIM,
        hid_dim:     int = cfg.HID_DIM,
        n_layers:    int = cfg.N_LAYERS,
        dropout:     float = cfg.DEC_DROPOUT,
        attention_type: str = cfg.ATTENTION_TYPE,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim    = hid_dim

        self.embedding = nn.Embedding(
            output_dim, emb_dim, padding_idx=cfg.PAD_IDX
        )

        self.attention = build_attention(attention_type)

        # LSTM input = embedded token  +  attention context
        self.rnn = nn.LSTM(
            emb_dim + hid_dim,
            hid_dim,
            num_layers  = n_layers,
            dropout     = dropout if n_layers > 1 else 0.0,
            batch_first = False,
        )

        # Final output projection: combine hidden, context, and embedding
        self.fc_out = nn.Linear(hid_dim * 2 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt_token:       torch.Tensor,  # (batch,)
        hidden:          torch.Tensor,  # (n_layers, batch, hid)
        cell:            torch.Tensor,  # (n_layers, batch, hid)
        encoder_outputs: torch.Tensor,  # (src_len, batch, hid)
        src_mask:        torch.Tensor,  # (batch, src_len)
    ):
        # ── 1. Embed input token ─────────────────────────────
        embedded = self.dropout(
            self.embedding(tgt_token.unsqueeze(0))   # (1, B, emb)
        )

        # ── 2. Attention ─────────────────────────────────────
        # Use the top layer hidden state for attention query
        query = hidden[-1]                            # (B, hid)
        context, attn_weights = self.attention(query, encoder_outputs, src_mask)
        # context: (B, hid),  attn_weights: (B, src_len)

        # ── 3. Concatenate embedding + context ───────────────
        rnn_input = torch.cat(
            [embedded, context.unsqueeze(0)], dim=2  # (1, B, emb+hid)
        )

        # ── 4. LSTM step ──────────────────────────────────────
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # output: (1, B, hid)

        # ── 5. Projection ─────────────────────────────────────
        # Concatenate lstm output, context, and embedding for rich prediction
        prediction = self.fc_out(
            torch.cat(
                [output.squeeze(0), context, embedded.squeeze(0)],
                dim=1,
            )
        )  # (B, output_dim)

        return prediction, hidden, cell, attn_weights


# ══════════════════════════════════════════════════════════════
# Seq2Seq (full model)
# ══════════════════════════════════════════════════════════════

class Seq2Seq(nn.Module):
    """
    Combines Encoder + Decoder into a single trainable module.

    forward()  uses teacher forcing during training.
    The teacher-forcing ratio is decayed by the training loop.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device  = device

    def forward(
        self,
        src:              torch.Tensor,  # (src_len, B)
        tgt:              torch.Tensor,  # (tgt_len, B)
        src_lengths:      torch.Tensor,  # (B,)
        src_mask:         torch.Tensor,  # (B, src_len)
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Returns
        -------
        outputs : (tgt_len, B, tgt_vocab_size)  – logits at each step
        """
        tgt_len   = tgt.size(0)
        batch     = tgt.size(1)
        tgt_vocab = self.decoder.output_dim

        # Storage for decoder outputs
        outputs = torch.zeros(tgt_len, batch, tgt_vocab, device=self.device)

        # ── Encode ───────────────────────────────────────────
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)

        # First input to decoder is always the <sos> token
        dec_input = tgt[0]  # (B,)

        for t in range(1, tgt_len):
            out, hidden, cell, _ = self.decoder(
                dec_input, hidden, cell, encoder_outputs, src_mask
            )
            outputs[t] = out

            # Teacher forcing: feed ground-truth token or model prediction
            use_teacher = random.random() < teacher_forcing_ratio
            dec_input = tgt[t] if use_teacher else out.argmax(dim=1)

        return outputs

    # ────────────────────────────────────────────────────────
    # Greedy decode (for quick inference / evaluation)
    # ────────────────────────────────────────────────────────
    @torch.no_grad()
    def greedy_decode(
        self,
        src:         torch.Tensor,   # (src_len, 1)
        src_lengths: torch.Tensor,   # (1,)
        src_mask:    torch.Tensor,   # (1, src_len)
        max_len:     int = cfg.MAX_PRED_LEN,
    ):
        """
        Returns
        -------
        token_ids    : list[int]   – predicted token indices (without <sos>/<eos>)
        attn_weights : (max_len, src_len)  – attention weights per step
        """
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)

        dec_input    = torch.tensor([cfg.SOS_IDX], device=self.device)
        token_ids    = []
        all_weights  = []

        for _ in range(max_len):
            out, hidden, cell, attn = self.decoder(
                dec_input, hidden, cell, encoder_outputs, src_mask
            )
            pred_token = out.argmax(dim=1)
            all_weights.append(attn.squeeze(0).cpu())

            if pred_token.item() == cfg.EOS_IDX:
                break

            token_ids.append(pred_token.item())
            dec_input = pred_token

        return token_ids, torch.stack(all_weights, dim=0)

    # ────────────────────────────────────────────────────────
    # Beam search decode
    # ────────────────────────────────────────────────────────
    @torch.no_grad()
    def beam_search(
        self,
        src:         torch.Tensor,   # (src_len, 1)
        src_lengths: torch.Tensor,   # (1,)
        src_mask:    torch.Tensor,   # (1, src_len)
        beam_size:   int = cfg.BEAM_SIZE,
        max_len:     int = cfg.MAX_PRED_LEN,
    ) -> list:
        """
        Beam search decoding.

        Each beam is a tuple: (log_prob, token_ids, hidden, cell)
        Returns the top-1 token sequence after search.
        """
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)

        # ── Initialise beams ─────────────────────────────────
        # Each beam: (cumulative_log_prob, token_list, hidden, cell)
        beams = [(0.0, [], hidden, cell)]
        completed = []

        for _ in range(max_len):
            candidates = []

            for log_prob, tokens, h, c in beams:
                if tokens and tokens[-1] == cfg.EOS_IDX:
                    completed.append((log_prob, tokens))
                    continue

                # Last token (or <sos> for the first step)
                last_token = torch.tensor(
                    [tokens[-1] if tokens else cfg.SOS_IDX],
                    device=self.device,
                )

                out, new_h, new_c, _ = self.decoder(
                    last_token, h, c, encoder_outputs, src_mask
                )
                log_probs = F.log_softmax(out, dim=-1).squeeze(0)  # (vocab,)

                # Expand each beam with top-k tokens
                topk_log_probs, topk_ids = log_probs.topk(beam_size)

                for lp, tid in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                    candidates.append((
                        log_prob + lp,
                        tokens + [tid],
                        new_h,
                        new_c,
                    ))

            if not candidates:
                break

            # Keep top beam_size candidates by score
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_size]

        # Collect any beams that haven't produced <eos> yet
        for log_prob, tokens, _, _ in beams:
            completed.append((log_prob, tokens))

        # Return token sequence of the best beam using length normalization
        def length_norm_score(candidate):
            log_prob, tokens = candidate
            # Wu et al. length penalty
            length = max(1, len(tokens))
            penalty = ((5.0 + length) / 6.0) ** 0.7
            return log_prob / penalty

        best = max(completed, key=length_norm_score)
        # Strip <eos> if present
        seq = [t for t in best[1] if t != cfg.EOS_IDX]
        return seq


# ══════════════════════════════════════════════════════════════
# Model factory
# ══════════════════════════════════════════════════════════════

def build_model(src_vocab_size: int, tgt_vocab_size: int) -> Seq2Seq:
    """Instantiate and return the full Seq2Seq model."""
    enc = Encoder(src_vocab_size)
    dec = Decoder(tgt_vocab_size)
    model = Seq2Seq(enc, dec, cfg.DEVICE).to(cfg.DEVICE)

    # Xavier uniform initialisation for all weight matrices
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() > 1:
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.zeros_(param)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Seq2Seq built – {n_params:,} trainable parameters.")
    return model
