"""
Turkish-AI Transformer Model — v4
Tüm geliştirmeler eklendi:

  v2:
  • RoPE, KV Cache, RMSNorm, SwiGLU, Pre-Norm, bias=False, Scaled init

  v3:
  • GroupedQueryAttention (GQA)
  • Sliding Window Attention (SWA)

  v4 (yeni):
  • Flash Attention  → torch SDPA ile MPS/CUDA'da otomatik hız artışı
  • Repetition penalty düzeltmesi → pozitif/negatif logit ayrımı
  • Beam Search      → daha kaliteli metin üretimi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


# ---------------------------------------------------------------------------
# Normalizasyon
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


# ---------------------------------------------------------------------------
# Pozisyon Kodlaması
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, d_k: int, max_seq_length: int = 4096, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("inv_freq", inv_freq)
        self.max_cached = max_seq_length
        self._build_cache(max_seq_length)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cache", emb.sin()[None, None, :, :])

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]
        needed  = start_pos + seq_len
        if needed > self.max_cached:
            self.max_cached = needed + 512
            self._build_cache(self.max_cached)

        cos = self.cos_cache[:, :, start_pos:start_pos + seq_len, :]
        sin = self.sin_cache[:, :, start_pos:start_pos + seq_len, :]

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot.type_as(q), k_rot.type_as(k)


# ---------------------------------------------------------------------------
# Dikkat Mekanizması — Flash Attention destekli
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    Flash Attention eklendi (v4):
      torch.nn.functional.scaled_dot_product_attention kullanılıyor.
      MPS ve CUDA'da otomatik olarak Flash Attention kernel'ı seçilir.
      CPU'da standart attention çalışır.
      Sliding Window ile birlikte çalışmaz (SWA olan katmanlarda devre dışı).
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float      = 0.1,
        max_seq_length: int = 2048,
        window_size: int    = 0
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model     = d_model
        self.num_heads   = num_heads
        self.d_k         = d_model // num_heads
        self.window_size = window_size
        self.dropout_p   = dropout

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale   = math.sqrt(self.d_k)
        self.rotary  = RotaryEmbedding(self.d_k, max_seq_length)

    def _apply_sliding_window(self, scores, total_len, query_len):
        if self.window_size <= 0:
            return scores
        device     = scores.device
        q_positions = torch.arange(total_len - query_len, total_len, device=device)
        k_positions = torch.arange(total_len, device=device)
        dist        = q_positions.unsqueeze(1) - k_positions.unsqueeze(0)
        window_mask = (dist < 0) | (dist >= self.window_size)
        return scores.masked_fill(window_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = x.shape

        Q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        start_pos = past_key_value[0].shape[2] if past_key_value is not None else 0
        Q, K = self.rotary(Q, K, start_pos)

        if past_key_value is not None:
            K = torch.cat([past_key_value[0], K], dim=2)
            V = torch.cat([past_key_value[1], V], dim=2)

        present_kv = (K, V) if use_cache else None
        total_len  = K.shape[2]

        # Flash Attention: SWA yoksa ve PyTorch >= 2.0 ise kullan
        use_flash = (
            self.window_size <= 0
            and hasattr(F, "scaled_dot_product_attention")
        )

        if use_flash:
            # scaled_dot_product_attention causal mask'ı kendi halleder
            dropout_p = self.dropout_p if self.training else 0.0
            out = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask  = None,
                dropout_p  = dropout_p,
                is_causal  = (past_key_value is None)
            )
        else:
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float("-inf"))
            scores = self._apply_sliding_window(scores, total_len, T)
            attn   = F.softmax(scores, dim=-1)
            attn   = self.dropout(attn)
            out    = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out), present_kv


# ---------------------------------------------------------------------------
# Grouped Query Attention — Flash Attention destekli
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int   = None,
        dropout: float      = 0.1,
        max_seq_length: int = 2048,
        window_size: int    = 0
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model      = d_model
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.d_k          = d_model // num_heads
        self.window_size  = window_size
        self.dropout_p    = dropout

        assert num_heads % self.num_kv_heads == 0
        self.groups = num_heads // self.num_kv_heads

        self.W_q = nn.Linear(d_model, d_model,                      bias=False)
        self.W_k = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model,                      bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale   = math.sqrt(self.d_k)
        self.rotary  = RotaryEmbedding(self.d_k, max_seq_length)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = x.shape

        Q = self.W_q(x).view(B, T, self.num_heads,    self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)

        start_pos = past_key_value[0].shape[2] if past_key_value is not None else 0
        Q_rope, K = self.rotary(Q, K, start_pos)
        Q = Q_rope

        if past_key_value is not None:
            K = torch.cat([past_key_value[0], K], dim=2)
            V = torch.cat([past_key_value[1], V], dim=2)

        present_kv = (K, V) if use_cache else None
        total_len  = K.shape[2]

        K_exp = K.repeat_interleave(self.groups, dim=1)
        V_exp = V.repeat_interleave(self.groups, dim=1)

        use_flash = (
            self.window_size <= 0
            and hasattr(F, "scaled_dot_product_attention")
        )

        if use_flash:
            dropout_p = self.dropout_p if self.training else 0.0
            out = F.scaled_dot_product_attention(
                Q, K_exp, V_exp,
                attn_mask = None,
                dropout_p = dropout_p,
                is_causal = (past_key_value is None)
            )
        else:
            scores = torch.matmul(Q, K_exp.transpose(-2, -1)) / self.scale
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float("-inf"))
            if self.window_size > 0:
                device  = scores.device
                q_pos   = torch.arange(total_len - T, total_len, device=device)
                k_pos   = torch.arange(total_len, device=device)
                dist    = q_pos.unsqueeze(1) - k_pos.unsqueeze(0)
                win_mask = (dist < 0) | (dist >= self.window_size)
                scores  = scores.masked_fill(win_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out  = torch.matmul(attn, V_exp)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out), present_kv


# ---------------------------------------------------------------------------
# Feed-Forward
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_gate  = nn.Linear(d_model, d_ff, bias=False)
        self.w_up    = nn.Linear(d_model, d_ff, bias=False)
        self.w_down  = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(self.dropout(F.silu(self.w_gate(x)) * self.w_up(x)))


# ---------------------------------------------------------------------------
# Transformer Bloğu
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float      = 0.1,
        max_seq_length: int = 2048,
        num_kv_heads: int   = None,
        window_size: int    = 0
    ):
        super().__init__()

        if num_kv_heads is not None and num_kv_heads != num_heads:
            self.attention = GroupedQueryAttention(
                d_model, num_heads, num_kv_heads, dropout, max_seq_length, window_size
            )
        else:
            self.attention = MultiHeadAttention(
                d_model, num_heads, dropout, max_seq_length, window_size
            )

        self.feed_forward = SwiGLU(d_model, d_ff, dropout)
        self.norm1        = RMSNorm(d_model)
        self.norm2        = RMSNorm(d_model)
        self.dropout      = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = x
        attn_out, present_kv = self.attention(self.norm1(x), mask, past_key_value, use_cache)
        x = residual + self.dropout(attn_out)
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x, present_kv


# ---------------------------------------------------------------------------
# Ana Model
# ---------------------------------------------------------------------------

class TurkishAITransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int        = 512,
        num_heads: int      = 8,
        num_layers: int     = 6,
        d_ff: int           = 1536,
        max_seq_length: int = 2048,
        dropout: float      = 0.1,
        num_kv_heads: int   = None,
        window_size: int    = 0
    ):
        super().__init__()

        self.d_model        = d_model
        self.vocab_size     = vocab_size
        self.max_seq_length = max_seq_length
        self.num_layers     = num_layers

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, dropout, max_seq_length,
                num_kv_heads=num_kv_heads,
                window_size=window_size
            )
            for _ in range(num_layers)
        ])

        self.dropout           = nn.Dropout(dropout)
        self.norm              = RMSNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight

        self._init_weights()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        scale = 0.02 / math.sqrt(2 * self.num_layers)
        for name, p in self.named_parameters():
            if "w_down" in name or "W_o" in name:
                nn.init.normal_(p, mean=0.0, std=scale)

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        B, T = input_ids.shape

        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.dropout(x)

        past_len   = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        causal_mask = self.create_causal_mask(past_len + T, x.device)
        causal_mask = causal_mask[:, :, -T:, :]

        if attention_mask is not None:
            mask        = attention_mask.unsqueeze(1).unsqueeze(2)
            causal_mask = causal_mask * mask

        new_key_values: Optional[List] = [] if use_cache else None

        for i, block in enumerate(self.transformer_blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(x, causal_mask, past_kv, use_cache)
            if use_cache:
                new_key_values.append(present_kv)

        x      = self.norm(x)
        logits = self.output_projection(x)
        return logits, new_key_values

    # ------------------------------------------------------------------
    # Greedy / Sampling üretimi — Repetition Penalty düzeltildi (v4)
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int       = 100,
        temperature: float        = 0.8,
        top_k: int                = 50,
        top_p: float              = 0.95,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        self.eval()

        with torch.no_grad():
            generated      = input_ids.clone()
            past_key_values = None

            logits, past_key_values = self.forward(input_ids, use_cache=True)

            for _ in range(max_new_tokens):
                next_logits = logits[:, -1, :] / temperature

                # --- Repetition penalty düzeltmesi (v4) ---
                # Pozitif logit → böl (olasılığı düşür)
                # Negatif logit → çarp (daha da negatif yap)
                if repetition_penalty != 1.0:
                    for tid in set(generated[0].tolist()):
                        if next_logits[0, tid] > 0:
                            next_logits[0, tid] /= repetition_penalty
                        else:
                            next_logits[0, tid] *= repetition_penalty

                # Top-k
                if top_k > 0:
                    k         = min(top_k, next_logits.size(-1))
                    threshold = torch.topk(next_logits, k)[0][:, -1, None]
                    next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

                # Top-p (nucleus sampling)
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove    = cum_probs > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0]  = False
                    to_remove   = remove.scatter(1, sorted_idx, remove)
                    next_logits = next_logits.masked_fill(to_remove, float("-inf"))

                probs      = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_token], dim=1)

                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break

                logits, past_key_values = self.forward(
                    next_token, past_key_values=past_key_values, use_cache=True
                )

        return generated

    # ------------------------------------------------------------------
    # Beam Search — v4 yeni eklendi
    # ------------------------------------------------------------------

    def beam_search(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int     = 100,
        num_beams: int          = 4,
        temperature: float      = 1.0,
        length_penalty: float   = 1.0,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Beam Search ile metin üretimi.

        generate() random sampling yapar (her seferinde farklı çıktı).
        beam_search() en yüksek olasılıklı yolu arar (deterministik, daha tutarlı).

        length_penalty:
          < 1.0 → kısa cevapları tercih eder
          > 1.0 → uzun cevapları tercih eder
          = 1.0 → nötr
        """
        self.eval()
        device   = input_ids.device
        B, T     = input_ids.shape
        assert B == 1, "Beam search şu an sadece batch_size=1 destekliyor"

        with torch.no_grad():
            # Her beam için (score, token_ids) tut
            beams: List[Tuple[float, torch.Tensor]] = [(0.0, input_ids[0])]
            completed: List[Tuple[float, torch.Tensor]] = []

            for step in range(max_new_tokens):
                all_candidates: List[Tuple[float, torch.Tensor]] = []

                for score, seq in beams:
                    if eos_token_id is not None and seq[-1].item() == eos_token_id:
                        completed.append((score, seq))
                        continue

                    inp    = seq.unsqueeze(0)
                    logits, _ = self.forward(inp, use_cache=False)
                    next_logits = logits[0, -1, :] / temperature
                    log_probs   = F.log_softmax(next_logits, dim=-1)

                    # En iyi num_beams token'ı al
                    topk_log_probs, topk_ids = torch.topk(log_probs, num_beams)

                    for i in range(num_beams):
                        new_score = score + topk_log_probs[i].item()
                        new_seq   = torch.cat([seq, topk_ids[i:i+1]])
                        all_candidates.append((new_score, new_seq))

                if not all_candidates:
                    break

                # Length penalty uygula ve en iyi num_beams beam'i seç
                all_candidates.sort(
                    key=lambda x: x[0] / (len(x[1]) ** length_penalty),
                    reverse=True
                )
                beams = all_candidates[:num_beams]

            # Tamamlanan beam yoksa mevcut en iyiyi al
            if not completed:
                completed = beams

            completed.sort(
                key=lambda x: x[0] / (len(x[1]) ** length_penalty),
                reverse=True
            )
            best_seq = completed[0][1]

        return best_seq.unsqueeze(0)


# ---------------------------------------------------------------------------
# Model konfigürasyonları
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "tiny": {
        "d_model":        256,
        "num_heads":      4,
        "num_layers":     4,
        "d_ff":           768,
        "max_seq_length": 1024,
        "num_kv_heads":   None,
        "window_size":    0
    },
    "small": {
        "d_model":        512,
        "num_heads":      8,
        "num_layers":     6,
        "d_ff":           1536,
        "max_seq_length": 2048,
        "num_kv_heads":   None,
        "window_size":    0
    },
    "medium": {
        "d_model":        768,
        "num_heads":      12,
        "num_layers":     12,
        "d_ff":           2048,
        "max_seq_length": 2048,
        "num_kv_heads":   2,
        "window_size":    512
    },
    "large": {
        "d_model":        1024,
        "num_heads":      16,
        "num_layers":     24,
        "d_ff":           2816,
        "max_seq_length": 4096,
        "num_kv_heads":   2,
        "window_size":    1024
    }
}


def create_model(vocab_size: int, model_size: str = "small", **kwargs) -> TurkishAITransformer:
    config = MODEL_CONFIGS[model_size].copy()
    config.update(kwargs)
    return TurkishAITransformer(vocab_size=vocab_size, **config)