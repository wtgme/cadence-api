# Static KV Cache + torch.compile for SongGeneration Pipeline

**Date:** 2026-05-13  
**Branch:** feat/static-kv-cache  
**Goal:** Reduce per-song generation time from ~6 min to under 90 s on 4× L40S.

---

## Background

Baseline measurement (2× LM workers on GPUs 1–2, 1 diff worker on GPU 3):
- Total generation time: **6 min 11 s** (`/generate` blocking endpoint)
- GPU 0 occupied by a separate Gemma vLLM engine (~42 GB)

### Root cause

The generation loop in `lm_levo.py` runs ~6 750 decode steps (270 s × 25 Hz). Each step calls `LlamaAttention.forward()` in `modeling_llama.py`, which extends the KV cache with:

```python
key_states   = torch.cat([past_key_value[0], key_states],   dim=2)
value_states = torch.cat([past_key_value[1], value_states], dim=2)
```

At step N this copies N tokens of K and V data. Over 6 750 steps × 36 layers × 2 (K+V) × 2 (CFG batch) this is ~972 000 growing `torch.cat` calls — O(N²) total memory bandwidth — and is the dominant cost.

Secondary cost: `torch.compile` is available but defaults to `dynamic=True`, which prevents CUDA graph capture.

---

## Architecture

### Files changed

| File | Change |
|---|---|
| `songgeneration/codeclm/models/llama/modeling_llama.py` | Add static-cache path to `LlamaAttention` and `LlamaFlashAttention2` |
| `songgeneration/codeclm/models/llama/modeling_llama.py` | Thread `cache_position` through `LlamaDecoderLayer` and `LlamaModel` |
| `songgeneration/codeclm/models/lm_levo.py` | Buffer allocation and `cache_position` tracking in `LmModel.forward()` |
| `songgeneration_pipeline_server.py` | Change compile mode to `dynamic=False` |

### No changes to

- Model weights, config, or checkpoints
- Diff worker
- API endpoints
- Batch scheduler

---

## Section 1 — Attention layer (`modeling_llama.py`)

Add `cache_position: Optional[int] = None` to the signatures of:
- `LlamaAttention.forward()`
- `LlamaFlashAttention2.forward()`
- `LlamaDecoderLayer.forward()`
- `LlamaModel.forward()` / `CausalLM.forward()`

When `cache_position` is not `None`, `past_key_value` is treated as a **pre-allocated static buffer** `[batch, heads, max_seq_len, head_dim]`. The new token's K/V is written in-place and the valid slice is read back:

```python
if cache_position is not None:
    # static path — in-place write, no allocation
    past_key_value[0][:, :, cache_position, :] = key_states[:, :, 0, :]
    past_key_value[1][:, :, cache_position, :] = value_states[:, :, 0, :]
    key_states   = past_key_value[0][:, :, :cache_position + 1, :]
    value_states = past_key_value[1][:, :, :cache_position + 1, :]
    kv_seq_len   = cache_position + 1
else:
    # old growing path — unchanged, backward compatible
    key_states   = torch.cat([past_key_value[0], key_states], dim=2)
    value_states = torch.cat([past_key_value[1], value_states], dim=2)
    kv_seq_len   = key_states.shape[-2]
```

`kv_seq_len` also feeds the RoPE call (`self.rotary_emb(value_states, seq_len=kv_seq_len)`) and the attention mask size check — both must use the updated value.

Both `LlamaAttention` (standard) and `LlamaFlashAttention2` (hot path) need this change. Flash attention additionally transposes K/V to `[B, seq, heads, head_dim]` before `_flash_attention_forward`; the in-place write happens before that transpose.

`past_key_value` returned by the layer is now the **same buffer object** (no new allocation):
```python
past_key_value = (past_key_value[0], past_key_value[1]) if use_cache else None
```

---

## Section 2 — Buffer allocation and position tracking (`lm_levo.py`)

`LmModel.forward()` currently reads/writes `_streaming_state['past_key_values_1']` and `_streaming_state['past_key_values_2']`.

### New streaming state keys

| Key | Type | Description |
|---|---|---|
| `kv_cache_1` | `tuple[tuple[Tensor, Tensor], ...]` | Pre-allocated K/V buffers, one pair per layer, main transformer |
| `kv_cache_2` | same | Sub-transformer |
| `cache_pos_1` | `int` | Next write position for main transformer |
| `cache_pos_2` | `int` | Next write position for sub-transformer |

### Logic in `LmModel.forward()`

**First streaming call** (prefill — `kv_cache_1` not yet in state):
1. Call `self.transformer` with `use_cache=True, past_key_values=None` (existing path).
2. Call `_init_static_cache(output.past_key_values, max_seq_len, key='kv_cache_1')`:
   - Allocate `[batch, heads, max_seq_len, head_dim]` zero buffers for every layer.
   - Copy prefill K/V values into positions `0..prefill_len-1`.
   - Store buffers in `_streaming_state['kv_cache_1']`.
3. Set `_streaming_state['cache_pos_1'] = prefill_len` where `prefill_len = fused_input1.shape[1]`.

**Subsequent calls** (decode — `kv_cache_1` present):
```python
pos = self._streaming_state['cache_pos_1']
output = self.transformer(
    inputs_embeds=fused_input1,
    use_cache=True,
    past_key_values=self._streaming_state['kv_cache_1'],
    cache_position=pos,
)
self._streaming_state['cache_pos_1'] = pos + fused_input1.shape[1]
```
Same pattern for transformer2 / `kv_cache_2` / `cache_pos_2`.

### `_init_static_cache` helper (new private method on `LmModel`)

```python
def _init_static_cache(self, past_key_values, max_seq_len: int, key: str):
    device = next(self.parameters()).device
    dtype  = next(self.parameters()).dtype
    static = []
    for k_past, v_past in past_key_values:
        B, H, T, D = k_past.shape
        k_buf = torch.zeros(B, H, max_seq_len, D, device=device, dtype=dtype)
        v_buf = torch.zeros(B, H, max_seq_len, D, device=device, dtype=dtype)
        k_buf[:, :, :T, :] = k_past
        v_buf[:, :, :T, :] = v_past
        static.append((k_buf, v_buf))
    self._streaming_state[key] = tuple(static)
```

### Memory cost per LM worker

Main transformer static KV (36 layers, batch=4 for CFG×2, 16 heads, 128 head-dim, max_seq=10 000):  
`36 × 2 × (4 × 16 × 10 000 × 128) × 2 bytes ≈ 11.8 GB`

Sub-transformer (12 layers):  
`12 × 2 × (4 × 16 × 10 000 × 128) × 2 bytes ≈ 3.9 GB`

Total additional per worker: **~15.7 GB**. Current worker uses 10.6 GB; total ~26 GB; L40S has 46 GB. Comfortable margin.

---

## Section 3 — torch.compile (`songgeneration_pipeline_server.py`)

Current compile call (line 303):
```python
model.lm = torch.compile(model.lm, mode="reduce-overhead", dynamic=True)
```

Change to:
```python
model.lm = torch.compile(model.lm, mode="reduce-overhead", dynamic=False)
```

With static tensor shapes (fixed-size KV buffer, 1-token input per decode step), `reduce-overhead` can capture a CUDA graph on the first decode step and replay it for all subsequent steps — eliminating Python dispatch overhead entirely.

The prefill step has different input size but happens once per song; fallback to eager for that step is acceptable.

---

## Testing plan

1. **Unit correctness:** generate a short clip (30 s, `min_dur=30`) with static cache, verify MP3 is valid and sounds correct vs baseline.
2. **Timing benchmark:** run 3 generations with and without `SONGGEN_COMPILE=1`, record wall time and TTFB.
3. **Regression:** run existing `/health` and `/usage` endpoints; confirm worker counts and alive status unchanged.
4. **Memory check:** `nvidia-smi` after one generation; confirm no OOM and buffers released after `reset_streaming()`.

---

## Expected outcome

| Scenario | Estimated time |
|---|---|
| Baseline (current) | ~6 min 11 s |
| Static KV cache only | ~2–3 min |
| Static KV cache + compile | ~1–2 min |
