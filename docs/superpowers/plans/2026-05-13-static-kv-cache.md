# Static KV Cache + torch.compile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the O(N²) growing `torch.cat` KV cache in the SongGeneration LM with a pre-allocated static buffer, then enable CUDA-graph-compatible `torch.compile`, reducing per-song generation time from ~6 min to under 2 min.

**Architecture:** Add a `cache_position: Optional[int]` parameter that threads from `LmModel.forward()` (lm_levo.py) down through `CausalLM` (levo.py) → `LlamaModel` → `LlamaDecoderLayer` → attention layers. When `cache_position` is set, attention writes K/V in-place into a pre-allocated buffer instead of calling `torch.cat`. Buffers are allocated once at the end of the prefill step and stored in the streaming state.

**Tech Stack:** PyTorch 2.6, Flash Attention 2, existing `StreamingModule` streaming state, `torch.compile(mode="reduce-overhead")`

---

## File Map

| File | What changes |
|---|---|
| `songgeneration/codeclm/models/llama/modeling_llama.py` | Add `cache_position` to `LlamaModel`, `LlamaDecoderLayer`, `LlamaAttention`, `LlamaFlashAttention2` |
| `songgeneration/codeclm/models/levo.py` | Add `cache_position` to `CausalLM.forward()` |
| `songgeneration/codeclm/models/lm_levo.py` | Add `_init_static_cache()`, update `LmModel.forward()` streaming logic |
| `songgeneration_pipeline_server.py` | Change `dynamic=True` → `dynamic=False` in torch.compile call |
| `tests/test_static_kv_cache.py` | New: correctness + benchmark tests |

All paths are relative to `/users/k1810895/data/musicgen/`.

---

## Task 1: Write failing correctness test

**Files:**
- Create: `tests/test_static_kv_cache.py`

- [ ] **Step 1: Create tests directory and test file**

```bash
mkdir -p /users/k1810895/data/musicgen/tests
```

- [ ] **Step 2: Write the test**

```python
# tests/test_static_kv_cache.py
"""
Correctness test: static KV cache path must produce identical attention outputs
to the original torch.cat growing-cache path for the same inputs.

Run with:
  cd /users/k1810895/data/musicgen
  conda run -n musicgen python -m pytest tests/test_static_kv_cache.py -v
"""
import sys
sys.path.insert(0, "songgeneration")
sys.path.insert(0, "songgeneration/codeclm/tokenizer")

import torch
import pytest
from codeclm.models.llama.modeling_llama import LlamaConfig, LlamaAttention, LlamaFlashAttention2


def _make_config(flash: bool = False) -> LlamaConfig:
    return LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=4,
        vocab_size=100,
        use_cache=True,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        _flash_attn_2_enabled=flash,
    )


def _run_growing_cache(attn, hidden_states_list, device):
    """Run N decode steps with the old torch.cat growing cache."""
    past_kv = None
    outputs = []
    pos_offset = 0
    for hs in hidden_states_list:
        B, S, D = hs.shape
        pos = torch.arange(pos_offset, pos_offset + S, device=device).unsqueeze(0)
        out, _, past_kv = attn(
            hs, position_ids=pos, past_key_value=past_kv, use_cache=True
        )
        outputs.append(out)
        pos_offset += S
    return outputs


def _run_static_cache(attn, hidden_states_list, device, max_seq=64):
    """Run N decode steps using pre-allocated static buffer + cache_position."""
    cfg = attn.config
    n_heads = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_heads
    B = hidden_states_list[0].shape[0]
    dtype = hidden_states_list[0].dtype

    k_buf = torch.zeros(B, n_heads, max_seq, head_dim, device=device, dtype=dtype)
    v_buf = torch.zeros(B, n_heads, max_seq, head_dim, device=device, dtype=dtype)
    static_kv = (k_buf, v_buf)

    outputs = []
    cache_pos = 0
    for hs in hidden_states_list:
        B, S, D = hs.shape
        pos = torch.arange(cache_pos, cache_pos + S, device=device).unsqueeze(0)
        out, _, _ = attn(
            hs, position_ids=pos,
            past_key_value=static_kv,
            use_cache=True,
            cache_position=cache_pos,
        )
        outputs.append(out)
        cache_pos += S
    return outputs


@pytest.mark.parametrize("flash", [False])
def test_static_cache_matches_growing_cache(flash):
    """Static cache must produce bit-identical outputs to growing cache."""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = _make_config(flash=flash)
    attn_cls = LlamaFlashAttention2 if flash else LlamaAttention
    attn = attn_cls(cfg).to(device).eval()

    # 3 decode steps: 1 token each
    hidden_list = [
        torch.randn(1, 1, cfg.hidden_size, device=device)
        for _ in range(5)
    ]

    with torch.no_grad():
        growing = _run_growing_cache(attn, hidden_list, device)
        static  = _run_static_cache(attn, hidden_list, device)

    for step, (g, s) in enumerate(zip(growing, static)):
        assert torch.allclose(g, s, atol=1e-4), (
            f"Step {step}: max diff = {(g - s).abs().max().item():.6f}"
        )


def test_static_cache_multi_token_prefill():
    """Prefill (S>1) followed by 1-token decode steps must stay consistent."""
    torch.manual_seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = _make_config(flash=False)
    attn = LlamaAttention(cfg).to(device).eval()

    prefill = torch.randn(1, 4, cfg.hidden_size, device=device)
    decode_steps = [torch.randn(1, 1, cfg.hidden_size, device=device) for _ in range(3)]
    all_steps = [prefill] + decode_steps

    with torch.no_grad():
        growing = _run_growing_cache(attn, all_steps, device)
        # For multi-token prefill in static path, first step still uses cache_position=0
        static  = _run_static_cache(attn, all_steps, device)

    for step, (g, s) in enumerate(zip(growing, static)):
        assert torch.allclose(g, s, atol=1e-4), (
            f"Step {step}: max diff = {(g - s).abs().max().item():.6f}"
        )
```

- [ ] **Step 3: Run test to confirm it fails (cache_position param doesn't exist yet)**

```bash
cd /users/k1810895/data/musicgen
conda run -n musicgen python -m pytest tests/test_static_kv_cache.py -v 2>&1 | tail -20
```

Expected: `TypeError: forward() got an unexpected keyword argument 'cache_position'`

---

## Task 2: Thread `cache_position` through `LlamaModel.forward()`

**Files:**
- Modify: `songgeneration/codeclm/models/llama/modeling_llama.py:833-962`

The key change: when `cache_position` is set, use it (not `past_key_values[0][0].shape[2]`) as the positional offset, and pass it into each decoder layer.

- [ ] **Step 1: Add `cache_position` to `LlamaModel.forward()` signature and fix position_ids / attention mask**

Replace the `LlamaModel.forward()` signature (line 833) and the `past_key_values_length` block (lines 863–895):

```python
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[int] = None,   # <-- NEW
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            if cache_position is not None:
                # Static cache: use the explicit position counter, not the buffer size
                past_key_values_length = cache_position
            else:
                past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
            padding_mask = None
        else:
            if 0 in attention_mask:
                padding_mask = attention_mask
            else:
                padding_mask = None

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions, padding_mask=padding_mask)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer), hidden_states, attention_mask, position_ids
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    padding_mask=padding_mask,
                    cache_position=cache_position,   # <-- NEW
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
```

- [ ] **Step 2: Add `cache_position` to `LlamaDecoderLayer.forward()` signature and pass it to self_attn**

Find the `LlamaDecoderLayer.forward()` definition (line 611). Replace the signature and the `self.self_attn(...)` call:

```python
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        padding_mask: Optional[torch.LongTensor] = None,
        cache_position: Optional[int] = None,   # <-- NEW
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
```

And within the body, update the `self.self_attn(...)` call (around line 640):

```python
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
            cache_position=cache_position,   # <-- NEW
        )
```

- [ ] **Step 3: Run the tests — still expected to fail (attention doesn't handle cache_position yet)**

```bash
cd /users/k1810895/data/musicgen
conda run -n musicgen python -m pytest tests/test_static_kv_cache.py -v 2>&1 | tail -20
```

Expected: `TypeError` on `LlamaAttention.forward()`

- [ ] **Step 4: Commit**

```bash
cd /users/k1810895/data/musicgen
git add songgeneration/codeclm/models/llama/modeling_llama.py tests/test_static_kv_cache.py
git commit -m "feat: thread cache_position through LlamaModel and LlamaDecoderLayer"
```

---

## Task 3: Static KV write path in `LlamaAttention`

**Files:**
- Modify: `songgeneration/codeclm/models/llama/modeling_llama.py:324-417`

- [ ] **Step 1: Add `cache_position` to `LlamaAttention.forward()` signature and replace the `torch.cat` block**

Replace the entire `LlamaAttention.forward()` method signature (line 324) and the `kv_seq_len` / `past_key_value` block (lines 362–373):

New signature (add `cache_position: Optional[int] = None` after `padding_mask`):
```python
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
        cache_position: Optional[int] = None,   # <-- NEW
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
```

Replace the lines from `kv_seq_len = key_states.shape[-2]` through `past_key_value = (key_states, value_states) if use_cache else None` (lines 362–373) with:

```python
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if cache_position is not None:
                # Static cache: write new K/V in-place at cache_position, no allocation
                past_key_value[0][:, :, cache_position, :] = key_states[:, :, 0, :]
                past_key_value[1][:, :, cache_position, :] = value_states[:, :, 0, :]
                kv_seq_len = cache_position + 1
                key_states   = past_key_value[0][:, :, :kv_seq_len, :]
                value_states = past_key_value[1][:, :, :kv_seq_len, :]
            else:
                # Original growing cache
                kv_seq_len += past_key_value[0].shape[-2]
                key_states   = torch.cat([past_key_value[0], key_states],   dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Return the buffer itself (no new tuple allocation in static mode)
        past_key_value = (past_key_value[0], past_key_value[1]) if (use_cache and cache_position is not None) \
                    else ((key_states, value_states) if use_cache else None)
```

Note: the original code called `apply_rotary_pos_emb` *before* the `torch.cat`. The static path must also call RoPE before the in-place write so the stored K values have rotary embeddings applied. Adjust the order:

The full replacement for lines 362–373 including RoPE:

```python
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if cache_position is not None:
                kv_seq_len = cache_position + 1
            else:
                kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            if cache_position is not None:
                # Static path: write at position, read back valid slice
                past_key_value[0][:, :, cache_position, :] = key_states[:, :, 0, :]
                past_key_value[1][:, :, cache_position, :] = value_states[:, :, 0, :]
                key_states   = past_key_value[0][:, :, :kv_seq_len, :]
                value_states = past_key_value[1][:, :, :kv_seq_len, :]
            else:
                # Growing cache (original path)
                key_states   = torch.cat([past_key_value[0], key_states],   dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (past_key_value[0], past_key_value[1]) if (use_cache and cache_position is not None) \
                    else ((key_states, value_states) if use_cache else None)
```

Remove the old standalone `cos, sin = self.rotary_emb(...)` and `apply_rotary_pos_emb(...)` lines (they were at lines 365–366 and are now inside the rewritten block).

- [ ] **Step 2: Run tests**

```bash
cd /users/k1810895/data/musicgen
conda run -n musicgen python -m pytest tests/test_static_kv_cache.py::test_static_cache_matches_growing_cache -v 2>&1 | tail -20
```

Expected: PASS for `flash=False`. (Flash attention test is Task 4.)

- [ ] **Step 3: Commit**

```bash
git add songgeneration/codeclm/models/llama/modeling_llama.py
git commit -m "feat: static KV cache in-place write path for LlamaAttention"
```

---

## Task 4: Static KV write path in `LlamaFlashAttention2`

**Files:**
- Modify: `songgeneration/codeclm/models/llama/modeling_llama.py:427-504`

Flash attention uses `[B, seq, heads, head_dim]` layout (transposed vs standard attention). The in-place write must happen **before** the transpose.

- [ ] **Step 1: Add `cache_position` to `LlamaFlashAttention2.forward()` and replace the KV block**

New signature:
```python
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
        cache_position: Optional[int] = None,   # <-- NEW
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
```

Replace the block from `kv_seq_len = key_states.shape[-2]` through `past_key_value = (key_states, value_states) if use_cache else None` (lines 453–466) with:

```python
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if cache_position is not None:
                kv_seq_len = cache_position + 1
            else:
                kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            if cache_position is not None:
                # Static path: buffers are [B, heads, max_seq, head_dim]
                past_key_value[0][:, :, cache_position, :] = key_states[:, :, 0, :]
                past_key_value[1][:, :, cache_position, :] = value_states[:, :, 0, :]
                key_states   = past_key_value[0][:, :, :kv_seq_len, :]
                value_states = past_key_value[1][:, :, :kv_seq_len, :]
            else:
                key_states   = torch.cat([past_key_value[0], key_states],   dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (past_key_value[0], past_key_value[1]) if (use_cache and cache_position is not None) \
                    else ((key_states, value_states) if use_cache else None)
```

Remove the old standalone `cos, sin` and `apply_rotary_pos_emb` lines (lines 457–459 in the original).

The transpose to flash-attention layout happens *after* this block (lines 468–470 in original):
```python
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
```
Leave those lines unchanged — they now operate on the valid slice.

- [ ] **Step 2: Add flash=True test variant to test file**

Open `tests/test_static_kv_cache.py` and change:
```python
@pytest.mark.parametrize("flash", [False])
```
to:
```python
@pytest.mark.parametrize("flash", [False, True])
```

- [ ] **Step 3: Run full test suite**

```bash
cd /users/k1810895/data/musicgen
conda run -n musicgen python -m pytest tests/test_static_kv_cache.py -v 2>&1 | tail -30
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add songgeneration/codeclm/models/llama/modeling_llama.py tests/test_static_kv_cache.py
git commit -m "feat: static KV cache path for LlamaFlashAttention2"
```

---

## Task 5: Thread `cache_position` through `CausalLM` in `levo.py`

**Files:**
- Modify: `songgeneration/codeclm/models/levo.py:23-88`

`CausalLM.forward()` calls `self.model(...)` which is `LmModel` (a `LlamaModel` subclass). We need to pass `cache_position` through.

- [ ] **Step 1: Add `cache_position` to `CausalLM.forward()` and thread it to `self.model()`**

Replace the `CausalLM.forward()` signature (line 23) and `self.model(...)` call (line 44):

```python
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[int] = None,   # <-- NEW
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,   # <-- NEW
        )
```

Leave everything after the `outputs = self.model(...)` call unchanged.

- [ ] **Step 2: Run tests — still all green**

```bash
cd /users/k1810895/data/musicgen
conda run -n musicgen python -m pytest tests/test_static_kv_cache.py -v 2>&1 | tail -10
```

Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add songgeneration/codeclm/models/levo.py
git commit -m "feat: thread cache_position through CausalLM"
```

---

## Task 6: Buffer allocation and streaming state in `lm_levo.py`

**Files:**
- Modify: `songgeneration/codeclm/models/lm_levo.py:245-292`

This is the core change: allocate static buffers at end of prefill, then use them for all decode steps.

- [ ] **Step 1: Add `_init_static_cache()` private method to `LmModel`**

Add this method to the `LmModel` class, just before the `forward()` method (before line 245):

```python
    def _init_static_cache(
        self,
        past_key_values: tuple,
        max_seq_len: int,
        key: str,
    ) -> None:
        """Allocate fixed KV buffers and copy prefill values in.

        past_key_values: tuple of (k, v) per layer, shape [B, heads, T, head_dim].
        max_seq_len: total buffer capacity (>= T + remaining decode steps).
        key: streaming_state key to store the buffers under.
        """
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

- [ ] **Step 2: Update `LmModel.forward()` to use static cache for decode steps**

Replace the entire body of `LmModel.forward()` (lines 260–292) with:

```python
    def forward(self,
                sequence: torch.Tensor,
                condition_tensors: ConditionTensors) -> torch.Tensor:
        B, K, S = sequence.shape
        assert K == self.code_depth

        input_1 = self.emb[0](sequence[:, 0])
        input_2 = sum([self.layer2_emb[k](sequence[:, k]) for k in range(1, K)])
        fused_input1, fused_input2 = self.fuser(input_1, input_2, condition_tensors)

        # ---- Main transformer (transformer1) ----
        if self._is_streaming and 'kv_cache_1' in self._streaming_state:
            # Decode path: static cache, in-place write
            pos = self._streaming_state['cache_pos_1']
            output = self.transformer(
                inputs_embeds=fused_input1,
                use_cache=True,
                past_key_values=self._streaming_state['kv_cache_1'],
                cache_position=pos,
            )
            self._streaming_state['cache_pos_1'] = pos + fused_input1.shape[1]
        else:
            # Prefill path (first call or non-streaming): original growing cache
            output = self.transformer(
                inputs_embeds=fused_input1,
                use_cache=self._is_streaming,
                past_key_values=self._streaming_state.get('past_key_values_1', None),
            )
            if self._is_streaming:
                prefill_len = fused_input1.shape[1]
                self._init_static_cache(
                    output.past_key_values,
                    max_seq_len=self.transformer.config.max_position_embeddings,
                    key='kv_cache_1',
                )
                self._streaming_state['cache_pos_1'] = prefill_len

        logits = output.logits          # [B, S, card]
        logits = logits.unsqueeze(1)    # [B, 1, S, card]

        # ---- Sub-transformer (transformer2) ----
        if K > 1:
            fused_input2 = torch.cat([fused_input2, output.hidden_states], dim=-1)
            fused_input2 = self.mlp(fused_input2)

            if self._is_streaming and 'kv_cache_2' in self._streaming_state:
                pos2 = self._streaming_state['cache_pos_2']
                output2 = self.transformer2(
                    inputs_embeds=fused_input2,
                    use_cache=True,
                    past_key_values=self._streaming_state['kv_cache_2'],
                    cache_position=pos2,
                )
                self._streaming_state['cache_pos_2'] = pos2 + fused_input2.shape[1]
            else:
                output2 = self.transformer2(
                    inputs_embeds=fused_input2,
                    use_cache=self._is_streaming,
                    past_key_values=self._streaming_state.get('past_key_values_2', None),
                )
                if self._is_streaming:
                    prefill_len2 = fused_input2.shape[1]
                    self._init_static_cache(
                        output2.past_key_values,
                        max_seq_len=self.transformer2.config.max_position_embeddings,
                        key='kv_cache_2',
                    )
                    self._streaming_state['cache_pos_2'] = prefill_len2

            res_logits = torch.stack(
                [self.linears[k](output2.hidden_states) for k in range(K - 1)], dim=1
            )
            logits = torch.cat([logits, res_logits], dim=1)

        if len(self.fuser.fuse2cond['prepend']) > 0:
            logits = logits[:, :, -S:, :]

        return logits
```

- [ ] **Step 3: Write an end-to-end generation test**

Append to `tests/test_static_kv_cache.py`:

```python
def test_end_to_end_generation_completes():
    """Full LmModel.generate() must complete without error using static cache."""
    import sys, os
    sys.path.insert(0, "songgeneration")
    sys.path.insert(0, "songgeneration/codeclm/tokenizer")
    sys.path.insert(0, "songgeneration/codeclm/tokenizer/Flow1dVAE")
    os.chdir("songgeneration")

    import torch
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval",      lambda x: eval(x),                       replace=True)
    OmegaConf.register_new_resolver("concat",    lambda *x: [i for s in x for i in s],   replace=True)
    OmegaConf.register_new_resolver("get_fname", lambda: "server",                        replace=True)
    OmegaConf.register_new_resolver("load_yaml", lambda x: list(OmegaConf.load(x)),      replace=True)

    from codeclm.models import builders, CodecLM

    cfg = OmegaConf.load("ckpt/songgeneration_v2_large/config.yaml")
    cfg.lm.use_flash_attn_2 = True
    cfg.mode = "inference"
    cfg.max_dur = 30   # short clip for speed

    audiolm = builders.get_lm_model(cfg, version="v2")
    ckpt = torch.load("ckpt/songgeneration_v2_large/model.pt", map_location="cpu", mmap=True)
    sd = {k.replace("audiolm.", ""): v for k, v in ckpt.items() if k.startswith("audiolm")}
    audiolm.load_state_dict(sd, strict=False)
    audiolm = audiolm.eval().cuda().to(torch.float16)

    auto_prompt = torch.load("tools/new_auto_prompt.pt", map_location="cpu")
    sep_tok = builders.get_audio_tokenizer_model_cpu(cfg.audio_tokenizer_checkpoint_sep, cfg)
    sep_tok.model.vae   = sep_tok.model.vae.to("cuda")
    sep_tok.model.model = sep_tok.model.model.to("cuda")
    sep_tok = sep_tok.eval()

    model = CodecLM(
        name="test",
        lm=audiolm,
        audiotokenizer=None,
        max_duration=cfg.max_dur,
        seperate_tokenizer=sep_tok,
    )
    model.set_generation_params(
        duration=cfg.max_dur, extend_stride=5, temperature=0.8,
        cfg_coef=1.5, top_k=5000, top_p=0.0,
        record_tokens=True, record_window=50,
    )

    import time
    t0 = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with torch.no_grad():
            tokens = model.generate(
                lyrics=["."],
                descriptions=["[Musicality-very-high], [Pure-Music], ambient,calm"],
                melody_wavs=None, vocal_wavs=None, bgm_wavs=None,
                melody_is_wav=True, return_tokens=True,
            )
    elapsed = time.time() - t0

    assert tokens is not None
    assert tokens.shape[-1] > 0, "No tokens generated"
    print(f"\nGeneration time for 30s clip: {elapsed:.1f}s  (tokens shape: {tokens.shape})")
    os.chdir("..")
```

- [ ] **Step 4: Run the end-to-end test**

```bash
cd /users/k1810895/data/musicgen
conda run -n musicgen python -m pytest tests/test_static_kv_cache.py::test_end_to_end_generation_completes -v -s 2>&1 | tail -30
```

Expected: PASS. The print line shows the new generation time.

- [ ] **Step 5: Commit**

```bash
git add songgeneration/codeclm/models/lm_levo.py tests/test_static_kv_cache.py
git commit -m "feat: static KV cache buffer allocation and decode path in LmModel"
```

---

## Task 7: Enable `torch.compile` with static shapes

**Files:**
- Modify: `songgeneration_pipeline_server.py:303`

- [ ] **Step 1: Change compile mode to `dynamic=False`**

Find line 303 in `songgeneration_pipeline_server.py`:
```python
        model.lm = torch.compile(model.lm, mode="reduce-overhead", dynamic=True)
```

Replace with:
```python
        model.lm = torch.compile(model.lm, mode="reduce-overhead", dynamic=False)
```

- [ ] **Step 2: Commit**

```bash
git add songgeneration_pipeline_server.py
git commit -m "feat: enable static-shape torch.compile for CUDA graph capture"
```

---

## Task 8: Benchmark and validate

- [ ] **Step 1: Restart the pipeline server with compile enabled**

```bash
# Kill existing pipeline workers
pkill -f "songgeneration_pipeline_server" || true
sleep 5

cd /users/k1810895/data/musicgen
nohup bash -c "source /software/spackages_v0_21_prod/apps/linux-ubuntu22.04-icelake/gcc-13.2.0/anaconda3-2022.10-tjkkt6f5oslpe3qj7vrpvqrm7vru4k6e/etc/profile.d/conda.sh && cd /users/k1810895/data/musicgen && SONGGEN_COMPILE=1 conda run -n musicgen --no-capture-output python -m uvicorn songgeneration_pipeline_server:app --host 0.0.0.0 --port 8888" >> logs/pipeline_stdout.log 2>&1 &

# Wait for workers to be ready (watch for "LM worker ready" in logs)
tail -f logs/pipeline_stdout.log
```

Wait until you see `LM worker ready (gpu_id=...)` for each worker.

- [ ] **Step 2: Run 3 timed generation requests**

```bash
for i in 1 2 3; do
  echo "=== Run $i ===";
  time curl -s -o /tmp/bench_$i.mp3 -w "HTTP %{http_code} | bytes=%{size_download} | total=%{time_total}s\n" \
    -X POST http://localhost:8888/generate \
    -H "Content-Type: application/json" \
    -d '{"lyric":".","descriptions":"ambient,calm,peaceful","auto_prompt_audio_type":"Soundtrack","generate_type":"bgm"}';
done
```

Record the `total=` values.

- [ ] **Step 3: Verify MP3 is valid audio**

```bash
conda run -n musicgen python -c "
import torchaudio, sys
wav, sr = torchaudio.load('/tmp/bench_1.mp3')
print(f'Duration: {wav.shape[-1]/sr:.1f}s  Sample rate: {sr}  Channels: {wav.shape[0]}')
assert wav.shape[-1] > sr * 20, 'Audio shorter than 20s — generation may have failed'
print('OK')
"
```

- [ ] **Step 4: Check GPU memory**

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
```

Expected: each LM worker GPU shows ~26–28 GB used (10.6 GB model + ~15.7 GB static KV cache). All within 46 GB.

- [ ] **Step 5: Commit benchmark results as a note**

```bash
git commit --allow-empty -m "bench: static KV cache results — record timing in PR description"
```

---

## Self-Review Checklist (completed inline)

- **Spec coverage:** All three spec sections covered — attention layer (Tasks 3–4), streaming state (Task 6), compile (Task 7). Memory calculation from spec verified in Task 8 Step 4.
- **No placeholders:** All code blocks are complete. No TBDs.
- **Type consistency:** `cache_position: Optional[int]` used consistently across all tasks. `_init_static_cache()` signature defined in Task 6 Step 1 and called in Task 6 Step 2 with matching args. `kv_cache_1`/`kv_cache_2` keys match across init and read.
- **RoPE ordering:** Task 3 and 4 explicitly note that RoPE is applied before the in-place write so stored K values have embeddings applied.
- **Flash attention transpose:** Task 4 notes the `transpose(1,2)` happens after the static write block, operating on the valid slice.
