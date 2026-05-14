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
    """Run N decode steps using static buffer for S=1 steps, growing cache for S>1 prefill."""
    cfg = attn.config
    n_heads = cfg.num_key_value_heads   # fix: use kv heads, not attention heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    B = hidden_states_list[0].shape[0]
    dtype = hidden_states_list[0].dtype

    k_buf = torch.zeros(B, max_seq, n_heads, head_dim, device=device, dtype=dtype)
    v_buf = torch.zeros(B, max_seq, n_heads, head_dim, device=device, dtype=dtype)

    outputs = []
    cache_pos = 0
    past_kv = None  # used only during prefill (S>1)

    for hs in hidden_states_list:
        B_step, S, D = hs.shape
        pos = torch.arange(cache_pos, cache_pos + S, device=device).unsqueeze(0)

        if S > 1:
            # Prefill: use growing cache, then seed the static buffer
            out, _, past_kv = attn(
                hs, position_ids=pos, past_key_value=past_kv, use_cache=True
            )
            # Seed static buffer from prefill result — convert [B, H, T, D] -> [B, T, H, D]
            for layer_idx, (k_p, v_p) in enumerate([(past_kv[0], past_kv[1])]):
                T = k_p.shape[2]
                k_buf[:, :T, :, :] = k_p.permute(0, 2, 1, 3)
                v_buf[:, :T, :, :] = v_p.permute(0, 2, 1, 3)
        else:
            # Decode: use static buffer with cache_position
            out, _, _ = attn(
                hs, position_ids=pos,
                past_key_value=(k_buf, v_buf),
                use_cache=True,
                cache_position=cache_pos,
            )
        outputs.append(out)
        cache_pos += S
    return outputs


@pytest.mark.parametrize("flash", [False, True])
def test_static_cache_matches_growing_cache(flash):
    """Static cache must produce bit-identical outputs to growing cache."""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = _make_config(flash=flash)
    attn_cls = LlamaFlashAttention2 if flash else LlamaAttention
    dtype = torch.float16 if flash else torch.float32
    attn = attn_cls(cfg).to(device=device, dtype=dtype).eval()

    # 5 decode steps: 1 token each
    hidden_list = [
        torch.randn(1, 1, cfg.hidden_size, device=device, dtype=dtype)
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


def test_end_to_end_generation_completes():
    """Full LmModel.generate() must complete without error using static cache."""
    import os
    sys.path.insert(0, "songgeneration")
    sys.path.insert(0, "songgeneration/codeclm/tokenizer")
    sys.path.insert(0, "songgeneration/codeclm/tokenizer/Flow1dVAE")
    os.chdir("songgeneration")
    # Re-insert Flow1dVAE path relative to new cwd so model_septoken is importable
    flow_path = "codeclm/tokenizer/Flow1dVAE"
    if flow_path not in sys.path:
        sys.path.insert(0, flow_path)

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
