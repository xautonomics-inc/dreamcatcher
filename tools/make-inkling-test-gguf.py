#!/usr/bin/env python3
"""Build a synthetic Inkling-arch GGUF for the D1 load smoke and the D2 two-build ppl diff.

Sized small (2 layers: 1 dense lead + 1 MoE) but with the exact metadata keys
and tensor shapes the reference loader expects, so llama-cli --load can verify
that every Inkling tensor is created with the right shape/type on CPU, and
llama-perplexity can diff two builds on ~20 MB instead of the 152 GB model.

The header values are chosen so the fixture exercises the code paths the real
model uses, not just fixture-shaped defaults: expert_weights_scale 8.0 and
expert_gating_func 2 as in the real header, a log-N floor small enough that tau
departs from 1.0 inside one 2048-token chunk, add_bos_token false, and three
distinct sliding-window / relative-extent values so the global and SWA index
tensors are not interchangeable.
"""
import hashlib
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))
import gguf

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("inkling-test.gguf")

# Model facts (from the Inkling-Small model card, scaled down for a smoke test).
N_LAYER     = 2
N_EMBD      = 256
N_VOCAB     = 1024
N_HEAD      = 8
N_HEAD_KV   = 2   # GQA 4:1, the ratio the real model uses (32/8)
HEAD_DIM    = 32  # n_embd / n_head
D_REL       = 16
# Three DISTINCT window/extent values (real model: n_swa 512, rel_extent_swa 512, rel_extent 1024).
# They used to be all 512, so swapping the global and SWA relative-index tensors, or using n_swa
# where rel_extent_swa belongs, changed nothing. Now the two attn_rel_proj shapes differ (a swap
# fails to load), the SWA mask width differs from both bias extents, and the global layer's
# 64-bucket bias runs into the zero pad column well inside a 2048-token chunk.
N_SWA          = 32   # inkling.attention.sliding_window: the SWA mask width
REL_EXTENT     = 64   # global (non-SWA) layer bias buckets
REL_EXTENT_SWA = 48   # SWA layer bias buckets; > N_SWA so every visible SWA cell has a learned bias
SHORTCONV_K = 4
DENSE_LEAD  = 1          # blk.0 dense, blk.1 MoE
N_EXPERT    = 16
N_EXPERT_USED = 6  # real model routes 6; >1 exercises the top-k path properly
N_SHEXP     = 2
N_FF_EXP    = 256
N_FF_DENSE  = 512
UNPADDED_VOCAB = 1000
LOGIT_SCALE_DENOM = 8.0
SWA_PATTERN = [1, 0]     # blk.0 local(SWA), blk.1 global
# Routing/scaling constants copied from the real header (Inkling-Small-UD-Q4_K_M, 2026-09-19):
# expert_weights_scale 8.0, expert_gating_func 2 (sigmoid), log_scaling_alpha 0.1. With the old
# scale of 1.0 a dropped ggml_scale was invisible; with n_floor 0 the tau input was never even
# built, in either build. n_floor is shrunk from the real 128000 so tau = 1 + alpha*log((pos+1)/
# n_floor) leaves 1.0 at pos 32 and reaches ~1.42 by pos 2047, i.e. inside one ppl chunk.
EXPERT_WEIGHTS_SCALE = 8.0
EXPERT_GATING_FUNC   = 2      # LLM_EXPERT_GATING_FUNC_TYPE_SIGMOID; loader reads it, graph hardcodes sigmoid
LOG_N_FLOOR = 32
LOG_ALPHA   = 0.1

writer = gguf.GGUFWriter(str(OUT), "inkling")

# --- general metadata ---
writer.add_name("inkling-test")
writer.add_uint32("inkling.block_count", N_LAYER)
writer.add_uint32("inkling.context_length", 4096)
writer.add_uint32("inkling.embedding_length", N_EMBD)
writer.add_uint32("inkling.feed_forward_length", N_FF_DENSE)
writer.add_uint32("inkling.attention.head_count", N_HEAD)
writer.add_uint32("inkling.attention.head_count_kv", N_HEAD_KV)
writer.add_float32("inkling.attention.layer_norm_rms_epsilon", 1e-5)
writer.add_uint32("inkling.rope.dimension_count", 0)  # no rope
writer.add_uint32("inkling.expert_count", N_EXPERT)
writer.add_uint32("inkling.expert_used_count", N_EXPERT_USED)
writer.add_uint32("inkling.expert_shared_count", N_SHEXP)
writer.add_uint32("inkling.expert_feed_forward_length", N_FF_EXP)
writer.add_float32("inkling.expert_weights_scale", EXPERT_WEIGHTS_SCALE)
writer.add_uint32("inkling.expert_gating_func", EXPERT_GATING_FUNC)
writer.add_uint32("inkling.attention.sliding_window", N_SWA)
writer.add_array("inkling.attention.sliding_window_pattern", SWA_PATTERN)
writer.add_uint32("inkling.d_rel", D_REL)
writer.add_uint32("inkling.rel_extent", REL_EXTENT)
writer.add_uint32("inkling.rel_extent_swa", REL_EXTENT_SWA)
writer.add_uint32("inkling.shortconv_kernel", SHORTCONV_K)
writer.add_uint32("inkling.dense_block_count", DENSE_LEAD)
writer.add_float32("inkling.logit_scale_denom", LOGIT_SCALE_DENOM)
writer.add_uint32("inkling.log_scaling_n_floor", LOG_N_FLOOR)
writer.add_float32("inkling.log_scaling_alpha", LOG_ALPHA)
writer.add_uint32("inkling.unpadded_vocab_size", UNPADDED_VOCAB)
# The real checkpoint carries inkling.vocab_size next to unpadded_vocab_size (it is in the
# D1 KV-key list of tests/bdd/features/inkling.feature). The gpt2 vocab loader ignores it
# (only the no_vocab path reads it), so it is metadata parity, not behaviour.
writer.add_uint32("inkling.vocab_size", N_VOCAB)

# tokenizer metadata (minimal BPE-ish to satisfy vocab init)
writer.add_string("tokenizer.ggml.model", "gpt2")

# Byte-level vocab for ids 0..255, so ORDINARY TEXT TOKENIZES.
# The previous vocab was "tok0".."tokN", which tokenized nothing: llama-cli exited with
# "input is empty" and llama-perplexity with "the data file ... tokenizes to only 1 tokens".
# That made the fixture load-only, which is why D2 had no cheap numerical gate and a bad
# port was not caught until a 152 GB parity run. With byte tokens the same 14 MB fixture can
# run perplexity, so two builds can be diffed against each other on a laptop-sized model.
def _gpt2_byte_encoder():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("\u00a1"), ord("\u00ac")+1)) \
       + list(range(ord("\u00ae"), ord("\u00ff")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256+n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

_be = _gpt2_byte_encoder()
tokens = [_be[b] for b in range(256)] + [f"tok{i}" for i in range(256, N_VOCAB)]
writer.add_array("tokenizer.ggml.tokens", tokens)
writer.add_array("tokenizer.ggml.scores", [float(-i) for i in range(N_VOCAB)])
writer.add_array("tokenizer.ggml.token_type", [gguf.TokenType.NORMAL] * N_VOCAB)
# The gpt2 loader REFUSES an empty merges list ("cannot find tokenizer merges in model
# file"), so supply a few pairs over rarely-adjacent high byte tokens. Byte-level fallback
# does the real work; these exist to satisfy the loader without shaping tokenization.
writer.add_array("tokenizer.ggml.merges",
                 [f"{tokens[200+i]} {tokens[201+i]}" for i in range(8)])
# BOS/EOS ids are defined, but add_bos_token is FALSE like the real model (its header carries
# add_bos_token=false with bos==eos==200006). It used to be true: every ppl chunk then opened
# with a BOS the real model never sees, and any add_bos-dependent logic went untested. The
# byte-level vocab above already guarantees ordinary text tokenizes, so nothing depends on a
# free BOS token any more; llama-cli just needs a non-empty prompt.
writer.add_uint32("tokenizer.ggml.bos_token_id", 1)
writer.add_uint32("tokenizer.ggml.eos_token_id", 2)
writer.add_bool("tokenizer.ggml.add_bos_token", False)
writer.add_bool("tokenizer.ggml.add_eos_token", False)

# --- tensors ---
def add_tensor(name, shape):
    """shape is gguf ne order (fastest-varying last) as the loader expects.

    The writer serializes the numpy shape REVERSED into ggml ne, so build the
    numpy array in the reversed order to land ne == shape in the file.

    Weights are DETERMINISTIC AND NON-ZERO, seeded per tensor name so the file is
    byte-reproducible on any host.

    This used to write np.zeros, which made the fixture useless as a correctness
    check: with every weight zero the logits are zero, softmax is uniform, and ANY
    graph -- correct or catastrophically wrong -- scores exactly PPL 1000.0000 over
    a 1024 vocab. A broken build_inkling passed this fixture and was only caught by
    a 152 GB parity run (ppl 640327 vs an expected 77). Zeros cannot distinguish
    right arithmetic from garbage; structured weights can.

    Norm-style weights sit near 1.0 (a zero RMS-norm gain annihilates the residual
    stream); everything else is small-scale noise so activations neither vanish nor
    saturate over a few layers.
    """
    npy_shape = tuple(reversed(shape))
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    if name.endswith("_norm.weight") or name.endswith("gscale.weight"):
        data = (1.0 + 0.02 * rng.standard_normal(npy_shape)).astype(np.float32)
    elif name.endswith(".bias"):
        data = (0.01 * rng.standard_normal(npy_shape)).astype(np.float32)
    else:
        data = (0.05 * rng.standard_normal(npy_shape)).astype(np.float32)
    writer.add_tensor(name, data)

# global
add_tensor("token_embd.weight", (N_EMBD, N_VOCAB))
add_tensor("token_embd_norm.weight", (N_EMBD,))
add_tensor("output_norm.weight", (N_EMBD,))
add_tensor("output.weight", (N_EMBD, N_VOCAB))

for i in range(N_LAYER):
    is_swa = bool(SWA_PATTERN[i])
    rel_extent = REL_EXTENT_SWA if is_swa else REL_EXTENT
    kvw = N_HEAD_KV * HEAD_DIM
    add_tensor(f"blk.{i}.attn_norm.weight", (N_EMBD,))
    add_tensor(f"blk.{i}.attn_q.weight", (N_EMBD, N_HEAD * HEAD_DIM))
    add_tensor(f"blk.{i}.attn_k.weight", (N_EMBD, kvw))
    add_tensor(f"blk.{i}.attn_v.weight", (N_EMBD, kvw))
    add_tensor(f"blk.{i}.attn_r.weight", (N_EMBD, N_HEAD * D_REL))
    add_tensor(f"blk.{i}.attn_output.weight", (N_HEAD * HEAD_DIM, N_EMBD))
    add_tensor(f"blk.{i}.attn_q_norm.weight", (HEAD_DIM,))
    add_tensor(f"blk.{i}.attn_k_norm.weight", (HEAD_DIM,))
    # [rel_extent, d_rel] in ne order; checkpoint stores [d_rel, E]
    add_tensor(f"blk.{i}.attn_rel_proj.weight", (rel_extent, D_REL))
    add_tensor(f"blk.{i}.shortconv_k.weight", (SHORTCONV_K, kvw))
    add_tensor(f"blk.{i}.shortconv_v.weight", (SHORTCONV_K, kvw))
    add_tensor(f"blk.{i}.shortconv_attn.weight", (SHORTCONV_K, N_EMBD))
    add_tensor(f"blk.{i}.shortconv_mlp.weight", (SHORTCONV_K, N_EMBD))
    add_tensor(f"blk.{i}.ffn_norm.weight", (N_EMBD,))
    add_tensor(f"blk.{i}.ffn_gscale.weight", (1,))
    if i < DENSE_LEAD:
        add_tensor(f"blk.{i}.ffn_gate.weight", (N_EMBD, N_FF_DENSE))
        add_tensor(f"blk.{i}.ffn_up.weight", (N_EMBD, N_FF_DENSE))
        add_tensor(f"blk.{i}.ffn_down.weight", (N_FF_DENSE, N_EMBD))
    else:
        add_tensor(f"blk.{i}.ffn_gate_inp.weight", (N_EMBD, N_EXPERT + N_SHEXP))
        add_tensor(f"blk.{i}.exp_probs_b.bias", (N_EXPERT,))
        add_tensor(f"blk.{i}.ffn_gate_exps.weight", (N_EMBD, N_FF_EXP, N_EXPERT))
        add_tensor(f"blk.{i}.ffn_up_exps.weight", (N_EMBD, N_FF_EXP, N_EXPERT))
        add_tensor(f"blk.{i}.ffn_down_exps.weight", (N_FF_EXP, N_EMBD, N_EXPERT))
        add_tensor(f"blk.{i}.ffn_gate_shexp.weight", (N_EMBD, N_FF_EXP, N_SHEXP))
        add_tensor(f"blk.{i}.ffn_up_shexp.weight", (N_EMBD, N_FF_EXP, N_SHEXP))
        add_tensor(f"blk.{i}.ffn_down_shexp.weight", (N_FF_EXP, N_EMBD, N_SHEXP))

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
print(f"wrote {OUT}")