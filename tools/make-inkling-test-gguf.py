#!/usr/bin/env python3
"""Build a synthetic Inkling-arch GGUF for the D1 load-only smoke.

Sized small (2 layers: 1 dense lead + 1 MoE) but with the exact metadata keys
and tensor shapes the reference loader expects, so llama-cli --load can verify
that every Inkling tensor is created with the right shape/type on CPU.
"""
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
HEAD_DIM    = 32  # n_embd / n_head
D_REL       = 16
REL_EXTENT  = 512
REL_EXTENT_SWA = 512
SHORTCONV_K = 4
DENSE_LEAD  = 1          # blk.0 dense, blk.1 MoE
N_EXPERT    = 8
N_SHEXP     = 2
N_FF_EXP    = 256
N_FF_DENSE  = 512
UNPADDED_VOCAB = 1000
LOGIT_SCALE_DENOM = 8.0
SWA_PATTERN = [1, 0]     # blk.0 local(SWA), blk.1 global

writer = gguf.GGUFWriter(str(OUT), "inkling")

# --- general metadata ---
writer.add_name("inkling-test")
writer.add_uint32("inkling.block_count", N_LAYER)
writer.add_uint32("inkling.context_length", 4096)
writer.add_uint32("inkling.embedding_length", N_EMBD)
writer.add_uint32("inkling.feed_forward_length", N_FF_DENSE)
writer.add_uint32("inkling.attention.head_count", N_HEAD)
writer.add_uint32("inkling.attention.head_count_kv", N_HEAD)
writer.add_float32("inkling.attention.layer_norm_rms_epsilon", 1e-5)
writer.add_uint32("inkling.rope.dimension_count", 0)  # no rope
writer.add_uint32("inkling.expert_count", N_EXPERT)
writer.add_uint32("inkling.expert_used_count", 2)
writer.add_uint32("inkling.expert_shared_count", N_SHEXP)
writer.add_uint32("inkling.expert_feed_forward_length", N_FF_EXP)
writer.add_float32("inkling.expert_weights_scale", 1.0)
writer.add_uint32("inkling.attention.sliding_window", REL_EXTENT_SWA)
writer.add_array("inkling.attention.sliding_window_pattern", SWA_PATTERN)
writer.add_uint32("inkling.d_rel", D_REL)
writer.add_uint32("inkling.rel_extent", REL_EXTENT)
writer.add_uint32("inkling.rel_extent_swa", REL_EXTENT_SWA)
writer.add_uint32("inkling.shortconv_kernel", SHORTCONV_K)
writer.add_uint32("inkling.dense_block_count", DENSE_LEAD)
writer.add_float32("inkling.logit_scale_denom", LOGIT_SCALE_DENOM)
writer.add_uint32("inkling.log_scaling_n_floor", 0)
writer.add_float32("inkling.log_scaling_alpha", 0.0)
writer.add_uint32("inkling.unpadded_vocab_size", UNPADDED_VOCAB)

# tokenizer metadata (minimal BPE-ish to satisfy vocab init)
writer.add_string("tokenizer.ggml.model", "gpt2")
writer.add_array("tokenizer.ggml.tokens", [f"tok{i}" for i in range(N_VOCAB)])
writer.add_array("tokenizer.ggml.scores", [float(-i) for i in range(N_VOCAB)])
writer.add_array("tokenizer.ggml.token_type", [gguf.TokenType.NORMAL] * N_VOCAB)
# gpt2 BPE needs merges; provide a minimal pair set so vocab init succeeds
writer.add_array("tokenizer.ggml.merges", [f"tok0 tok{i}" for i in range(1, min(64, N_VOCAB))])

# --- tensors ---
def add_tensor(name, shape):
    """shape is gguf ne order (fastest-varying last) as the loader expects.

    The writer serializes the numpy shape REVERSED into ggml ne, so build the
    numpy array in the reversed order to land ne == shape in the file.
    """
    npy_shape = tuple(reversed(shape))
    data = np.zeros(npy_shape, dtype=np.float32)
    writer.add_tensor(name, data)

# global
add_tensor("token_embd.weight", (N_EMBD, N_VOCAB))
add_tensor("token_embd_norm.weight", (N_EMBD,))
add_tensor("output_norm.weight", (N_EMBD,))
add_tensor("output.weight", (N_EMBD, N_VOCAB))

for i in range(N_LAYER):
    is_swa = bool(SWA_PATTERN[i])
    rel_extent = REL_EXTENT_SWA if is_swa else REL_EXTENT
    kvw = N_HEAD * HEAD_DIM
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