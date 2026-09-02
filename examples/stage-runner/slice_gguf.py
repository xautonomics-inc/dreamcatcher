#!/usr/bin/env python3
# Slice a GLM-DSA split gguf into a single-file per-stage layer-window slice.
# usage: slice_gguf.py "<model_glob>" <out.gguf> <il_start> <il_end>
#   first stage (start==0) auto-includes token_embd; tail auto-includes output_norm+output.
#   block_count -> window size; nextn_predict_layers -> count of nextn blocks INSIDE the window
#   (so a tail window that includes the model nextn block keeps MTP; middle windows get 0).
import sys, re, glob
import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType

mg, out, a, b = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
shards = sorted(glob.glob(mg))
assert shards, f"no shards match {mg}"
meta = GGUFReader(shards[0], "r")
arch = meta.get_field("general.architecture").contents()
LEADING = 3
def srcget(k):
    f = meta.get_field(f"{arch}.{k}"); return f.contents() if f else None
src_blocks = srcget("block_count")
src_nextn = srcget("nextn_predict_layers") or 0
nextn_lo = src_blocks - src_nextn            # first nextn block index in the full model
nextn_in_win = max(0, min(b, src_blocks) - max(a, nextn_lo))
nlayer = b - a
dense_in_win = max(0, min(b, LEADING) - max(a, 0))
inc_embd = True
inc_output = True
print(f"[slice] arch={arch} window=[{a},{b}) nlayer={nlayer} dense={dense_in_win} nextn={nextn_in_win} (src blocks={src_blocks} nextn={src_nextn})", flush=True)

overrides = {
    f"{arch}.block_count": (nlayer, GGUFValueType.UINT32),
    f"{arch}.leading_dense_block_count": (dense_in_win, GGUFValueType.UINT32),
    f"{arch}.nextn_predict_layers": (nextn_in_win, GGUFValueType.UINT32),
}
w = GGUFWriter(out, arch, endianess=meta.endianess)
applied = set()
for f in meta.fields.values():
    if f.name == "general.architecture" or f.name.startswith("GGUF.") or f.name.startswith("split."):
        continue
    vt = f.types[0]
    st = f.types[-1] if vt == GGUFValueType.ARRAY else None
    if f.name in overrides:
        val, vt = overrides[f.name]; st = None; applied.add(f.name)
    else:
        val = f.contents()
    if val is not None:
        w.add_key_value(f.name, val, vt, sub_type=st)
for k, (v, vt) in overrides.items():
    if k not in applied:
        w.add_key_value(k, v, vt)

def newname(name):
    m = re.match(r"blk\.(\d+)\.(.*)", name)
    if m:
        i = int(m.group(1))
        return f"blk.{i-a}.{m.group(2)}" if a <= i < b else None
    if name == "token_embd.weight":   return name if inc_embd else None
    if name in ("output_norm.weight", "output.weight"): return name if inc_output else None
    return None

readers = [meta] + [GGUFReader(s, "r") for s in shards[1:]]
picked = []
for r in readers:
    for t in r.tensors:
        nn = newname(t.name)
        if nn:
            picked.append((nn, t))
print(f"[slice] {len(picked)} tensors -> {out}", flush=True)
for nn, t in picked:
    w.add_tensor_info(nn, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
tot = 0
for nn, t in picked:
    w.write_tensor_data(t.data, tensor_endianess=meta.endianess); tot += t.n_bytes
w.close()
print(f"[slice] DONE wrote {tot/1e9:.1f} GB", flush=True)
