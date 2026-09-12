# gslot — global slot compute arbiter

Host-local arbiter for CPU/GPU co-tenancy on multi-model inference hosts. It
hands out whole-core partitions and timed device turns so co-resident llama.cpp
tenants stop oversubscribing each other, and it exposes read-only observation
endpoints so a dashboard can see what is running where.

Design: [`docs/SPEC-016-global-slot-compute-scheduler.md`](docs/SPEC-016-global-slot-compute-scheduler.md).
Worked deployment recipe + measurements: [`docs/runbook-gslot-expert-server.md`](docs/runbook-gslot-expert-server.md).

## Install is a directory copy

**stdlib-only by contract.** A GPU host here has no uv, no venv and often no
pip, so the package may never gain a third-party import — `dependencies` in
`pyproject.toml` is empty on purpose, not by omission.

```sh
scp -r tools/gslot host:/opt/gslot      # that is the whole install
python3 -m gslot --socket /run/gslotd.sock --http 127.0.0.1:8099 --reserve-cpus 0-3
```

The daemon logs its measured topology on start, then serves two planes:

* **control** — the AF_UNIX socket. Tenants register, request turns, heartbeat.
  Filesystem access to the socket *is* the authorisation model.
* **observation** — read-only HTTP on loopback: `/occupancy`, `/tenants`,
  `/resources`, `/inventory`, `/healthz`. There is no write path on this plane
  at all, so a fleet-wide scraper cannot actuate anything.

Inert by default: zero tenants, zero grants, no autostart unit. It does nothing
until a process connects to the socket and registers.

## Layout

| path | what |
|---|---|
| `gslot/` | the package — `python3 -m gslot` runs the daemon from here |
| `gslot-run` | tenant launcher: registers a lease, then execs a command pinned to what the arbiter granted it |
| `c/gslot_client.h` | C tenant client (speaks the NDJSON protocol) |
| `bench/` | load drivers and rig benchmarks |
| `tests/` | unit + daemon tests |
| `docs/` | the spec and the runbook |

The wire protocol on the unix socket is NDJSON and is **byte-frozen**: the C
client in `c/` and the stage-server gate are not being changed, so the framing
must not move either. See `gslot/protocol.py`.

## Development

```sh
pip install -e '.[dev]'    # pytest, ruff, mypy — never imported by the package
ruff check . && mypy --strict gslot/ && python3 -m pytest tests/ -q
```

## Provenance

Moved out of prism (repo `prism`, `src/prism/gslot/` + its tests and tools) into
this repository so the scheduler, its C client and its benchmarks have one home
(meta#58). Prism's copy stays in place with a pointer here until the operator
cuts the deployed arbiters over; the code is identical apart from the import
namespace (`prism.gslot` → `gslot`) and two docstring corrections.
