# Light Field Project

Goal: photo → light (radiance field) → photo, indistinguishable from reality.
Phase 0 = reconstruction quality ceiling. Generation comes later.

- `infra/runner.sh` — pull-based job runner on the GPU box (polls this branch, runs new `jobs/*.sh`, output to stdout / Vast logs). No inbound ports.
- `jobs/` — numbered job scripts, each runs once.
- `scripts/` — local helpers (launch instance, read logs).
