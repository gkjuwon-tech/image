# Light Field Project

Goal: photo → light (radiance field) → photo, indistinguishable from reality.
Phase 0 = reconstruction quality ceiling. Generation comes later.

- `infra/runner.sh` — pull-based job runner on the GPU box (polls this branch, runs new `jobs/*.sh`, output to stdout / Vast logs). No inbound ports.
- `jobs/` — numbered job scripts, each runs once.
- `scripts/` — local helpers (launch instance, read logs).

## Wave Session Deliverable (`agy-wave-session`)

The `wave_session/` directory contains the complete codebase, training logs, evaluation benchmarks, visual comparisons, and architectural post-mortem for the 3D continuous wave packet project.

- **Full Report**: [`wave_session/SUMMARY.md`](wave_session/SUMMARY.md)
- **Codebase**: [`wave_session/code/`](wave_session/code/) (Restorer, Generator, Training)
- **Metrics & Logs**: [`wave_session/metrics_and_logs/`](wave_session/metrics_and_logs/) (PSNR / SSIM / LPIPS traces)
- **Visual Comparisons**: [`wave_session/visual_results/`](wave_session/visual_results/)
- **Plans & Docs**: [`wave_session/plans_and_docs/`](wave_session/plans_and_docs/)
