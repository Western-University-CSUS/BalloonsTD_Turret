# AGENTS.md

Pinpoint: CV-aimed turret that finds and pops balloons. Camera → Detector → Tracker → Controller. See README.md.

## Talking to the user

- Terse. Sacrifice grammar for concision.

## Setup and checks

- Python 3.11. `pip install -e ".[dev]"`
- Before every commit: `ruff check .` and `pytest` pass (CI runs both on every PR).
- Ruff config in pyproject.toml (line length 100).

## Git

- Commit small and often, one logical change each.
- Commit as the user. No `Co-Authored-By` or "Generated with" lines in commits or PRs.
- Branch off `main`, open a PR to `main`. Don't push to `main` directly.

## Layout

- `src/pinpoint/`: package (camera, detectors, tracker, controller)
- `scripts/`: runnable tools. `benchmarks/`: scoring detectors. `tests/`: pytest.
- `docs/`: shared docs. `docs.local/`: personal, gitignored, never commit.
- Footage (`footage/`, `*.mp4`) and model weights (`*.pt`, `*.onnx`, ...) are gitignored; share via team drive.

## Project rules

- Detectors are interchangeable: classical and YOLO return the same detection format; the tracker doesn't care which one feeds it.
- Write core algorithms (Kalman filter, control loop) by hand before reaching for libraries.
- Classical detector stays as the safe fallback; YOLO work may be experimental.
- Saved footage is raw: no overlays burned into frames.

## Agent skills

### Issue tracker

GitHub Issues on this repo, via `gh`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five roles. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context. See `docs/agents/domain.md`.
