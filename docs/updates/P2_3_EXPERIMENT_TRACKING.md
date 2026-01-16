# P2.3: Experiment tracking (local artifacts)

This update adds **filesystem-first experiment tracking** so each research run is:

- **Reproducible**: the exact config and full output payload are saved.
- **Comparable**: a stable per-alpha metrics table is exported.
- **Auditable**: a human-readable summary is generated.

No external service is required.

---

## What changed

### New module

- `src/agent/services/experiment_tracking.py`
  - `make_run_id(...)`: timestamp + thread_id + short config hash
  - `save_run_artifacts(...)`: writes a run directory with JSON/CSV/MD artifacts

### New CLI tools

You can run these **without** installing the package:

- `python list_runs.py --runs-root runs --limit 20`
- `python compare_runs.py runs/<A> runs/<B> --output /tmp/compare.md`
- `python replay_run.py runs/<run_id>`

(They are wrappers around `agent.tools.*`.)

---

## Run artifacts

By default, `main.py` writes artifacts to:

- `runs/<run_id>/`

Where `run_id` defaults to `YYYYMMDDTHHMMSSZ_<thread_id>_<hash>`.

Artifacts written:

- `config.json`: the `configurable` dict used by the graph
- `result.json`: the full output payload from the graph
- `alpha_metrics.csv`: flattened metrics for all coded alphas (sorted by IR)
- `sota_alphas.json`: the selected top-K SOTA alphas
- `SUMMARY.md`: a human-readable summary (best alpha + top table)
- `REPORT.md`: a deeper debugging report (cost attribution + constraints + optimizer usage)
- `daily/<alpha_id>_oos_daily.csv`: optional (top-N) OOS daily return series

The runs root also includes:

- `runs/_index.jsonl`: append-only index (one JSON per line)
- `runs/factor_registry.jsonl`: append-only factor registry (one factor per line)
- `runs/LATEST`: the most recent run_id

---

## How to use

### Run and save artifacts (default)

```bash
python main.py --idea "Momentum + liquidity" --eval-mode p2
```

### Disable saving

```bash
python main.py --save-run false
```

### Control where artifacts go

```bash
python main.py --runs-root ./my_runs
```

### Compare runs

```bash
python compare_runs.py runs/<runA> runs/<runB>
```

### Replay a run

```bash
python replay_run.py runs/<run_id>
```

Notes:
- If `OPENAI_API_KEY` is set, LLM outputs may differ between replays.
- With stubs (no API key), replay is deterministic.
