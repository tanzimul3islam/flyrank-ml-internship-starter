# Reproduce the research paper

The locked lane is **Refresh / Content Opportunity Scoring**. The primary June test improves the global top-20 decline count from 15 to 17, but the tree loses within every client with more than 20 pages. The paper reports both findings and does not recommend broad deployment.

## Local analysis

From the repository root, with Python 3.14 (the tested interpreter):

```bash
python -m venv work/.venv
work/.venv/bin/python -m pip install -r work/requirements-capstone.txt
work/.venv/bin/hf auth login
work/.venv/bin/python work/scripts/capstone_pipeline.py
work/.venv/bin/python work/scripts/build_research_paper.py
work/.venv/bin/python work/scripts/execute_capstone_notebooks.py
```

Accept access to the FlyRank warehouse before authenticating. A READ token is sufficient. Use the saved login, HF_TOKEN, or a Colab Secret; never write a token into a notebook. Run notebooks from a cloned repo, because they import the shared scripts. If using the task's ignored local login cache, set HF_HOME to its absolute `work/outputs/hf_cache` path when launching Python/Jupyter.

The pipeline pins revision `50cbf7c3909d07be4d1b5906b4d09e882e5acbf2` and reads only March and June. March is development; June is final temporal/client evaluation. The method was frozen in commit `01a5c7b`; commit `edab474` documents exact-duplicate removal before the first June model score. No tuning follows June. Reruns verify the same fixed result.

The notebook runner executes Weeks 5–7 and the capstone with saved outputs. Weeks 1–4 already carry their assignment outputs. The two optional Week 3/4 deep-dive skeletons are intentionally unfilled, not claimed as completed work.

## Artifacts

- `work/capstone_protocol.json`: frozen choices.
- `work/capstone_data_amendment.json`: grain-repair disclosure.
- `work/outputs/capstone_metrics.json`: aggregate results, data counts, split fingerprints, uncertainty and anonymous review slots.
- `work/outputs/capstone_tree.txt`: readable fitted tree.
- `work/outputs/capstone_recommendations.csv`: **ignored**, historical ranked routing queue.
- `work/paper/`: standalone public page and local chart assets.
- `work/capstone_report.md`: text companion.
- `work/figures/`: reproducible research charts (SVG/PNG).

## Check and publish

The paper verifier checks nine sections, a five-sentence abstract, ten recommendation slots, data credit and publication fingerprints. For rendering checks, install Playwright and its Chromium headless browser; it is a development check, not a dependency of the deployed page.

```bash
work/.venv/bin/python -m pip install playwright
PLAYWRIGHT_BROWSERS_PATH="$PWD/work/outputs/publish_auth/browsers" work/.venv/bin/python -m playwright install chromium --only-shell
work/.venv/bin/python work/scripts/verify_paper.py
```

If using Playwright's default browser directory, set PLAYWRIGHT_BROWSERS_PATH appropriately or remove the task-cache override in your environment. The provided verification script defaults to the ignored task browser cache.

After committing the reviewed artifacts, publish with a GitHub login for `tanzimul3islam` that can push and manage Pages:

```bash
python3 work/scripts/publish_paper.py
```

This enables the Pages workflow, pushes the commit, waits for the matching `publication.json`, verifies the page and FlyRank credit, and **only then** writes and commits the exact one-line URL in `submission/paper_url.txt`. It never prints credentials or changes your active GitHub account. An existing custom domain is preserved. The workflow deploys only `work/paper`, never the raw caches or repository contents.

If deployment needs more time, run `python3 work/scripts/publish_paper.py --verify-only` later. A local HTML build is not a completed public submission.
