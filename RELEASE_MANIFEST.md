# Release manifest: main diploma snapshot

- Repository: https://github.com/vValkroVv/unlearning-diploma
- Release branch: `main`
- Commit SHA: cite the exact `main` commit used in the diploma
- Upstream repository: https://github.com/locuslab/open-unlearning
- Upstream base: research fork derived from OpenUnlearning; this public repo was reinitialized from the `dualfc_v2_6` working snapshot
- License: MIT

## Main workflow

- Main runbook: `prod-run-dual-gpu.md`
- Public reproducibility guide: `docs/diploma_repro.md`
- Artifact validators: `src/tools/validate_dual_cf_artifact.py`
- Campaign wrapper: `scripts/dualcf/run_campaign_one_lr.sh`

## Not stored in Git

- Base model weights
- SFT checkpoints
- Local dataset mirrors
- Generated counterfactual artifacts
- Full training checkpoints
- Private logs or API keys

## External artifacts

If public result artifacts are provided, list links here:

- Summary metrics archive: not provided in Git
- Clean saves archive: not provided in Git
