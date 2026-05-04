# Open Source Release Checklist

- Keep datasets, checkpoints, result tables, and generated metadata out of git.
- Configure local paths through CLI flags or environment variables such as `ROOT_DIR`, `DATA_DIR`, `MODEL_BASE`, and `PRETRAINED_MODEL_PATH`.
- Run `rg -n "(/home/|/scratch|/project/|/archive/|/datasets/)"` before publishing.
- Run `git status --short` and review every tracked addition before release.
- If publishing this repository's existing history, rewrite history or publish from a fresh clean export so removed data artifacts are not still present in old commits.
