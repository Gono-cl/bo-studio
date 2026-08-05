# Contributing to BenchBO

## Scope

BenchBO is a Streamlit application for Bayesian Optimization workflows, teaching, and experiment tracking. Contributions should keep the app usable for non-programmer researchers and reproducible for reviewers.

## Before opening changes

- Open an issue when the change is non-trivial, changes behavior, or affects saved campaign formats.
- Keep pull requests focused on one concern at a time.
- Preserve backwards compatibility for stored data and saved campaign files when practical.

## Local setup

```bash
pip install -r requirements.txt
python run_bo_studio.py
```

Run tests before submitting changes:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Coding expectations

- Prefer small, reviewable changes.
- Keep UI text precise, especially in classroom sections where the teaching claims must match the implementation.
- Add or update tests when changing deterministic logic in `core/utils/`.
- Avoid committing local databases, credentials, or generated run artifacts.

## Documentation expectations

- Update `README.md`, `docs/USER_GUIDE.md`, or `PACKAGING.md` when behavior visible to users changes.
- If a change affects JOSS-readiness, update `docs/PUBLICATION_CHECKLIST.md`.

## Review checklist for contributors

- [ ] Code runs locally.
- [ ] Tests pass locally.
- [ ] New behavior is documented where needed.
- [ ] No secrets, local databases, or build artifacts are included.
