# Publication Checklist

checklist before submitting BO Studio to a software journal .

## Repository and licensing

- [ ] Public repository with stable default branch.
- [ ] Open-source license file present (for example MIT, BSD-3-Clause, Apache-2.0).
- [ ] Clear contribution guidelines and issue templates (optional but recommended).

## Documentation quality

- [ ] `README.md` explains purpose, installation, and core workflows.
- [ ] User documentation is complete (`docs/USER_GUIDE.md`).
- [ ] Packaging instructions are reproducible (`PACKAGING.md`).
- [ ] All key pages/features in the app are documented.

## Reproducibility and quality

- [ ] Dependency versions are pinned where needed for reproducibility.
- [ ] Basic automated tests exist for critical logic paths.
- [ ] CI pipeline runs tests and basic checks on push/PR.
- [ ] Example data or reproducible example workflow is included.

## Release preparation

- [ ] Create a tagged release in GitHub (or equivalent).
- [ ] Update `CITATION.cff` with final author metadata and version.
- [ ] Add release notes summarizing features and fixes.
- [ ] Archive release and generate DOI (for example via Zenodo).

## Journal submission package

- [ ] Statement of need is explicit and evidence-based.
- [ ] Comparison with related tools is included.
- [ ] Usage examples reflect realistic workflows.
- [ ] References are complete and formatted.
- [ ] Submission paper metadata matches repository metadata.

## Final validation

- [ ] Fresh clone install works using documented steps.
- [ ] Windows executable build succeeds from scripts.
- [ ] Core app flows run without manual patching.
- [ ] A reviewer can follow docs and reproduce key outputs.
