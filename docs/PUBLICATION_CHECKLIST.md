# JOSS Submission Checklist

Checklist for submitting `BenchBO` to the Journal of Open Source Software (JOSS).

This list is aligned with the current JOSS author and review guidance checked on August 4, 2026.

## Pre-review gates

- [ ] Repository has been public for more than 6 months before submission.
- [ ] Public development history shows iterative work over time, not a short code dump.
- [ ] The software is already used in research workflows, with evidence you can point to.
- [ ] The repository is public, cloneable without registration, and has a public issue tracker.
- [ ] The software is open source under an OSI-approved license.

## Open-source practice signals

- [x] `LICENSE` is present.
- [ ] `CONTRIBUTING.md` is present and reflects how outside users can contribute.
- [ ] Tagged release exists for the submission version.
- [ ] Changelog or release notes exist for the submission version.
- [ ] Automated tests cover critical logic paths.
- [ ] CI runs tests automatically on push and pull request.

## Documentation and reproducibility

- [x] `README.md` explains purpose, installation, and main workflows.
- [x] User-facing documentation exists (`docs/USER_GUIDE.md`).
- [x] Packaging/build documentation exists (`PACKAGING.md`).
- [ ] A fresh clone can be installed by following the documented steps exactly.
- [ ] A reviewer can run the software locally without hidden credentials or manual patching.
- [x] Example workflows are documented clearly enough for a reviewer to reproduce key outputs.

## JOSS paper package

- [ ] Add `paper.md` and `paper.bib` to the repository.
- [ ] `paper.md` includes: Summary, Statement of need, State of the field, Software design, Research impact statement, AI usage disclosure, Acknowledgements, and References.
- [ ] The paper is about the software, not about new scientific results produced with it.
- [ ] Related tools are compared fairly and specifically.
- [ ] Research impact claims are backed by real usage, not future plans.
- [ ] AI usage disclosure is complete and accurate for code, docs, and paper writing.

## Metadata and release alignment

- [ ] `CITATION.cff` author list matches the JOSS paper author list.
- [ ] `CITATION.cff` version matches the tagged release.
- [ ] Release date metadata is correct.
- [ ] Archive the release and mint a DOI (for example via Zenodo) before final publication.

## Final validation before submission

- [ ] Run the local test suite successfully from a clean checkout.
- [ ] Confirm the Windows executable build still succeeds from source.
- [ ] Confirm the main Streamlit app flows work end-to-end on a clean machine.
- [ ] Confirm any claims in the paper are directly supported by repo contents or cited references.

## BenchBO specific notes

- The repository currently still needs a JOSS paper package (`paper.md`, `paper.bib`).
- Reviewer walkthrough is documented in `docs/EXAMPLE_WORKFLOW.md`.
- Evidence of research use should be prepared before submission.
- Tests and CI should exist before submission; reviewers now treat these as core good-practice signals.
