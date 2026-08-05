# BenchBO

[![DOI](https://zenodo.org/badge/1023034369.svg)](https://doi.org/10.5281/zenodo.21808694)

![BenchBO logo](images/benchbo_logo_lockup_tight.png)

*A chemistry-first platform for Bayesian Optimization*

BenchBO is a local-first application for Bayesian Optimization (BO) in experimental workflows. It combines interactive optimization tools, teaching modules, and local experiment tracking in a single interface, with a strong emphasis on chemistry-oriented use cases.

The project is designed for:

- researchers running iterative experimental campaigns,
- students learning how Bayesian Optimization works,
- collaborators who need a graphical workflow instead of a code-first interface.

BenchBO can be used in two ways:

- from source with Python and Streamlit,
- as a packaged Windows desktop application for users who do not need a Python installation.

Recommended distribution model:

- end users download a prebuilt Windows release from the shared portable package,
- developers and reviewers can still run the software from source.

## Why BenchBO

Many BO libraries are powerful but code-centric. BenchBO focuses on a different need: helping users understand, configure, run, and review BO campaigns through a graphical interface while keeping data local and workflows reproducible.

This makes the software useful both for practical campaign support and for teaching the logic of BO in experimental science.

## What the software includes

Main application sections:

- `Home`: project overview and entry point.
- `Single Objective Optimization`: interactive BO campaigns for one target response.
- `Multi Objective Optimization`: tradeoff analysis and Pareto-oriented workflows.
- `Data Analysis`: inspection and comparison of campaign results.
- `Bayesian Optimization Classroom`: guided modules for BO intuition, mechanics, chemist workflow, and multiobjective decisions.
- `Experiment Database`: local storage and retrieval of experiment records.

Core workflow features:

- interactive control of initialization strategy, acquisition function, and campaign budget,
- save, resume, and reuse support for campaign workflows,
- chemistry-first teaching examples for 4D optimization problems,
- local database-backed experiment tracking,
- Windows executable packaging for non-programmer users.

## Installation From Source

Prerequisites:

- Python 3.10+
- `pip`

If you already have a local copy of the repository, start at dependency installation.

```bash
git clone <repository-url>
cd BO_Studio
pip install -r requirements.txt
python run_bo_studio.py
```

Alternative direct Streamlit launch:

```bash
streamlit run main.py
```

Notes:

- `python run_bo_studio.py` is the recommended launcher for normal local use.
- When launched from a terminal, that terminal remains occupied while the Streamlit server is running.

## First Reviewer Run

For a no-credentials, no-lab-hardware verification path:

1. Launch BenchBO.
2. Open `Bayesian Optimization Classroom`.
3. Run `4) Chemist Workflow` once with the default settings.
4. Run `5) Multiobjective Decisions` once with fixed `0.50 / 0.50` weights.
5. Confirm that tables, metrics, and plots are rendered in both sections.

The full step-by-step walkthrough is documented in `docs/EXAMPLE_WORKFLOW.md`.

## Windows Desktop App

For most end users, the recommended route is to download a prebuilt Windows release instead of building the app locally.

Download:

[BenchBO v0.1.0 Windows portable release (.zip)](https://fraunhofer-my.sharepoint.com/:u:/g/personal/gonzalo_araya_vargas_ict_fraunhofer_de/IQADfLmFb8VyRZF9KsuBv1JkAWRjZGiStnWK5JWl-TO5OeY?e=8CY8RX)

After extracting the zip, run:

`BenchBO\BenchBO.exe`

Important:

- Extract the full zip before launching the application.
- The packaged app folder includes `QUICK_START.txt` for new users.
- Share the whole portable app folder or a zip of it, not only `BenchBO.exe`.
- Keep `BenchBO.exe` together with the bundled `_internal` folder.
- No Python installation is required on the target machine.
- If you distribute the installer build instead, users can run `dist\BenchBO-Setup.exe`.

## Data Storage

BenchBO is local-first:

- in source mode, persistent data is stored under the current working directory,
- in packaged desktop mode, persistent data is stored next to the executable,
- advanced users can override the storage location with the `BENCHBO_STORAGE_ROOT` environment variable.

Persistent outputs include:

- `data/experiments.db`
- `resumable_manual_runs/`

## Build The Windows Release

This section is mainly for developers preparing a new packaged release.

Portable app build:

```bat
scripts\build_windows.bat
```

Installer build:

```bat
scripts\build_installer.bat
```

Main outputs:

- `dist\BenchBO\BenchBO.exe`
- `dist\BenchBO-Setup.exe`

See `PACKAGING.md` for packaging details.

## Documentation

- Bundled executable guide: `QUICK_START.txt`
- User guide: `docs/USER_GUIDE.md`
- Example workflow: `docs/EXAMPLE_WORKFLOW.md`
- Packaging guide: `PACKAGING.md`
- Publication checklist: `docs/PUBLICATION_CHECKLIST.md`
- Citation metadata: `CITATION.cff`
- Contribution guide: `CONTRIBUTING.md`

## Repository Layout

- `main.py`: main app entry point and sidebar navigation.
- `run_bo_studio.py`: recommended launcher and desktop bootstrap.
- `navigation/`: app pages.
- `ui/`: reusable interface components.
- `core/`: BO logic, utilities, database helpers, and storage path helpers.
- `tests/`: automated tests for critical logic.
- `scripts/`: build scripts for desktop packaging.
- `installer/`: installer configuration.
- `docs/`: user and publication-oriented documentation.

## Citation

If you use BenchBO in teaching, research, or experimental workflow development, cite the archived software release described in `CITATION.cff`.

Current Zenodo identifiers:

- Version DOI for `v0.1.0`: [10.5281/zenodo.21808695](https://doi.org/10.5281/zenodo.21808695)
- Concept DOI for all BenchBO versions: [10.5281/zenodo.21808694](https://doi.org/10.5281/zenodo.21808694)

For release archiving and DOI preparation, see `docs/PUBLICATION_CHECKLIST.md`.

## License

BenchBO is distributed under the MIT License. See `LICENSE`.
