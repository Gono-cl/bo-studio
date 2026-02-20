# BO Studio

BO Studio is a local-first Streamlit application for Bayesian Optimization (BO) in experimental workflows, with a strong focus on chemistry-oriented use cases.

It combines:
- interactive single-objective optimization campaigns,
- interactive multi-objective optimization campaigns,
- a guided BO classroom (beginner and advanced),
- data analysis and experiment database utilities,
- Windows desktop packaging (`.exe`) so end users do not need Python.

## Why this project

BO Studio is designed for users who want to run and understand BO without writing code during day-to-day campaign execution. The app supports both learning and practical decision-making in iterative experiments.

## Main capabilities

- Manual single-objective BO campaign management.
- Multi-objective optimization with Pareto-based analysis.
- Save/resume/reuse campaign workflows.
- BO Classroom with intuition, mechanics, and chemistry-oriented explanations.
- Database-backed experiment storage and retrieval.
- Windows packaging for distribution to non-programmer users.

## Quick start (local source run)

### 1. Clone

```bash
git clone <your-repo-url>
cd BO_Studio
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

Recommended launcher:

```bash
python run_bo_studio.py
```

Alternative direct Streamlit run:

```bash
streamlit run main.py
```

## Quick start (Windows executable)

If you already built the desktop app, run:

`dist\BOStudio\BOStudio.exe`

No Python installation is required on the target machine.

## Build the Windows executable

From repository root:

```bat
scripts\build_windows.bat
```

Output:
- `dist\BOStudio\BOStudio.exe`

For installer packaging, see:
- `PACKAGING.md`

## Documentation

- GitHub Pages docs: https://gono-cl.github.io/bo-studio/index.html
- User guide: `docs/USER_GUIDE.md`
- Packaging guide: `PACKAGING.md`
- Publication checklist: `docs/PUBLICATION_CHECKLIST.md`
- Citation metadata: `CITATION.cff`

## Project layout

- `main.py`: app entrypoint and navigation.
- `run_bo_studio.py`: local launcher and desktop bootstrap.
- `navigation/`: top-level pages (home, classroom, optimization pages, analysis, database).
- `ui/`: reusable UI sections/components.
- `core/`: optimization logic, utilities, database helpers.
- `scripts/`: build scripts for executable/installer.
- `installer/`: Inno Setup configuration.

## Notes for stable local development

- Prefer `python run_bo_studio.py` for local testing.
- If you suspect stale sessions, close existing BO Studio windows/processes before relaunching.
- When the app is started from terminal, that terminal remains occupied while the server runs (normal Streamlit behavior).

## Journal/publication readiness checklist (high level)

To prepare this repository for software publication (for example JOSS), complete:

- Open-source license file (MIT/BSD/Apache-2.0).
- Stable tagged release.
- Clear installation and usage docs (this README + user guide).
- Basic automated tests and CI.
- Citation metadata (`CITATION.cff`) and optional DOI workflow (for example Zenodo).
