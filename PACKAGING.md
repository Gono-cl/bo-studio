# Desktop Packaging (Windows)

This project can be shipped as a desktop app so end users do not need Python installed.

## Prerequisites (build machine only)

- Windows 10/11
- Python 3.10+ available in `PATH`
- Inno Setup (only if you want an installer `.exe`)

## Build portable app folder

From repository root:

```bat
scripts\build_windows.bat
```

Output:

- `dist\BenchBO\BenchBO.exe`

Users can run `BenchBO.exe` directly from that folder.

## Build installer `.exe`

From repository root:

```bat
scripts\build_installer.bat
```

Output:

- `dist\BenchBO-Setup.exe`

## Notes

- Entry point is `run_bo_studio.py` (boots Streamlit app in desktop mode).
- Packaging config is `BenchBO.spec`.
- If antivirus flags the binary, sign the executable/installer in your release process.
- Rebuild after dependency changes or when adding new modules/assets.

