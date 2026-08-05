@echo off
setlocal

REM Build BenchBO as a portable Windows app (no Python required for end users).
REM Optional override:
REM   set BENCHBO_PYTHON=C:\full\path\to\python.exe

cd /d "%~dp0\.."

set "PY_EXE="
set "PY_ARGS="
if defined BENCHBO_PYTHON (
    if exist "%BENCHBO_PYTHON%" (
        set "PY_EXE=%BENCHBO_PYTHON%"
        set "PY_ARGS="
    )
)
if not defined PY_EXE (
    if defined BOSTUDIO_PYTHON (
        if exist "%BOSTUDIO_PYTHON%" (
            set "PY_EXE=%BOSTUDIO_PYTHON%"
            set "PY_ARGS="
        )
    )
)
if not defined PY_EXE (
    if exist "%USERPROFILE%\AppData\Local\anaconda4\envs\Streamlit_VOL\python.exe" (
        set "PY_EXE=%USERPROFILE%\AppData\Local\anaconda4\envs\Streamlit_VOL\python.exe"
        set "PY_ARGS="
    )
)
if not defined PY_EXE (
    py -3 -V >nul 2>&1
    if not errorlevel 1 (
        set "PY_EXE=py"
        set "PY_ARGS=-3"
    )
)
if not defined PY_EXE (
    python -V >nul 2>&1
    if not errorlevel 1 (
        set "PY_EXE=python"
        set "PY_ARGS="
    )
)
if not defined PY_EXE goto :no_python

echo [1/4] Installing build dependencies...
call "%PY_EXE%" %PY_ARGS% -m pip install --upgrade pip
if errorlevel 1 goto :fail
call "%PY_EXE%" %PY_ARGS% -m pip install -r requirements.txt pyinstaller
if errorlevel 1 goto :fail

echo [2/4] Cleaning previous artifacts...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo [3/4] Building executable with PyInstaller...
call "%PY_EXE%" %PY_ARGS% -m PyInstaller --noconfirm --clean BenchBO.spec
if errorlevel 1 goto :fail

echo [4/4] Build completed.
echo Output folder: dist\BenchBO
echo Launch file : dist\BenchBO\BenchBO.exe
goto :eof

:no_python
echo Python launcher not found. Install Python or ensure "python" or "py" is in PATH.
exit /b 1

:fail
echo Build failed.
exit /b 1
