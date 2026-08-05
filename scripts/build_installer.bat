@echo off
setlocal

REM Build BenchBO executable and create Windows installer using Inno Setup.

cd /d "%~dp0\.."

call scripts\build_windows.bat
if errorlevel 1 goto :fail

set "ISCC_EXE="
where iscc >nul 2>&1
if not errorlevel 1 set "ISCC_EXE=iscc"
if not defined ISCC_EXE (
    if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC_EXE=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
)
if not defined ISCC_EXE (
    if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC_EXE=%ProgramFiles%\Inno Setup 6\ISCC.exe"
)
if not defined ISCC_EXE (
    echo Inno Setup compiler not found.
    echo Install Inno Setup 6 and add ISCC to PATH, then rerun this script.
    exit /b 1
)

call "%ISCC_EXE%" installer\BenchBO.iss
if errorlevel 1 goto :fail

echo Installer created in dist\.
goto :eof

:fail
echo Installer build failed.
exit /b 1
