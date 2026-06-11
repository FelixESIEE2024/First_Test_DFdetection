@echo off
setlocal

set "ARCHIVE_DIR=%~dp0"
for %%I in ("%ARCHIVE_DIR%..\..") do set "REPO_ROOT=%%~fI"
cd /d "%REPO_ROOT%"

if exist "%REPO_ROOT%\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%REPO_ROOT%\.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" -m streamlit run "archives\streamlit\app.py"
