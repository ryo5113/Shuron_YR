@echo off
setlocal

pushd "%~dp0"

rem 仮想環境の python を直接指定（確実）
set VENV_PY=%~dp0st_env\Scripts\python.exe

rem どの python が使われるか確認
"%VENV_PY%" -c "import sys; print(sys.executable)"

rem 起動
"%VENV_PY%" stLauncher.py

pause
popd
endlocal
