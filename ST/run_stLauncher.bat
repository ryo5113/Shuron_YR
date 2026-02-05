@echo off
setlocal

rem bat自身のフォルダ（=保存先にしたい場所）
set ROOT=%~dp0

rem スクリプト/venvが入っているSTフォルダ（batと同階層に ST がある前提）
set ST_DIR=%ROOT%ST
set VENV_PY=%ST_DIR%\st_env\Scripts\python.exe

rem ここが「保存先」になる（Path.cwd()）
pushd "%ROOT%"

rem 存在チェック
if not exist "%ST_DIR%\stLauncher.py" (
  echo [ERROR] stLauncher.py not found: "%ST_DIR%\stLauncher.py"
  pause
  exit /b 1
)
if not exist "%VENV_PY%" (
  echo [ERROR] venv python not found: "%VENV_PY%"
  pause
  exit /b 1
)

rem STをsys.pathに追加して、stLauncher.pyをスクリプトとして実行
"%VENV_PY%" -c "import sys, runpy; sys.path.insert(0, r'%ST_DIR%'); runpy.run_path(r'%ST_DIR%\stLauncher.py', run_name='__main__')"

pause
popd
endlocal
