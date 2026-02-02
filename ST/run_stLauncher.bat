@echo off
setlocal

rem STフォルダ（stLauncher.py と st_env がある場所）を固定
set ST_DIR=C:\Users\edu01\Documents\GitHub\Shuron_YR\ST

pushd "%ST_DIR%"

"%ST_DIR%\st_env\Scripts\python.exe" "%ST_DIR%\stLauncher.py"

pause
popd
endlocal
