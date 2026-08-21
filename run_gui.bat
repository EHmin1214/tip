@echo off
chcp 65001 >nul
setlocal
rem -- TIP interactive web app launcher (double-click to run) -----------------
rem
rem KEEP THIS FILE PURE ASCII.  cmd.exe tracks its read position in this file
rem as a byte offset but decodes the text with the current console codepage.
rem The "chcp 65001" above therefore shifts the parser mid-file if any
rem non-ASCII byte follows: the "rem" prefixes get eaten and the tails of the
rem comments are executed as commands.  That is exactly what happened when
rem these comments were written in Korean -- and it only failed on some runs,
rem depending on the codepage the console happened to start in.
rem
rem An older version also hard-coded the Sim4Life 9.0 bundled Python by
rem absolute path, which broke on this machine.  We now probe the usable
rem environments in order.  Sim4Life is NOT needed to run the GUI.
set PYTHONIOENCODING=utf-8
set "HERE=%~dp0"
set "APP=%HERE%src\tip\gui\app.py"

echo [tip-gui] starting server... a browser will open at http://127.0.0.1:8765
echo [tip-gui] press Ctrl+C in this window to stop

rem 1st: dedicated conda env, independent of Sim4Life (recommended in SETUP.md)
set "ENVDIR=%USERPROFILE%\miniconda3\envs\tip"
set "PY=%ENVDIR%\python.exe"
if exist "%PY%" goto :conda

rem 2nd: venv built from the Sim4Life 9.6 bundle (numpy 1.26.4, scipy 1.14.1)
set "PY=%HERE%..\.venv-s4l\Scripts\python.exe"
if exist "%PY%" goto :run

rem 3rd: the Sim4Life 9.6 bundled Python itself
set "PY=C:\Program Files\Sim4Life_9.6\Python\python.exe"
if exist "%PY%" goto :run

echo.
echo [tip-gui] error: no usable Python environment found.
echo           numpy and scipy are required, for example:
echo             conda create -n tip python=3.11 numpy=1.26.4 scipy matplotlib
echo.
pause
exit /b 1

:conda
rem Calling the env's python.exe by absolute path is not enough: numpy and
rem scipy load DLLs from the env's Library\bin, which only "conda activate"
rem puts on PATH.  Without this the launcher had to be run from an already
rem activated shell.  These are the same directories activation prepends, and
rem the "setlocal" above keeps them out of the calling shell.
set "PATH=%ENVDIR%;%ENVDIR%\Library\mingw-w64\bin;%ENVDIR%\Library\usr\bin;%ENVDIR%\Library\bin;%ENVDIR%\Scripts;%ENVDIR%\bin;%PATH%"
set "CONDA_PREFIX=%ENVDIR%"
set "CONDA_DEFAULT_ENV=tip"
goto :run

:run
echo [tip-gui] using: %PY%
"%PY%" "%APP%"
pause
