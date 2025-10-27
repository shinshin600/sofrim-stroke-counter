@echo off
setlocal ENABLEDELAYEDEXPANSION
cd /d "C:\Users\02\Downloads\sofrim-stroke-counter-main"

:: Activate venv (PowerShell policies sometimes block .ps1)
call venv\Scripts\activate.bat

echo Checking for updates from GitHub...
git pull origin main

:: Show current branch and commit
for /f "delims=" %%b in ('git rev-parse --abbrev-ref HEAD') do set BRANCH=%%b
for /f "delims=" %%h in ('git rev-parse --short HEAD') do set SHORTHASH=%%h
echo Now on %BRANCH% @ %SHORTHASH%

:: Run app; show version in console too
python sutam_counter.py --version
start "" python sutam_counter.py --mode color_dual_rings
