@echo off
setlocal
echo ======================================================================
echo VOCALAID AI - RUN COMPLETE PIPELINE (STEPS 1 TO 5)
echo ======================================================================

set CONDA_ENV=dysarthric
set CONDA_ACTIVATE=C:\Users\Heli P Shah\ANACONDA\Scripts\activate.bat

if exist "%CONDA_ACTIVATE%" (
    echo [*] Activating Conda environment '%CONDA_ENV%'...
    call "%CONDA_ACTIVATE%" %CONDA_ENV%
)

echo.
echo [*] Starting pipeline steps 1 to 5...
python run_pipeline.py

echo.
echo ======================================================================
echo Pipeline complete!
echo ======================================================================
pause
