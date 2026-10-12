@echo off
setlocal
echo ======================================================================
echo VOCALAID AI - QUICK SMOKE TEST (GENERAL ADAPTER)
echo Runs 1 epoch on a small subset to verify convergence in 2-3 minutes
echo ======================================================================

set CONDA_ENV=dysarthric
set CONDA_ACTIVATE=C:\Users\Heli P Shah\ANACONDA\Scripts\activate.bat

if exist "%CONDA_ACTIVATE%" (
    echo [*] Activating Conda environment '%CONDA_ENV%'...
    call "%CONDA_ACTIVATE%" %CONDA_ENV%
)

echo.
echo [*] Running smoke test...
python step5_train.py --mode general --smoke_test

echo.
echo ======================================================================
echo Smoke test finished!
echo ======================================================================
pause
