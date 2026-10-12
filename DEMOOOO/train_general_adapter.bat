@echo off
setlocal
echo ======================================================================
echo VOCALAID AI - DYSARTHRIC ASR GENERAL ADAPTER TRAINING
echo ======================================================================

set CONDA_ENV=dysarthric
set CONDA_ACTIVATE=C:\Users\Heli P Shah\ANACONDA\Scripts\activate.bat

if exist "%CONDA_ACTIVATE%" (
    echo [*] Activating Conda environment '%CONDA_ENV%'...
    call "%CONDA_ACTIVATE%" %CONDA_ENV%
) else (
    echo [!] Warning: Anaconda activate script not found at standard path.
    echo [*] Attempting to run with current environment...
)

echo.
echo [*] Training GENERALIZED adapter (all train speakers pooled)...
python step5_train.py --mode general

echo.
echo ======================================================================
echo Training command finished. Check output/logs/05_training.log for details.
echo ======================================================================
pause
