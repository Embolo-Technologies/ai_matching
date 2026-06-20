@echo off
rem -----------------------------------------------------------------------------
rem compile.bat — Standalone Executable Compiler for Windows (low-spec PCs)
rem -----------------------------------------------------------------------------
setlocal enabledelayedexpansion

echo.
echo =======================================================================
echo          AI Matcher Engine - Standalone Windows PE Compiler
echo =======================================================================
echo.

rem 1. Check for Catalog CSV
set "CSV_FILE=Item_export_2026-05-29_17-33-30.csv"
if not exist "%CSV_FILE%" (
    echo [Step 1/4] Catalog database not found in current folder. Checking Downloads...
    if exist "%USERPROFILE%\Downloads\item_export_2026-05-30_09-08-22.csv" (
        set "SOURCE_CSV=%USERPROFILE%\Downloads\item_export_2026-05-30_09-08-22.csv"
    ) else if exist "%USERPROFILE%\Downloads\Item_export_2026-05-29_17-33-30.csv" (
        set "SOURCE_CSV=%USERPROFILE%\Downloads\Item_export_2026-05-29_17-33-30.csv"
    )

    if defined SOURCE_CSV (
        echo Copying catalog database: !SOURCE_CSV! -^> %CSV_FILE%
        copy /Y "!SOURCE_CSV!" "%CSV_FILE%"
    ) else (
        echo.
        echo [ERROR] Unified catalog CSV database not found.
        echo Please ensure 'Item_export_2026-05-29_17-33-30.csv' is placed in this folder
        echo or in your Windows Downloads folder.
        echo.
        pause
        exit /b 1
    )
) else (
    echo [Step 1/4] Catalog database found in current folder.
)

rem 2. Check for Python installation
echo [Step 2/4] Verifying Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python is not installed or not added to your system PATH.
    echo Please install Python 3.9 - 3.12 on your Windows PC first.
    echo Make sure to check the box "Add Python.exe to PATH" during installation.
    echo.
    echo You can download it from: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

rem 3. Create Windows Virtual Environment
echo [Step 3/4] Creating virtual environment inside SSD...
if not exist "qwen3_engine\venv_win" (
    python -m venv qwen3_engine\venv_win
)

echo Activating virtual environment and installing Flask, RapidFuzz, Flask-CORS, PyInstaller...
call qwen3_engine\venv_win\Scripts\activate.bat
python -m pip install --upgrade pip
pip install Flask flask-cors rapidfuzz pyinstaller

rem 4. Run PyInstaller Compilation
echo [Step 4/4] Compiling standalone executable...
pyinstaller --onefile --clean --name fast_search_server --add-data "Item_export_2026-05-29_17-33-30.csv;." qwen3_engine\fast_server.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Compilation failed.
    pause
    exit /b 1
)

echo Cleaning up build assets...
move /Y dist\fast_search_server.exe .\
rmdir /S /Q build dist
del /F /Q fast_search_server.spec

echo.
echo =======================================================================
echo   Stand-alone Windows executable built successfully!
echo   Filename: fast_search_server.exe
echo =======================================================================
echo.
echo You can now run 'fast_search_server.exe' on any Windows PC without Python.
echo When started, it launches the matching server on http://localhost:8080.
echo.
pause
