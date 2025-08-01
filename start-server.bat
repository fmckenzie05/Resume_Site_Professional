@echo off
echo Starting local web server...
echo.

cd /d "%~dp0"

:: Try python3 first (common on many systems)
where python3 >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Using python3...
    echo Your website will be available at: http://localhost:8000
    echo Press Ctrl+C to stop the server
    echo.
    python3 -m http.server 8000 --bind 127.0.0.1
    goto end
)

:: Try python if python3 not found
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Using python...
    echo Your website will be available at: http://localhost:8000
    echo Press Ctrl+C to stop the server
    echo.
    python -m http.server 8000 --bind 127.0.0.1
    goto end
)

:: Try py launcher (Windows Python Launcher)
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Using py launcher...
    echo Your website will be available at: http://localhost:8000
    echo Press Ctrl+C to stop the server
    echo.
    py -3 -m http.server 8000 --bind 127.0.0.1
    goto end
)

:: No Python found
echo ERROR: Python is not installed or not in PATH!
echo.
echo Please install Python 3 from https://www.python.org/downloads/
echo Make sure to check "Add Python to PATH" during installation.
echo.

:end
pause