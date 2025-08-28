@echo off
REM Crypto Trading Bot Startup Script (Windows)
REM Version: 1.0.0

REM Set default values
set MODE=paper
set CONFIG_FILE=config.json
set LOG_DIR=logs
set PYTHON_EXEC=python
set VENV_DIR=venv

REM Parse command line arguments
:parse_args
if "%1"=="" goto end_args
if "%1"=="-m" (set MODE=%2) & shift & shift & goto parse_args
if "%1"=="-c" (set CONFIG_FILE=%2) & shift & shift & goto parse_args
if "%1"=="-l" (set LOG_DIR=%2) & shift & shift & goto parse_args
if "%1"=="-p" (set PYTHON_EXEC=%2) & shift & shift & goto parse_args
if "%1"=="-v" (set VENV_DIR=%2) & shift & shift & goto parse_args
shift
goto parse_args
:end_args

REM Check if virtual environment exists and activate it
if exist "%VENV_DIR%" (
  echo Activating virtual environment...
  call "%VENV_DIR%\Scripts\activate"
) else (
  echo Virtual environment not found at %VENV_DIR%
  echo Creating virtual environment...
  %PYTHON_EXEC% -m venv "%VENV_DIR%"
  call "%VENV_DIR%\Scripts\activate"
  
  REM Install requirements
  echo Installing requirements...
  pip install --upgrade pip
  pip install -r requirements.txt
)

REM Create log directory if it doesn't exist
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Set environment variables
set BOT_MODE=%MODE%
set CONFIG_PATH=%CONFIG_FILE%
set LOG_PATH=%LOG_DIR%\crypto_bot.log

REM Check Python version
for /f "usebackq" %%i in (`%PYTHON_EXEC% -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')`) do set PYTHON_VERSION=%%i
if "%PYTHON_VERSION%" LSS "3.9" (
  echo Error: Python 3.9 or higher is required (found %PYTHON_VERSION%)
  exit /b 1
)

REM Run the bot
echo Starting Crypto Trading Bot in %MODE% mode...
echo Using config: %CONFIG_FILE%
echo Logs will be written to: %LOG_PATH%

%PYTHON_EXEC% -u main.py %MODE% >> "%LOG_PATH%" 2>&1

REM Deactivate virtual environment when done
deactivate
echo Bot stopped.