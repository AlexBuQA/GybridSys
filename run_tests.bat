@echo off
chcp 65001 >nul
cd /d "%~dp0"
call .venv\Scripts\activate.bat
pytest -q
if errorlevel 1 goto error
python scripts/validate_artifacts.py
if errorlevel 1 goto error
pause
exit /b 0
:error
echo Проверки завершились с ошибкой.
pause
exit /b 1
