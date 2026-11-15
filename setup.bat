@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -3.11 -m venv .venv
if errorlevel 1 goto error
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto error
pip install -r requirements.txt
if errorlevel 1 goto error
echo.
echo Установка завершена. Запустите run_demo.bat и run_tests.bat
pause
exit /b 0
:error
echo Ошибка установки. Проверьте Python 3.11 и доступ в интернет.
pause
exit /b 1
