@echo off
chcp 65001 >nul
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python scripts/recommend.py --user-id 1 --n 10
pause
