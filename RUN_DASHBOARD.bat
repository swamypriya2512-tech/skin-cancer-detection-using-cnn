@echo off
echo Starting DermAI Dashboard...
call .\venv\Scripts\activate.bat
python -m streamlit run app.py
pause
