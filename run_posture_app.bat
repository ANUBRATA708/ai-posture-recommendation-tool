@echo off
REM Launch Streamlit Posture App
echo Launching AI Posture Advisor in your browser...
cd /d "%~dp0"
streamlit run app.py
pause
