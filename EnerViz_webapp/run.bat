@echo off
title Installing Requirements
pip install -r requirements.txt
python -m nltk.downloader stopwords
echo Launching Visualization...
start "Visualization" cmd /c 
start cmd /c "title Enerviz && py app.py"
timeout /t 15
start http://127.0.0.1:8050 