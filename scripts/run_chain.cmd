@echo off
cd /d "C:\Users\suremdra singhmlc2026"
set PYTHONPATH=src
set AMLC_KA=10
set PYTHONIOENCODING=utf-8
"C:\Users\suremdra singhmlc2026\.venv\Scripts\python.exe" -u -m amlc.pipeline.chain_day2 >> "C:\Users\suremdra singhmlc2026\data\_v2\logs\chain_driver.log" 2>&1
