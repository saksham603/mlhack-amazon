@echo off
title AMLC PIPELINE (day-2 chain) - DO NOT CLOSE THIS WINDOW
echo This window runs the AMLC 2026 pipeline (day-2 chain). Closing it stops the pipeline.
echo Progress: C:\Users\suremdra singh\amlc2026\data\_v2\logs\chain_driver.log
cd /d "C:\Users\suremdra singh\amlc2026"
set PYTHONPATH=src
set AMLC_KA=10
set PYTHONIOENCODING=utf-8
"C:\Users\suremdra singh\amlc2026\.venv\Scripts\python.exe" -u -m amlc.pipeline.chain_day2 >> "C:\Users\suremdra singh\amlc2026\data\_v2\logs\chain_driver.log" 2>&1
