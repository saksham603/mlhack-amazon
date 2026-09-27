@echo off
title AMLC PIPELINE (v3.2 chain) - DO NOT CLOSE THIS WINDOW
echo This window runs the AMLC 2026 pipeline (v3.2 chain; it waits for the day-2 chain first). Closing it stops the pipeline.
echo Progress: C:\Users\suremdra singh\amlc2026\data\_v2\logs\chain_v32_driver.log
cd /d "C:\Users\suremdra singh\amlc2026"
set PYTHONPATH=src
set AMLC_KA=10
set PYTHONIOENCODING=utf-8
"C:\Users\suremdra singh\amlc2026\.venv\Scripts\python.exe" -u -m amlc.pipeline.chain_v32 >> "C:\Users\suremdra singh\amlc2026\data\_v2\logs\chain_v32_driver.log" 2>&1
