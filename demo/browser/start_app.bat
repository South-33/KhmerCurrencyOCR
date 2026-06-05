@echo off
title CashSnap Browser App Server
:: Change directory to the repository root so that configs/ and runs/ paths resolve correctly
cd /d "%~dp0..\.."
echo ==========================================================
echo               CASHSNAP BROWSER APP SERVER
echo ==========================================================
echo.
echo Starting local web server at: %CD%
echo.
echo Launching default browser at:
echo      http://127.0.0.1:8000/demo/browser/index.html
echo.
start "" "http://127.0.0.1:8000/demo/browser/index.html"
echo.
echo Press Ctrl+C in this terminal to stop the server.
echo.
echo ==========================================================
echo.
python -m http.server 8000 --bind 127.0.0.1
