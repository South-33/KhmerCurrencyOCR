@echo off
title CashSnap Browser Demo Server
echo ==========================================================
echo               CASHSNAP BROWSER DEMO SERVER
echo ==========================================================
echo.
echo Starting local web server...
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
