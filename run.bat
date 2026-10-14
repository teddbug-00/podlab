@echo off
cd /d "%~dp0"
where cargo >nul 2>nul
if errorlevel 1 (
  echo Rust is not installed yet.
  echo Install it from https://rustup.rs , then close this window and double-click run.bat again.
  pause
  exit /b 1
)
cargo run --release -- %*
echo.
echo Done. The table above is also saved in results.txt - send me that file.
pause
