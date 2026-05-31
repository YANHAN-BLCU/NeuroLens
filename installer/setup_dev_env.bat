@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "APP_ROOT=%~dp0.."
set "REQ=%APP_ROOT%\install_support\requirements-dev.txt"
set "VENV=%APP_ROOT%\dev_env"

where python >nul 2>&1
if errorlevel 1 (
  echo [NeuroLens] 未在 PATH 中找到 python，请安装 Python 3.10+ 并勾选 "Add to PATH"。
  exit /b 1
)

echo [NeuroLens] 创建虚拟环境: "%VENV%"
python -m venv "%VENV%"
if errorlevel 1 exit /b 1

call "%VENV%\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r "%REQ%"
if errorlevel 1 exit /b 1

echo [NeuroLens] 开发环境就绪。
exit /b 0
