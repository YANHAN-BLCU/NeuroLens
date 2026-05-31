@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

rem 在安装目录下创建默认数据目录结构（可选；实际数据路径仍可由 NEUROLENS_OUTPUTS_DIR 等配置）
set "DATA_ROOT=%CD%\Data\outputs"
mkdir "%DATA_ROOT%" 2>nul
mkdir "%DATA_ROOT%\layer_evolution" 2>nul
mkdir "%DATA_ROOT%\gradient_dependency" 2>nul
mkdir "%DATA_ROOT%\quadrant_classification" 2>nul
mkdir "%DATA_ROOT%\assessment" 2>nul
mkdir "%DATA_ROOT%\tsft_finetuning" 2>nul

exit /b 0
