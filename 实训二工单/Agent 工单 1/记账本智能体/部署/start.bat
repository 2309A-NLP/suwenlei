@echo off
REM 家庭记账本智能体 - 启动脚本（Windows）

echo ==============================================
echo 家庭记账本智能体 - 启动脚本
echo ==============================================

REM 切换到项目目录
cd /d "%~dp0"

REM 检查 Python 环境
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo 错误：未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

echo 使用 Python: 
python --version

REM 检查依赖
python -c "import flask" 2>nul
if %errorlevel% neq 0 (
    echo 正在安装依赖...
    python -m pip install -r requirements.txt
)

REM 启动 Web 服务
echo.
echo 正在启动 Web 服务...
echo 访问地址：http://localhost:5000
echo 按 Ctrl+C 停止服务
echo.

REM 源码位于 研发\ 目录，切换过去再启动（同目录导入 + templates 才能正常工作）
cd /d "%~dp0..\研发"
python web_server.py

pause
