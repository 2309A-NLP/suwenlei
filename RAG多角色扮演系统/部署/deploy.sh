#!/bin/bash
# ============================================================
#  RAG多角色扮演系统 — 一键部署脚本
#  适用环境：Ubuntu/Debian (WSL2, Linux服务器)
#  生成时间：2026-05-29
# ============================================================

set -e

# ==================== 颜色输出 ====================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC} $1"; }
ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail()  { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

# ==================== 项目配置 ====================
# 根据实际情况修改以下变量
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"
MODEL_DIR="${PROJECT_DIR}/model"
DATASET_DIR="${PROJECT_DIR}/dataset"
LOG_DIR="${PROJECT_DIR}/logs"
PID_DIR="${PROJECT_DIR}/pids"

# 服务端口
API_PORT="${API_PORT:-8001}"
API_HOST="${API_HOST:-0.0.0.0}"

# MySQL
MYSQL_HOST="${MYSQL_HOST:-localhost}"
MYSQL_PORT="${MYSQL_PORT:-3306}"
MYSQL_USER="${MYSQL_USER:-root}"
MYSQL_PASSWORD="${MYSQL_PASSWORD:-123456}"
MYSQL_DB="${MYSQL_DB:-2309a}"

# Redis
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"

# Milvus
MILVUS_HOST="${MILVUS_HOST:-localhost}"
MILVUS_PORT="${MILVUS_PORT:-19530}"

# DeepSeek API (必须配置)
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"
DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-flash}"

# ==================== 工具函数 ====================
mkdir -p "$LOG_DIR" "$PID_DIR"

check_port() {
    local port=$1
    local name=$2
    if ss -tlnp 2>/dev/null | grep -q ":${port} " || netstat -tlnp 2>/dev/null | grep -q ":${port} "; then
        ok "$name 已在端口 $port 运行"
        return 0
    else
        return 1
    fi
}

wait_for_port() {
    local port=$1
    local name=$2
    local max_wait=${3:-30}
    local count=0
    while ! check_port "$port" "$name" 2>/dev/null; do
        sleep 1
        count=$((count + 1))
        if [ $count -ge $max_wait ]; then
            fail "$name 启动超时（${max_wait}s），请手动检查"
        fi
    done
    ok "$name 就绪"
}

# ==================== Step 1: 系统依赖 ====================
echo ""
echo "=========================================="
echo "  RAG多角色扮演系统 部署脚本"
echo "=========================================="
echo ""
info "Step 1/8: 检查系统依赖..."

# Python3
if ! command -v python3 &> /dev/null; then
    warn "未找到 python3，正在安装..."
    sudo apt-get update -qq && sudo apt-get install -y -qq python3 python3-venv python3-pip
fi
PYTHON_VERSION=$(python3 --version 2>&1)
ok "Python: $PYTHON_VERSION"

# pip (系统级，用于 venv 内部)
if ! python3 -c "import ensurepip" 2>/dev/null; then
    warn "python3-ensurepip 缺失，正在安装..."
    sudo apt-get install -y -qq python3-ensurepip
fi

# ==================== Step 2: 基础服务 ====================
echo ""
info "Step 2/8: 检查基础服务..."

# --- MySQL ---
if check_port 3306 "MySQL"; then
    info "MySQL 已运行，跳过安装"
else
    warn "MySQL 未运行，尝试启动..."
    if command -v mysqld &> /dev/null; then
        sudo systemctl start mysql 2>/dev/null || sudo service mysql start 2>/dev/null || sudo mysqld_safe &
        wait_for_port 3306 "MySQL" 15
    else
        warn "MySQL 未安装，请先安装：sudo apt-get install mysql-server"
        warn "安装后运行：sudo systemctl start mysql"
    fi
fi

# 初始化数据库和用户表
info "初始化 MySQL 数据库..."
python3 -c "
import mysql.connector
from mysql.connector import Error
try:
    conn = mysql.connector.connect(
        host='${MYSQL_HOST}', port=${MYSQL_PORT},
        user='${MYSQL_USER}', password='${MYSQL_PASSWORD}'
    )
    cursor = conn.cursor()
    cursor.execute('CREATE DATABASE IF NOT EXISTS ${MYSQL_DB} DEFAULT CHARSET utf8mb4')
    cursor.execute('USE ${MYSQL_DB}')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) NOT NULL UNIQUE,
            password VARCHAR(128) NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    ''')
    conn.commit()
    cursor.close()
    conn.close()
    print('  MySQL 数据库 ${MYSQL_DB} 和 users 表初始化完成')
except Error as e:
    print(f'  MySQL 初始化警告: {e}')
" 2>&1 || warn "MySQL 初始化跳过（可能需要手动配置密码）"

# --- Redis ---
if check_port 6379 "Redis"; then
    info "Redis 已运行，跳过安装"
else
    warn "Redis 未运行，尝试启动..."
    if command -v redis-server &> /dev/null; then
        redis-server --daemonize yes --port 6379 2>/dev/null || sudo systemctl start redis 2>/dev/null
        wait_for_port 6379 "Redis" 10
    else
        warn "Redis 未安装，请先安装：sudo apt-get install redis-server"
    fi
fi

# --- Milvus ---
if check_port 19530 "Milvus"; then
    info "Milvus 已运行，跳过安装"
else
    warn "Milvus 未运行"
    if command -v milvus &> /dev/null; then
        info "尝试启动 Milvus standalone..."
        milvus server -p 19530 &>/dev/null &
        wait_for_port 19530 "Milvus" 30
    else
        warn "Milvus 未安装"
        echo ""
        echo "  安装 Milvus Standalone（推荐方式）:"
        echo "  wget https://github.com/milvus-io/milvus/releases/download/v2.4.5/milvus_2.4.5_cuda12.1.0_amd64.tar.gz"
        echo "  tar xzf milvus_2.4.5_cuda12.1.0_amd64.tar.gz"
        echo "  cd milvus && ./start_standalone.sh"
        echo ""
        echo "  或使用 Docker:"
        echo "  docker run -d --name milvus -p 19530:19530 -p 9091:9091 \\"
        echo "    -v /var/lib/milvus:/var/lib/milvus milvusdb/milvus:v2.4.5 milvus run standalone"
        echo ""
    fi
fi

# ==================== Step 3: DeepSeek API Key ====================
echo ""
info "Step 3/8: 检查 DeepSeek API Key..."

if [ -z "$DEEPSEEK_API_KEY" ]; then
    warn "DEEPSEEK_API_KEY 未设置！"
    echo ""
    echo "  请设置环境变量后重新运行，或在 .env 文件中配置："
    echo "  export DEEPSEEK_API_KEY='sk-your-api-key-here'"
    echo ""
    echo "  获取方式：访问 https://platform.deepseek.com/api_keys"
    echo ""
    # 交互式输入
    read -p "  现在输入 DeepSeek API Key（留空跳过，稍后手动配置）: " input_key
    if [ -n "$input_key" ]; then
        DEEPSEEK_API_KEY="$input_key"
        ok "已设置 DeepSeek API Key"
    else
        warn "跳过 API Key 配置，请确保后续手动设置"
    fi
fi

# ==================== Step 4: Python 虚拟环境 ====================
echo ""
info "Step 4/8: 创建 Python 虚拟环境..."

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    ok "虚拟环境创建完成: $VENV_DIR"
else
    ok "虚拟环境已存在: $VENV_DIR"
fi

# 激活虚拟环境
source "${VENV_DIR}/bin/activate"

# 升级 pip
pip install --upgrade pip -q 2>&1 | tail -1

# ==================== Step 5: 安装 Python 依赖 ====================
echo ""
info "Step 5/8: 安装 Python 依赖..."

# 核心依赖
pip install -q \
    fastapi==0.115.0 \
    uvicorn[standard]==0.30.0 \
    python-multipart==0.0.9 \
    jinja2==3.1.4 \
    pydantic==2.9.0 \
    mysql-connector-python==9.0.0 \
    redis==5.0.8 \
    pymilvus==2.4.7 \
    openai==1.45.0 \
    sentence-transformers==3.1.1 \
    rank-bm25==0.2.2 \
    jieba==0.42.1 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    2>&1 | tail -3

# PyTorch (CUDA 版本，如果 GPU 可用)
info "检查 PyTorch..."
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    ok "PyTorch CUDA 已就绪"
else
    warn "安装 PyTorch (CUDA 12.1)..."
    pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3 || {
        warn "CUDA 版安装失败，回退到 CPU 版本..."
        pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -3
    }
fi

ok "Python 依赖安装完成"

# ==================== Step 6: 模型文件检查 ====================
echo ""
info "Step 6/8: 检查模型文件..."

# bge-m3 嵌入模型
if [ -d "${MODEL_DIR}/bge-m3" ] && [ "$(ls -A ${MODEL_DIR}/bge-m3 2>/dev/null)" ]; then
    ok "bge-m3 嵌入模型已就绪"
else
    warn "bge-m3 嵌入模型缺失"
    echo "  下载地址: https://huggingface.co/BAAI/bge-m3"
    echo "  保存到: ${MODEL_DIR}/bge-m3/"
    echo ""
    echo "  如果 HuggingFace 无法访问，使用国内镜像："
    echo "  export HF_ENDPOINT=https://hf-mirror.com"
    echo "  huggingface-cli download BAAI/bge-m3 --local-dir ${MODEL_DIR}/bge-m3"
    echo ""

    read -p "  是否现在下载 bge-m3？(y/N): " download_bge
    if [ "$download_bge" = "y" ] || [ "$download_bge" = "Y" ]; then
        pip install -q huggingface_hub 2>/dev/null
        mkdir -p "${MODEL_DIR}/bge-m3"
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('BAAI/bge-m3', local_dir='${MODEL_DIR}/bge-m3')
print('下载完成')
" 2>&1 || warn "下载失败，请手动下载"
    fi
fi

# bge-reranker-v2-m3 重排序模型
if [ -d "${MODEL_DIR}/bge-reranker-v2-m3" ] && [ "$(ls -A ${MODEL_DIR}/bge-reranker-v2-m3 2>/dev/null)" ]; then
    ok "bge-reranker-v2-m3 重排序模型已就绪"
else
    warn "bge-reranker-v2-m3 重排序模型缺失"
    echo "  下载地址: https://huggingface.co/BAAI/bge-reranker-v2-m3"

    read -p "  是否现在下载？(y/N): " download_rerank
    if [ "$download_rerank" = "y" ] || [ "$download_rerank" = "Y" ]; then
        mkdir -p "${MODEL_DIR}/bge-reranker-v2-m3"
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('BAAI/bge-reranker-v2-m3', local_dir='${MODEL_DIR}/bge-reranker-v2-m3')
print('下载完成')
" 2>&1 || warn "下载失败，请手动下载"
    fi
fi

# ==================== Step 7: 构建知识库 ====================
echo ""
info "Step 7/8: 知识库构建..."

# 检查 CSV 数据文件
if [ -f "${DATASET_DIR}/guideline_chunks_advanced.csv" ] || [ -f "${PROJECT_DIR}/offline part/dataset/guideline_chunks_advanced.csv" ]; then
    ok "高血压指南数据集存在"
else
    warn "高血压指南数据集缺失: guideline_chunks_advanced.csv"
fi

if [ -f "${DATASET_DIR}/civil_code_chunks.csv" ] || [ -f "${PROJECT_DIR}/offline part/dataset/civil_code_chunks.csv" ]; then
    ok "民法典数据集存在"
else
    warn "民法典数据集缺失: civil_code_chunks.csv"
fi

# 检查 Milvus 是否可用，可用则构建知识库
if check_port 19530 "Milvus" 2>/dev/null; then
    echo ""
    info "Milvus 可用，检查是否需要构建知识库..."

    python3 -c "
from pymilvus import MilvusClient
client = MilvusClient(uri='http://${MILVUS_HOST}:${MILVUS_PORT}')
# 检查高血压指南知识库
has_guideline = client.has_collection('hypertension_guideline_kb')
# 检查民法典知识库
has_civil = client.has_collection('civil_code_kb')
client.close()
if has_guideline:
    print('  高血压指南知识库已存在')
else:
    print('  高血压指南知识库需要构建')
if has_civil:
    print('  民法典知识库已存在')
else:
    print('  民法典知识库需要构建')
" 2>&1

    read -p "  是否构建/重建知识库？(y/N): " build_kb
    if [ "$build_kb" = "y" ] || [ "$build_kb" = "Y" ]; then
        # 构建高血压指南知识库
        OFFLINE_DIR="${PROJECT_DIR}/offline part"
        if [ -f "${OFFLINE_DIR}/knowledge_base_construction.py" ]; then
            info "构建高血压指南知识库..."
            cd "${OFFLINE_DIR}"
            python3 knowledge_base_construction.py 2>&1 | tee "${LOG_DIR}/build_guideline.log"
            ok "高血压指南知识库构建完成"
        fi

        # 构建民法典知识库
        if [ -f "${OFFLINE_DIR}/civil_code_vectorization.py" ]; then
            info "构建民法典知识库..."
            cd "${OFFLINE_DIR}"
            python3 civil_code_vectorization.py 2>&1 | tee "${LOG_DIR}/build_civil_code.log"
            ok "民法典知识库构建完成"
        fi

        cd "${PROJECT_DIR}"
    fi
else
    warn "Milvus 未运行，跳过知识库构建"
    warn "Milvus 启动后请运行以下命令构建知识库："
    echo "  cd '${PROJECT_DIR}/offline part'"
    echo "  python3 knowledge_base_construction.py"
    echo "  python3 civil_code_vectorization.py"
fi

# ==================== Step 8: 启动服务 ====================
echo ""
info "Step 8/8: 启动 API 服务..."

# 生成 .env 文件（供手动管理时使用）
cat > "${PROJECT_DIR}/.env" << EOF
# === MySQL ===
MYSQL_HOST=${MYSQL_HOST}
MYSQL_PORT=${MYSQL_PORT}
MYSQL_USER=${MYSQL_USER}
MYSQL_PASSWORD=${MYSQL_PASSWORD}
MYSQL_DB=${MYSQL_DB}

# === Redis ===
REDIS_HOST=${REDIS_HOST}
REDIS_PORT=${REDIS_PORT}

# === Milvus ===
MILVUS_HOST=${MILVUS_HOST}
MILVUS_PORT=${MILVUS_PORT}

# === DeepSeek API ===
DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
DEEPSEEK_BASE_URL=${DEEPSEEK_BASE_URL}
DEEPSEEK_MODEL=${DEEPSEEK_MODEL}

# === API 服务 ===
API_HOST=${API_HOST}
API_PORT=${API_PORT}
EOF
ok ".env 文件已生成"

# 停止旧进程（如果存在）
if [ -f "${PID_DIR}/api.pid" ]; then
    OLD_PID=$(cat "${PID_DIR}/api.pid")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        warn "停止旧的 API 进程 (PID: $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null
        sleep 2
    fi
    rm -f "${PID_DIR}/api.pid"
fi

# 设置环境变量
export MYSQL_HOST MYSQL_PORT MYSQL_USER MYSQL_PASSWORD MYSQL_DB
export REDIS_HOST REDIS_PORT
export MILVUS_HOST MILVUS_PORT
export DEEPSEEK_API_KEY DEEPSEEK_BASE_URL DEEPSEEK_MODEL
export API_HOST API_PORT

# 启动 FastAPI
cd "${PROJECT_DIR}/api_service"
nohup python3 -m uvicorn main:app \
    --host "${API_HOST}" \
    --port "${API_PORT}" \
    --workers 1 \
    > "${LOG_DIR}/api_service.log" 2>&1 &

echo $! > "${PID_DIR}/api.pid"
API_PID=$!

info "等待服务启动..."
sleep 3

# 验证服务是否启动
if kill -0 "$API_PID" 2>/dev/null; then
    # 尝试健康检查
    sleep 2
    if curl -sf "http://127.0.0.1:${API_PORT}/api/health" > /dev/null 2>&1; then
        HEALTH=$(curl -s "http://127.0.0.1:${API_PORT}/api/health" 2>/dev/null)
        ok "API 服务启动成功！"
        echo ""
        echo "  健康状态: $HEALTH"
    else
        ok "API 进程已启动 (PID: $API_PID)"
    fi
else
    fail "API 服务启动失败，请查看日志: ${LOG_DIR}/api_service.log"
fi

# ==================== 完成 ====================
echo ""
echo "=========================================="
echo -e "  ${GREEN}部署完成！${NC}"
echo "=========================================="
echo ""
echo "  项目目录:   ${PROJECT_DIR}"
echo "  API 地址:   http://${API_HOST}:${API_PORT}"
echo "  前端页面:   http://${API_HOST}:${API_PORT}/"
echo ""
echo "  日志目录:   ${LOG_DIR}/"
echo "  PID 文件:   ${PID_DIR}/api.pid"
echo "  环境配置:   ${PROJECT_DIR}/.env"
echo ""
echo "  === 管理命令 ==="
echo "  启动服务:   bash deploy.sh start"
echo "  停止服务:   bash deploy.sh stop"
echo "  查看日志:   bash deploy.sh logs"
echo "  重启服务:   bash deploy.sh restart"
echo "  健康检查:   bash deploy.sh health"
echo ""
echo "  === API 端点 ==="
echo "  首页:       GET  /"
echo "  注册:       POST /api/register  {username, password}"
echo "  登录:       POST /api/login     {username, password}"
echo "  聊天:       POST /api/chat      {user_id, user_question, role}"
echo "  健康检查:   GET  /api/health"
echo ""
echo "  === 支持角色 ==="
echo "  高血压:     hypertension (基层全科医生)"
echo "  中医:       tcm (中医师)"
echo "  患者教育:   patient_edu (健康教育护士)"
echo "  法律顾问:   lawyer (法律顾问)"
echo ""


# ==================== 管理命令（子命令模式） ====================
# 用法: bash deploy.sh [start|stop|restart|logs|health|status]
cmd_start() {
    # 加载 .env
    if [ -f "${PROJECT_DIR}/.env" ]; then
        set -a; source "${PROJECT_DIR}/.env"; set +a
    fi
    source "${VENV_DIR}/bin/activate"

    if [ -f "${PID_DIR}/api.pid" ] && kill -0 "$(cat ${PID_DIR}/api.pid)" 2>/dev/null; then
        warn "API 服务已在运行 (PID: $(cat ${PID_DIR}/api.pid))"
        return 0
    fi

    cd "${PROJECT_DIR}/api_service"
    nohup python3 -m uvicorn main:app \
        --host "${API_HOST:-0.0.0.0}" \
        --port "${API_PORT:-8001}" \
        --workers 1 \
        > "${LOG_DIR}/api_service.log" 2>&1 &
    echo $! > "${PID_DIR}/api.pid"
    sleep 2
    if kill -0 "$(cat ${PID_DIR}/api.pid)" 2>/dev/null; then
        ok "API 服务已启动 (PID: $(cat ${PID_DIR}/api.pid))"
    else
        fail "启动失败，查看日志: ${LOG_DIR}/api_service.log"
    fi
}

cmd_stop() {
    if [ -f "${PID_DIR}/api.pid" ]; then
        PID=$(cat "${PID_DIR}/api.pid")
        if kill -0 "$PID" 2>/dev/null; then
            info "停止 API 服务 (PID: $PID)..."
            kill "$PID"
            sleep 2
            if kill -0 "$PID" 2>/dev/null; then
                warn "进程未退出，强制终止..."
                kill -9 "$PID"
            fi
            ok "API 服务已停止"
        else
            warn "进程 $PID 已不存在"
        fi
        rm -f "${PID_DIR}/api.pid"
    else
        warn "没有找到运行中的服务"
    fi
}

cmd_restart() {
    cmd_stop
    cmd_start
}

cmd_logs() {
    if [ -f "${LOG_DIR}/api_service.log" ]; then
        echo "=== API 服务日志（最近 50 行）==="
        tail -50 "${LOG_DIR}/api_service.log"
    else
        warn "日志文件不存在: ${LOG_DIR}/api_service.log"
    fi
}

cmd_health() {
    local port="${API_PORT:-8001}"
    echo "=== 服务健康检查 ==="
    echo ""

    # API 服务
    if curl -sf "http://127.0.0.1:${port}/api/health" > /dev/null 2>&1; then
        echo "  [✓] API 服务:     运行中 (端口 ${port})"
        curl -s "http://127.0.0.1:${port}/api/health" | python3 -m json.tool 2>/dev/null || true
    else
        echo "  [✗] API 服务:     未响应"
    fi
    echo ""

    # MySQL
    if check_port 3306 "MySQL" 2>/dev/null; then
        echo "  [✓] MySQL:        运行中 (端口 3306)"
    else
        echo "  [✗] MySQL:        未运行"
    fi

    # Redis
    if check_port 6379 "Redis" 2>/dev/null; then
        echo "  [✓] Redis:        运行中 (端口 6379)"
    else
        echo "  [✗] Redis:        未运行"
    fi

    # Milvus
    if check_port 19530 "Milvus" 2>/dev/null; then
        echo "  [✓] Milvus:       运行中 (端口 19530)"
    else
        echo "  [✗] Milvus:       未运行"
    fi

    echo ""
}

cmd_status() {
    echo "=== 进程状态 ==="
    if [ -f "${PID_DIR}/api.pid" ]; then
        PID=$(cat "${PID_DIR}/api.pid")
        if kill -0 "$PID" 2>/dev/null; then
            echo "  API 服务:  PID $PID (运行中)"
        else
            echo "  API 服务:  PID $PID (已停止)"
        fi
    else
        echo "  API 服务:  未启动"
    fi
}

# 子命令入口
case "${1:-}" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    restart) cmd_restart ;;
    logs)    cmd_logs ;;
    health)  cmd_health ;;
    status)  cmd_status ;;
    "")      ;; # 首次部署（上面的步骤已经执行）
    *)
        echo "用法: bash deploy.sh [start|stop|restart|logs|health|status]"
        echo ""
        echo "  start   - 启动 API 服务"
        echo "  stop    - 停止 API 服务"
        echo "  restart - 重启 API 服务"
        echo "  logs    - 查看最近日志"
        echo "  health  - 服务健康检查"
        echo "  status  - 查看进程状态"
        echo ""
        echo "  不带参数运行 = 首次部署"
        ;;
esac
