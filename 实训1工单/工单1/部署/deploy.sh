#!/bin/bash
# RAG问答系统部署脚本
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  RAG问答系统部署脚本${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查Python版本
echo -e "\n${YELLOW}[1/6] 检查Python环境...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓ Python版本: ${PYTHON_VERSION}${NC}"
else
    echo -e "${RED}✗ 未找到Python3，请先安装Python3.8+${NC}"
    exit 1
fi

# 检查pip
echo -e "\n${YELLOW}[2/6] 检查pip...${NC}"
if command -v pip3 &> /dev/null; then
    echo -e "${GREEN}✓ pip已安装${NC}"
else
    echo -e "${RED}✗ 未找到pip，请先安装pip${NC}"
    exit 1
fi

# 创建虚拟环境（可选）
echo -e "\n${YELLOW}[3/6] 创建虚拟环境...${NC}"
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
    echo -e "${GREEN}✓ 虚拟环境已创建: ${VENV_DIR}${NC}"
else
    echo -e "${GREEN}✓ 虚拟环境已存在: ${VENV_DIR}${NC}"
fi

# 激活虚拟环境
source $VENV_DIR/bin/activate

# 安装依赖
echo -e "\n${YELLOW}[4/6] 安装依赖...${NC}"
pip install --upgrade pip
pip install fastapi uvicorn pydantic requests pymupdf pymilvus python-multipart
echo -e "${GREEN}✓ 依赖安装完成${NC}"

# 检查环境变量
echo -e "\n${YELLOW}[5/6] 检查环境变量...${NC}"
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo -e "${YELLOW}⚠ DEEPSEEK_API_KEY未设置${NC}"
    echo -e "${YELLOW}  请设置环境变量: export DEEPSEEK_API_KEY='your_api_key'${NC}"
    echo -e "${YELLOW}  或在.env文件中配置${NC}"
else
    echo -e "${GREEN}✓ DEEPSEEK_API_KEY已设置${NC}"
fi

# 检查PDF文件
echo -e "\n${YELLOW}[6/6] 检查PDF文件...${NC}"
PDF_FILE="招股说明书1.pdf"
if [ -f "$PDF_FILE" ]; then
    echo -e "${GREEN}✓ PDF文件存在: ${PDF_FILE}${NC}"
else
    echo -e "${YELLOW}⚠ PDF文件不存在: ${PDF_FILE}${NC}"
    echo -e "${YELLOW}  请将PDF文件放置在项目根目录${NC}"
fi

# 启动服务
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  部署完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\n启动命令:"
echo -e "  ${GREEN}python app.py${NC}"
echo -e "\n访问地址: http://localhost:8888"
echo -e "\n${YELLOW}提示:${NC}"
echo -e "  1. 首次启动会加载PDF并建立索引，可能需要1-2分钟"
echo -e "  2. 支持上传多个PDF文件扩展知识库"
echo -e "  3. 建议使用Chrome浏览器以获得最佳体验"
