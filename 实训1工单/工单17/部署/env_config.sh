#!/bin/bash
# RAG压测环境配置 - 自动生成
# 日期: 2026-06-09

# JMeter
export JMETER_HOME="/mnt/e/jmeter/apache-jmeter-5.6.3"
export JMETER_BIN="${JMETER_HOME}/bin/jmeter"

# RAGFlow API
export RAGFLOW_API_BASE="http://localhost:9380"

# Redis
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
export REDIS_CMD="docker exec redis redis-cli -a ragflow12345678"

# Docker容器名
export CONTAINER_RAGFLOW="ragflow"
export CONTAINER_REDIS="redis"
export CONTAINER_ES="es01"
export CONTAINER_MYSQL="mysql"
export CONTAINER_MINIO="minio"

# 压测参数
export TARGET_URL="${RAGFLOW_API_BASE}/api/v1/chats"
export TEST_DURATION=300
export CONCURRENT_USERS_LIST="1 5 10 20 50"
export REDIS_PASSWORD=ragflow12345678
