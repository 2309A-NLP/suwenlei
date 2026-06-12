# RAGFlow 部署文档 (DEPLOYMENT.md)

## 环境要求

### 硬件配置
- CPU: 8 核+
- 内存：16GB+
- 磁盘：100GB+ SSD
- GPU: 可选 (用于本地模型)

### 软件依赖
- Docker 20.10+
- Docker Compose v2.0+
- WSL2 (Windows 用户)

## 快速开始

### 1. 克隆代码
```bash
git clone https://github.com/infiniflow/ragflow.git
cd ragflow/docker
```

### 2. 配置环境变量
```bash
cp .env.example .env
# 编辑.env 文件，设置密码等配置
```

### 3. 启动服务
```bash
docker compose up -d
```

### 4. 验证部署
```bash
# 检查容器状态
docker ps

# 查看日志
docker logs -f ragflow
```

### 5. 访问 Web UI
- 地址：http://localhost
- 默认账号：admin@test.com
- 默认密码：12345678

## 配置说明

### 数据库配置
```yaml
mysql:
  host: mysql
  port: 3306
  user: root
  password: ragflow12345678
  database: rag_flow
```

### Redis 配置
```yaml
redis:
  host: redis
  port: 6379
  password: ragflow12345678
```

### Elasticsearch 配置
```yaml
es:
  hosts: http://es01:9200
  username: elastic
  password: ragflow12345678
```

## 常见问题

### 端口冲突
如果 80 端口被占用，修改 docker-compose.yml:
```yaml
ports:
  - "8080:80"  # 改为 8080
```

### 内存不足
调整容器内存限制:
```yaml
services:
  ragflow:
    deploy:
      resources:
        limits:
          memory: 4G
```

### 网络问题
WSL2 用户需使用网关 IP 访问 Windows 服务:
```bash
# 获取网关 IP
ip route | grep default
```

## 性能优化

### 1. 模型缓存
已启用模型实例缓存，减少重复初始化开销。

### 2. 连接池配置
确保数据库连接池大小合理:
```yaml
mysql:
  max_connections: 900
  stale_timeout: 300
```

### 3. 外部 API 优化
对于 DeepSeek 等外部 API，建议配置 HTTP 连接池。

## 监控

### 容器监控
```bash
docker stats
```

### 日志查看
```bash
docker logs -f ragflow --tail 100
```

### 健康检查
```bash
curl http://localhost/api/v1/system/status
```

## 备份

### 数据库备份
```bash
docker exec mysql mysqldump -u root -p rag_flow > backup.sql
```

### 文档备份
```bash
docker cp minio:/data /backup/path
```

## 升级

### 版本升级
```bash
# 停止服务
docker compose down

# 拉取新镜像
docker pull infiniflow/ragflow:latest

# 启动新版本
docker compose up -d
```

---

**最后更新**: 2026-06-10  
**版本**: v0.25.0
