# RAGFlow 运维文档 (OPERATION.md)

## 日常运维

### 服务状态检查

#### 1. 容器健康状态
```bash
docker ps --filter name=ragflow
```

期望输出：
```
NAMES     STATUS
ragflow   Up 2 hours
```

#### 2. API 健康检查
```bash
curl http://localhost:9380/api/v1/datasets \
  -H "Authorization: Bearer YOUR_API_TOKEN"
```

#### 3. 数据库连接
```bash
docker exec mysql mysql -u root -pragflow12345678 -e "SELECT 1"
```

#### 4. Redis 连接
```bash
docker exec redis redis-cli -a ragflow12345678 ping
```

### 日志管理

#### 查看实时日志
```bash
docker logs -f ragflow --tail 100
```

#### 查看错误日志
```bash
docker logs ragflow 2>&1 | grep -i error | tail -50
```

#### 日志轮转
```bash
# 清理旧日志
docker logs --tail 0 ragflow
```

### 性能监控

#### CPU/内存使用
```bash
docker stats --no-stream
```

#### 磁盘使用
```bash
docker system df
```

#### 网络流量
```bash
watch -n 1 'docker exec ragflow netstat -tuln'
```

### 备份策略

#### 每日备份
```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d)
docker exec mysql mysqldump -u root -pragflow12345678 rag_flow > /backup/ragflow_${DATE}.sql
docker cp minio:/data /backup/minio_${DATE}
```

#### 每周清理
```bash
# 清理 7 天前的备份
find /backup -name "*.sql" -mtime +7 -delete
```

### 故障排查

#### 问题 1: API 响应慢
**症状**: P95 响应时间>10 秒

**排查步骤**:
1. 检查 DeepSeek API 状态
2. 查看容器 CPU 使用率
3. 检查数据库连接数

**解决方案**:
```bash
# 查看慢查询
docker exec mysql mysql -u root -p -e "SHOW PROCESSLIST"

# 重启服务
docker restart ragflow
```

#### 问题 2: 内存泄漏
**症状**: 容器内存持续增长

**排查步骤**:
1. 监控内存趋势
2. 检查是否有未释放的连接

**解决方案**:
```bash
# 定期重启 (临时方案)
docker restart ragflow

# 永久方案：更新代码修复泄漏点
```

#### 问题 3: 文档解析失败
**症状**: 上传文档后状态一直为"解析中"

**排查步骤**:
1. 检查 task executor 日志
2. 验证文档格式

**解决方案**:
```bash
# 查看任务执行器日志
docker logs -f ragflow | grep task_executor

# 重新解析文档
# 在 Web UI 中点击"重新解析"
```

### 性能调优

#### 1. 调整并发数
编辑 docker-compose.yml:
```yaml
services:
  ragflow:
    environment:
      - MAX_CONCURRENT_REQUESTS=20
```

#### 2. 优化数据库
```bash
# 添加索引
docker exec mysql mysql -u root -p rag_flow -e "
  ALTER TABLE document ADD INDEX idx_status (status);
  ALTER TABLE chunk ADD INDEX idx_doc_id (doc_id);
"
```

#### 3. 调整缓存大小
编辑 service_conf.yaml:
```yaml
redis:
  maxmemory: 2gb
  maxmemory-policy: allkeys-lru
```

### 监控告警

#### Prometheus 配置
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'ragflow'
    static_configs:
      - targets: ['ragflow:9380']
```

#### 告警规则
```yaml
# 内存告警
- alert: HighMemoryUsage
  expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
  for: 5m
  
# 响应时间告警
- alert: HighResponseTime
  expr: http_request_duration_seconds{quantile="0.95"} > 5
  for: 5m
```

### 安全加固

#### 1. 修改默认密码
```bash
# 修改数据库密码
docker exec mysql mysql -u root -p -e "ALTER USER 'root'@'localhost' IDENTIFIED BY 'NEW_PASSWORD';"

# 修改 Redis 密码
docker exec redis redis-cli CONFIG SET requirepass NEW_PASSWORD
```

#### 2. 启用 HTTPS
```yaml
# nginx 配置
server {
    listen 443 ssl;
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
}
```

#### 3. 限制访问 IP
```yaml
# docker-compose.yml
services:
  ragflow:
    ports:
      - "127.0.0.1:80:80"  # 只允许本地访问
```

### 升级流程

#### 1. 备份数据
```bash
./backup.sh
```

#### 2. 停止服务
```bash
docker compose down
```

#### 3. 拉取新镜像
```bash
docker pull infiniflow/ragflow:latest
```

#### 4. 启动新版本
```bash
docker compose up -d
```

#### 5. 验证功能
```bash
# API 测试
curl http://localhost/api/v1/datasets

# Web UI 测试
# 浏览器访问 http://localhost
```

---

**最后更新**: 2026-06-10  
**版本**: v0.25.0  
**维护团队**: 八维文化与产业研究院
