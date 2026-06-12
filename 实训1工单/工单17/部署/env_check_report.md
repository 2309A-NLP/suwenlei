# RAG 压测环境检查报告

**检查时间**: 2026-06-09 21:38
**状态**: ✅ 全部就绪

---

## 工具检查结果

| 工具 | 状态 | 详情 |
|------|------|------|
| JMeter | ✅ 可用 | v5.6.3，路径: `/mnt/e/jmeter/apache-jmeter-5.6.3/bin/jmeter` |
| Python requests | ✅ 已安装 | via pip |
| py-spy | ✅ 已安装 | v0.4.2，用于CPU性能分析 |
| memory-profiler | ✅ 已安装 | v0.61.0，用于内存分析 |
| nvidia-smi | ✅ 可用 | RTX 3060 Laptop, 6GB显存, 当前占用2001MiB, GPU利用率32% |
| docker stats | ✅ 可用 | 可监控所有容器资源 |
| redis-cli | ✅ 可用 | 通过 `docker exec redis redis-cli -a ragflow12345678` |

## Docker 容器状态

| 容器 | CPU% | 内存使用 | 内存% |
|------|------|----------|-------|
| ragflow | 0.26% | 4.08GiB / 7.76GiB | 52.57% |
| redis | 0.13% | 6.1MiB / 7.76GiB | 0.08% |
| minio | 0.00% | 128.5MiB / 7.76GiB | 1.62% |
| es01 | 0.27% | 746.2MiB / 7.76GiB | 9.39% |
| mysql | 0.76% | 189.5MiB / 7.76GiB | 2.39% |

## 关键信息

- **Redis密码**: `ragflow12345678`（从docker inspect获取）
- **API Token**: `ragflow-EyYTgyMjBkOWUxNDExZjBiNzkxMGEyNm`
- **GPU**: NVIDIA RTX 3060 Laptop, CUDA 13.2, 驱动 595.79
- **系统总内存**: 7.76 GiB（ragflow已占52.57%，压测时需关注）

## 输出文件

- `env_config.sh` - 环境变量配置文件，后续脚本可 `source` 使用
- `env_check_report.md` - 本报告
