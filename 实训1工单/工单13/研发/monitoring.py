# -*- coding: utf-8 -*-
"""
监控与追踪模块
- 分布式追踪: OpenTelemetry链路追踪，记录请求在各组件间的流转
- 监控指标: Prometheus指标（请求计数、延迟分布、各阶段耗时）
- 告警: 超阈值自动记录告警日志
"""
import time
import uuid
import json
import logging
import threading
from contextlib import contextmanager
from typing import Dict, Optional
from collections import defaultdict

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import StatusCode

logger = logging.getLogger(__name__)

# ============ Prometheus 指标 ============

# 请求计数
REQUEST_COUNT = Counter(
    'rag_request_total', '总请求数',
    ['endpoint', 'status']
)

# 请求延迟
REQUEST_LATENCY = Histogram(
    'rag_request_duration_seconds', '请求延迟(秒)',
    ['endpoint'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
)

# 各阶段耗时
STAGE_LATENCY = Histogram(
    'rag_stage_duration_seconds', '各阶段耗时(秒)',
    ['stage'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)

# 当前并发请求数
ACTIVE_REQUESTS = Gauge(
    'rag_active_requests', '当前并发请求数'
)

# 告警计数
ALERT_COUNT = Counter(
    'rag_alert_total', '告警总数',
    ['alert_type']
)

# 检索结果数量
RETRIEVAL_RESULTS = Histogram(
    'rag_retrieval_results_count', '检索结果数量',
    buckets=[1, 3, 5, 10, 15, 20, 25]
)

# LLM token数
LLM_TOKENS = Histogram(
    'rag_llm_tokens', 'LLM生成token数',
    buckets=[10, 50, 100, 200, 500, 1000]
)

# ============ 告警阈值配置 ============

ALERT_THRESHOLDS = {
    'request_duration': 5.0,      # 单次请求超过5s告警
    'stage_duration': 3.0,        # 单阶段超过3s告警
    'retrieval_duration': 1.0,    # 检索超过1s告警
    'llm_duration': 5.0,          # LLM超过5s告警
    'error_rate_window': 60,      # 错误率统计窗口(秒)
    'error_rate_threshold': 0.1,  # 错误率超过10%告警
}

# ============ OpenTelemetry 追踪初始化 ============

_resource = Resource.create({"service.name": "rag-system"})
_provider = TracerProvider(resource=_resource)
_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(_provider)
_tracer = trace.get_tracer("rag-tracer")


# ============ 请求追踪上下文 ============

class RequestTrace:
    """单个请求的追踪上下文"""

    def __init__(self, trace_id: str = None, endpoint: str = ""):
        self.trace_id = trace_id or uuid.uuid4().hex[:16]
        self.endpoint = endpoint
        self.start_time = time.time()
        self.stages: Dict[str, float] = {}
        self.stage_order = []
        self.metadata: Dict = {}
        self._current_stage = None
        self._current_stage_start = None

    @contextmanager
    def stage(self, name: str):
        """追踪某个阶段的耗时"""
        self._current_stage = name
        self._current_stage_start = time.time()
        span = _tracer.start_span(f"{self.endpoint}.{name}")
        span.set_attribute("trace_id", self.trace_id)
        try:
            yield span
            span.set_status(StatusCode.OK)
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise
        finally:
            elapsed = time.time() - self._current_stage_start
            self.stages[name] = round(elapsed, 4)
            self.stage_order.append(name)
            STAGE_LATENCY.labels(stage=name).observe(elapsed)
            span.set_attribute("duration_seconds", elapsed)
            span.end()

            # 告警检查
            threshold_key = f"{name}_duration"
            if threshold_key in ALERT_THRESHOLDS and elapsed > ALERT_THRESHOLDS[threshold_key]:
                ALERT_COUNT.labels(alert_type=f"slow_{name}").inc()
                logger.warning(f"[告警] trace={self.trace_id} 阶段'{name}'耗时{elapsed:.3f}s超过阈值{ALERT_THRESHOLDS[threshold_key]}s")

    def finish(self, status: str = "success") -> Dict:
        """完成追踪，返回追踪摘要"""
        total = round(time.time() - self.start_time, 4)

        # 记录Prometheus指标
        REQUEST_COUNT.labels(endpoint=self.endpoint, status=status).inc()
        REQUEST_LATENCY.labels(endpoint=self.endpoint).observe(total)

        # 总耗时告警
        if total > ALERT_THRESHOLDS['request_duration']:
            ALERT_COUNT.labels(alert_type="slow_request").inc()
            logger.warning(f"[告警] trace={self.trace_id} 总耗时{total:.3f}s超过阈值{ALERT_THRESHOLDS['request_duration']}s")

        summary = {
            'trace_id': self.trace_id,
            'endpoint': self.endpoint,
            'total_time': total,
            'stages': {s: self.stages[s] for s in self.stage_order},
            'status': status,
            'metadata': self.metadata,
        }

        logger.info(f"[追踪] trace={self.trace_id} endpoint={self.endpoint} total={total:.3f}s stages={self.stages}")
        return summary


# ============ 追踪历史存储 ============

class TraceStore:
    """追踪记录存储（内存，最近1000条）"""

    def __init__(self, max_size: int = 1000):
        self._traces = []
        self._lock = threading.Lock()
        self._max_size = max_size

    def add(self, trace_summary: Dict):
        with self._lock:
            self._traces.append(trace_summary)
            if len(self._traces) > self._max_size:
                self._traces = self._traces[-self._max_size:]

    def get_recent(self, n: int = 50) -> list:
        with self._lock:
            return list(self._traces[-n:])

    def get_stats(self) -> Dict:
        with self._lock:
            if not self._traces:
                return {'total': 0}
            durations = [t['total_time'] for t in self._traces]
            return {
                'total': len(self._traces),
                'avg_duration': round(sum(durations) / len(durations), 3),
                'max_duration': round(max(durations), 3),
                'min_duration': round(min(durations), 3),
                'slow_requests': sum(1 for d in durations if d > 3.0),
            }


trace_store = TraceStore()


# ============ 告警历史 ============

class AlertStore:
    """告警记录存储"""

    def __init__(self, max_size: int = 500):
        self._alerts = []
        self._lock = threading.Lock()
        self._max_size = max_size

    def add(self, alert_type: str, message: str, trace_id: str = ""):
        with self._lock:
            self._alerts.append({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'type': alert_type,
                'message': message,
                'trace_id': trace_id,
            })
            if len(self._alerts) > self._max_size:
                self._alerts = self._alerts[-self._max_size:]

    def get_recent(self, n: int = 50) -> list:
        with self._lock:
            return list(self._alerts[-n:])


alert_store = AlertStore()


# ============ Prometheus指标端点 ============

def get_metrics() -> bytes:
    """返回Prometheus格式的指标"""
    return generate_latest()


def get_metrics_content_type() -> str:
    return CONTENT_TYPE_LATEST
