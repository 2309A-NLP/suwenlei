"""
日程提醒智能体 - 提醒引擎
工单编号：人工智能NLP-Agent数字人项目-日程提醒智能体任务

职责：
- 后台线程每隔若干秒轮询数据库；
- 当某条日程的时间 == 当前时间（HH:MM），且当天尚未提醒过时，触发提醒回调；
- 通过 last_reminded 字段防止同一天重复提醒；
- 提醒文案采用工单约定的多种温馨话术。
"""

import threading
import logging
from datetime import datetime
from typing import Callable, Dict, Any

from schedule_db import ScheduleDB

logger = logging.getLogger("reminder")

# 提醒话术模板（{c} 为事项内容），与工单示例一致
REMINDER_TEMPLATES = [
    "温馨提醒：{c}的时间到啦，主人！",
    "主人！是时候{c}了哦~",
    "亲爱的主人，现在是{c}的时候啦！",
    "嘿，主人，该{c}了哦~",
]


def build_reminder(schedule: Dict[str, Any]) -> str:
    """根据日程生成一句提醒文案（按 ID 轮换话术，保证可复现）。"""
    template = REMINDER_TEMPLATES[schedule["id"] % len(REMINDER_TEMPLATES)]
    return template.format(c=schedule["content"])


class ReminderEngine:
    """后台提醒引擎。"""

    def __init__(self, db: ScheduleDB, callback: Callable[[str, Dict[str, Any]], None],
                 interval: int = 20):
        """
        :param db:        数据库实例
        :param callback:  到点回调，签名 callback(message, schedule)
        :param interval:  轮询间隔（秒）
        """
        self.db = db
        self.callback = callback
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()
        logger.info(f"提醒引擎已启动（每 {self.interval}s 轮询一次）")

    def stop(self) -> None:
        self._stop.set()

    def check_once(self) -> int:
        """执行一次检查，返回本次触发的提醒数量（也便于测试直接调用）。"""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        hm = now.strftime("%H:%M")
        fired = 0
        for s in self.db.list_for_date(date_str):
            if s["time"] == hm and s["last_reminded"] != date_str:
                message = build_reminder(s)
                self.db.update(s["id"], last_reminded=date_str)
                self.callback(message, s)
                fired += 1
        return fired

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.check_once()
            except Exception as e:  # noqa: BLE001
                logger.error(f"提醒轮询出错：{e}")
            self._stop.wait(self.interval)
