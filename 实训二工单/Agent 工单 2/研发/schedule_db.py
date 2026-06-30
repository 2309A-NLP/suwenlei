"""
日程提醒智能体 - 数据库工具
工单编号：人工智能NLP-Agent数字人项目-日程提醒智能体任务

职责：
- 初始化 SQLite 数据库与日程表（schedules）；
- 提供日程的增（add）、删（delete）、改（update）、查（query）；
- 日程信息（事项内容、时间、循环规则）全部持久化到数据库；
- 提供「按日期取应出现的日程」能力，供提醒引擎使用。

循环规则 recurrence：7 位字符串，依次对应 周一~周日，'1' 表示当天循环，'0' 表示不循环。
例如 '1000100' 表示每周一、周五循环；'0000000' 表示不循环（一次性日程）。
"""

import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("schedule_db")

# 不循环（一次性日程）的循环规则常量
NO_RECUR = "0000000"


class ScheduleDB:
    """日程数据库操作类（SQLite + 上下文管理器，自动提交/回滚）。"""

    def __init__(self, db_path: str = "schedules.db"):
        self.db_path = db_path
        self.init_db()

    @contextmanager
    def _conn(self):
        """获取数据库连接：正常提交，异常回滚，最终关闭。"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:  # noqa: BLE001
            conn.rollback()
            logger.error(f"数据库操作失败：{e}")
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        """初始化数据库与日程表。"""
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schedules (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    time          TEXT    NOT NULL,                 -- 提醒时间 HH:MM（24 小时制）
                    recurrence    TEXT    NOT NULL DEFAULT '0000000', -- 循环规则：周一~周日 7 位，1 循环 0 不循环
                    content       TEXT    NOT NULL,                 -- 事项内容
                    schedule_date TEXT,                             -- 一次性日程的具体日期 YYYY-MM-DD；循环日程为 NULL
                    enabled       INTEGER NOT NULL DEFAULT 1,       -- 是否启用（1 启用 0 停用）
                    last_reminded TEXT,                             -- 最近一次提醒的日期，避免同一天重复提醒
                    created_at    TEXT    DEFAULT (datetime('now','localtime'))
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_time ON schedules(time)")
            logger.info("数据库初始化完成（表：schedules）")

    # ----------------------------- 增 -----------------------------
    def add(self, time: str, recurrence: str, content: str,
            schedule_date: Optional[str] = None) -> int:
        """新增一条日程，返回自增 ID。"""
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO schedules (time, recurrence, content, schedule_date) "
                "VALUES (?, ?, ?, ?)",
                (time, recurrence, content, schedule_date),
            )
            sid = cur.lastrowid
            logger.info(
                f"[DB 调用] 新增日程 ID={sid} {time}|{recurrence}|{content} date={schedule_date}"
            )
            return sid

    # ----------------------------- 查 -----------------------------
    def get(self, sid: int) -> Optional[Dict[str, Any]]:
        """按 ID 查询单条日程。"""
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM schedules WHERE id=?", (sid,)).fetchone()
            logger.info(f"[DB 调用] 查询日程 ID={sid} -> {'命中' if row else '未命中'}")
            return dict(row) if row else None

    def list_all(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """列出全部日程（按时间升序）。"""
        sql = "SELECT * FROM schedules"
        if enabled_only:
            sql += " WHERE enabled=1"
        sql += " ORDER BY time ASC, id ASC"
        with self._conn() as conn:
            rows = [dict(r) for r in conn.execute(sql).fetchall()]
            logger.info(f"[DB 调用] 列出全部日程，共 {len(rows)} 条")
            return rows

    def list_for_date(self, date_str: str) -> List[Dict[str, Any]]:
        """返回某一天（YYYY-MM-DD）应出现的日程：当天的一次性日程 + 命中当天星期的循环日程。"""
        weekday = datetime.strptime(date_str, "%Y-%m-%d").weekday()  # 0=周一 ... 6=周日
        result: List[Dict[str, Any]] = []
        for s in self.list_all():
            if s["schedule_date"] == date_str:
                result.append(s)
            elif s["recurrence"] != NO_RECUR and s["recurrence"][weekday] == "1":
                result.append(s)
        result.sort(key=lambda x: x["time"])
        return result

    # ----------------------------- 改 -----------------------------
    def update(self, sid: int, **fields: Any) -> bool:
        """更新指定字段，返回是否有行被修改。"""
        if not fields:
            return False
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._conn() as conn:
            cur = conn.execute(
                f"UPDATE schedules SET {cols} WHERE id=?",
                (*fields.values(), sid),
            )
            ok = cur.rowcount > 0
            if ok:
                logger.info(f"[DB 调用] 更新日程 ID={sid} {fields}")
            return ok

    # ----------------------------- 删 -----------------------------
    def delete(self, sid: int) -> bool:
        """按 ID 删除日程，返回是否删除成功。"""
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM schedules WHERE id=?", (sid,))
            ok = cur.rowcount > 0
            logger.info(f"[DB 调用] 删除日程 ID={sid} -> {'成功' if ok else '未找到'}")
            return ok

    def clear(self) -> int:
        """清空所有日程（主要用于测试），返回删除条数。"""
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM schedules")
            return cur.rowcount


if __name__ == "__main__":
    # 简单自测
    db = ScheduleDB("test_schedules.db")
    db.clear()
    print("新增：", db.add("17:00", NO_RECUR, "开会", schedule_date=datetime.now().strftime("%Y-%m-%d")))
    print("新增：", db.add("15:15", "0000001", "提醒我买咖啡"))
    print("全部：")
    for s in db.list_all():
        print("  ", s["id"], s["time"], s["recurrence"], s["content"], s["schedule_date"])
