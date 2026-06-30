"""
日程提醒智能体 - 命令行入口
工单编号：人工智能NLP-Agent数字人项目-日程提醒智能体任务

运行：python main.py
- 启动后进入对话循环；
- 同时在后台启动提醒引擎，日程到点会主动弹出温馨提醒；
- 输入「测试」运行内置用例，输入「退出」结束。
"""

import sys
import logging
from datetime import datetime
from typing import Dict, Any

from schedule_db import ScheduleDB
from schedule_agent import ScheduleAgent
from reminder import ReminderEngine

# 控制台只显示提醒等关键信息，DB 调用日志降到 WARNING 以免刷屏
logging.getLogger("schedule_db").setLevel(logging.WARNING)


def on_reminder(message: str, schedule: Dict[str, Any]) -> None:
    """提醒到点时的回调：打印到控制台。"""
    print(f"\n[提醒] {message}\n您：", end="", flush=True)


def main() -> None:
    db = ScheduleDB()
    agent = ScheduleAgent(db)
    engine = ReminderEngine(db, on_reminder, interval=20)
    engine.start()

    print("=" * 60)
    print("日程提醒智能体")
    print("=" * 60)
    print(agent.opening_message)
    print("\n（输入「测试」运行内置用例，输入「退出」结束）")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n您：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见，祝您一切顺利！")
            break

        if not user_input:
            continue
        if user_input in ("退出", "exit", "quit", "q"):
            print("再见，祝您一切顺利！")
            break
        if user_input == "测试":
            from test_agent import run_acceptance_tests
            run_acceptance_tests(agent, db)
            continue

        print(f"\n智能体：{agent.chat(user_input)}")

    engine.stop()


if __name__ == "__main__":
    sys.exit(main())
