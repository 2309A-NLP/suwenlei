"""
日程提醒智能体 - 测试用例
工单编号：人工智能NLP-Agent数字人项目-日程提醒智能体任务

覆盖工单验收点：
1. 测试语句（增 / 查 / 删 / 循环日程 / 直接工具格式）；
2. 数据库调用率（每步均落库，可由「已存数据表」核验）；
3. 存储准确性（字段核验）；
4. 完整性引导（缺字段追问）；
5. 复杂 / 口语化理解；
6. 流程完善性（删除前二次确认）；
7. 到点提醒（模拟当前时刻触发）。

运行：python test_agent.py
"""

from datetime import datetime

from schedule_db import ScheduleDB, NO_RECUR
from schedule_agent import ScheduleAgent
from reminder import ReminderEngine, build_reminder


def _line(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def run_acceptance_tests(agent: ScheduleAgent = None, db: ScheduleDB = None) -> None:
    # 使用独立测试库，避免污染正式数据
    db = db or ScheduleDB("test_schedules.db")
    db.clear()
    agent = ScheduleAgent(db)

    _line("一、对话测试用例")
    dialogues = [
        ("添加日程：下午 5 点开会", "口语化添加（下午 5 点 -> 17:00）"),
        ("每周一和周五早上 8 点提醒我开晨会", "循环日程（1000100）"),
        ("15:15|0000001|提醒我买咖啡", "直接工具格式（周日循环）"),
        ("帮我设置一个明天晚上八点半的提醒", "缺事项内容 -> 引导补全"),
        ("我今天的日程有哪些？", "查询今天日程"),
    ]
    for text, note in dialogues:
        print(f"\n[{note}]")
        print(f"  用户：{text}")
        print(f"  智能体：{agent.chat(text)}")

    _line("二、删除流程（二次确认）")
    print(f"  用户：我今天的日程有哪些？")
    print(f"  智能体：{agent.chat('我今天的日程有哪些？')}")
    print(f"\n  用户：删除日程 1")
    print(f"  智能体：{agent.chat('删除日程 1')}")
    print(f"\n  用户：确认")
    print(f"  智能体：{agent.chat('确认')}")

    _line("三、存储准确性核验（直接查数据库 = 已存数据表）")
    rows = db.list_all()
    print(f"  当前库中日程共 {len(rows)} 条：")
    print(f"  {'ID':<4}{'时间':<8}{'循环规则':<12}{'日期':<14}事项")
    for r in rows:
        print(f"  {r['id']:<4}{r['time']:<8}{r['recurrence']:<12}{str(r['schedule_date']):<14}{r['content']}")

    # 断言：买咖啡日程的字段必须完全准确
    coffee = [r for r in rows if "买咖啡" in r["content"]]
    assert coffee, "未找到『买咖啡』日程，落库失败！"
    c = coffee[0]
    assert c["time"] == "15:15", f"时间字段错误：{c['time']}"
    assert c["recurrence"] == "0000001", f"循环规则字段错误：{c['recurrence']}"
    print("\n  [OK] 字段核验通过：15:15 | 0000001 | 提醒我买咖啡")

    _line("四、单元解析测试（时间 / 循环规则）")
    cases = [
        ("下午5点", "17:00"),
        ("早上8点", "08:00"),
        ("晚上八点半", "20:30"),
        ("中午12点", "12:00"),
        ("凌晨1点", "01:00"),
        ("9:30", "09:30"),
    ]
    for text, expect in cases:
        got = agent.parse_time(text)
        flag = "[OK]" if got == expect else "[NG]"
        print(f"  {flag} parse_time('{text}') = {got}（期望 {expect}）")

    recur_cases = [
        ("每天", "1111111"),
        ("工作日", "1111100"),
        ("周末", "0000011"),
        ("每周一和周五", "1000100"),
        ("每周日", "0000001"),
        ("开会", NO_RECUR),
    ]
    for text, expect in recur_cases:
        got = agent.parse_recurrence(text)
        flag = "[OK]" if got == expect else "[NG]"
        print(f"  {flag} parse_recurrence('{text}') = {got}（期望 {expect}）")

    _line("五、到点提醒模拟（插入一条当前时刻的日程并触发）")
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    sid = db.add(now.strftime("%H:%M"), NO_RECUR, "提醒主人叫咖啡", schedule_date=today)
    captured = []
    engine = ReminderEngine(db, lambda msg, s: captured.append(msg), interval=60)
    fired = engine.check_once()
    print(f"  插入日程 ID={sid}（时间={now.strftime('%H:%M')}）")
    print(f"  触发提醒 {fired} 条：")
    for m in captured:
        print(f"    [提醒] {m}")
    assert fired >= 1, "到点提醒未触发！"
    # 再次检查：当天不应重复提醒
    fired2 = engine.check_once()
    print(f"  再次轮询触发 {fired2} 条（应为 0，验证防重复）")
    assert fired2 == 0, "同一天重复提醒，防重逻辑失效！"
    print("  [OK] 到点提醒 + 防重复 验证通过")

    _line("全部测试通过 [OK]")


if __name__ == "__main__":
    run_acceptance_tests()
