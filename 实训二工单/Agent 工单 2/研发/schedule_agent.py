"""
日程提醒智能体 - 核心智能体
工单编号：人工智能NLP-Agent数字人项目-日程提醒智能体任务

设计为「工具调用型」Agent：
- 智能体负责理解用户的自然语言（含口语化表达），识别意图（增/删/查/确认）；
- 「添加日程」统一翻译为标准工具格式  HH:MM|YYYYYYY|事项内容  再调用 set_schedule 工具；
- 所有日程数据通过 ScheduleDB 持久化到数据库（满足"任何情况下都要调用数据库"的要求）；
- 对缺失字段（时间/内容）主动追问引导，而非记录残缺数据；
- 删除操作做二次确认，降低误删风险与模型不确定性。

工具格式规范（与工单一致）：
  用于设置日程，接受 3 个参数，格式为 HH:MM|YYYYYYY|事项内容，标点必须为英文字符。
  其中 HH:MM 表示时间（24 小时制）；YYYYYYY 表示循环规则（每位代表一天，周一~周日，
  1 为循环，0 为不循环，如 '1000100' 代表每周一和周五循环）；事项内容为提醒的具体内容。
  返回例子：15:15|0000000|提醒主人叫咖啡
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List

from schedule_db import ScheduleDB, NO_RECUR

logger = logging.getLogger("schedule_agent")

# 中文数字 -> 整数（用于解析「下午五点」「日程一」等）
CN_NUM = {
    "零": 0, "〇": 0, "一": 1, "两": 2, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}

# 星期字 -> 索引（0=周一 ... 6=周日）
WEEK_INDEX = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}

# 直接输入工具格式时的匹配（HH:MM|YYYYYYY|内容，容忍中文标点与空格）
DIRECT_RE = re.compile(r"^\s*(\d{1,2})\s*[:：]\s*(\d{1,2})\s*[|｜]\s*([01]{7})\s*[|｜]\s*(.+?)\s*$")


def cn_to_int(s: str) -> Optional[int]:
    """把一个（可能是中文的）数字片段转为整数，支持 0~99。"""
    s = s.strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    if s in CN_NUM:
        return CN_NUM[s]
    if "十" in s:  # 处理 十一/十二/二十/二十三 等
        left, _, right = s.partition("十")
        tens = CN_NUM.get(left, 1) if left else 1
        ones = CN_NUM.get(right, 0) if right else 0
        return tens * 10 + ones
    return None


class ScheduleAgent:
    """日程提醒智能体：自然语言 -> 标准工具格式 -> 数据库。"""

    def __init__(self, db: Optional[ScheduleDB] = None):
        self.db = db or ScheduleDB()
        self.pending_delete: Optional[int] = None  # 待确认删除的日程 ID
        self.index_map: Dict[int, int] = {}        # 列表序号 -> 日程 ID（供"删除日程N"使用）
        self.opening_message = (
            "您好，我是您的日程提醒小助手~ 您可以这样跟我说：\n"
            "  · 添加日程：下午 5 点开会\n"
            "  · 每周一和周五 早上 8 点 提醒我晨会（循环日程）\n"
            "  · 我今天的日程有哪些？\n"
            "  · 删除日程 1"
        )

    # ===================== 自然语言解析 =====================

    @staticmethod
    def _detect_period(text: str) -> Optional[str]:
        """识别时间段：凌晨/上午/中午/下午/晚上。"""
        if "凌晨" in text:
            return "midnight"
        if "中午" in text or "正午" in text:
            return "noon"
        if any(w in text for w in ["下午", "午后", "傍晚"]):
            return "pm"
        if any(w in text for w in ["晚上", "今晚", "夜里", "夜晚", "晚间"]):
            return "pm"
        if any(w in text for w in ["上午", "早上", "早晨", "清晨", "一早"]):
            return "am"
        return None

    @staticmethod
    def _apply_period(hour: int, period: Optional[str]) -> int:
        """根据时间段把 12 小时制小时数换算为 24 小时制。"""
        if period == "pm" and hour < 12:
            hour += 12
        elif period == "am" and hour == 12:
            hour = 0
        elif period == "noon":
            hour = 12
        elif period == "midnight" and hour == 12:
            hour = 0
        return hour % 24

    def parse_time(self, text: str) -> Optional[str]:
        """从文本中解析出 HH:MM（24 小时制）。"""
        period = self._detect_period(text)

        # 1) 直接的 HH:MM / HH：MM
        m = re.search(r"([01]?\d|2[0-3])\s*[:：]\s*([0-5]?\d)", text)
        if m:
            hour, minute = int(m.group(1)), int(m.group(2))
            hour = self._apply_period(hour, period)
            return f"{hour:02d}:{minute:02d}"

        # 2) X 点[Y 分 | 半]，X 可为阿拉伯或中文数字
        m = re.search(
            r"(十[一二]?|二十[一二三四]?|两|[一二三四五六七八九]|\d{1,2})\s*[点时]\s*"
            r"(半|过\s*(?:[一二三四五六七八九十]+|\d{1,2})\s*分?|(?:[一二三四五六七八九十]+|\d{1,2})\s*分)?",
            text,
        )
        if m:
            hour = cn_to_int(m.group(1))
            if hour is None:
                return None
            frag = (m.group(2) or "").strip()
            if frag == "半":
                minute = 30
            elif frag:
                num = re.search(r"([一二三四五六七八九十]+|\d{1,2})", frag)
                minute = cn_to_int(num.group(1)) if num else 0
                minute = minute or 0
            else:
                minute = 0
            hour = self._apply_period(hour, period)
            return f"{hour:02d}:{minute:02d}"

        return None

    @staticmethod
    def parse_recurrence(text: str) -> str:
        """解析循环规则，返回 7 位字符串（周一~周日）。无循环则返回 '0000000'。"""
        if any(w in text for w in ["每天", "每日", "天天"]):
            return "1111111"
        if "工作日" in text:
            return "1111100"
        if "周末" in text:
            return "0000011"

        chars = ["0"] * 7
        found = False
        # 匹配 每周一 / 周一 / 星期一 / 礼拜一（含「周一和周五」这种并列）
        for m in re.finditer(r"(?:每\s*)?(?:周|星期|礼拜)\s*([一二三四五六日天])", text):
            idx = WEEK_INDEX.get(m.group(1))
            if idx is not None:
                chars[idx] = "1"
                found = True
        return "".join(chars) if found else NO_RECUR

    def _extract_content(self, text: str) -> Optional[str]:
        """抽取事项内容：去掉触发词、时间词、循环词后剩余的部分。"""
        t = text
        # 触发词 / 礼貌用语
        for w in ["添加日程", "新增日程", "设置日程", "创建日程", "新建日程",
                  "添加", "新增", "设置", "创建", "新建",
                  "帮我记一下", "帮我记", "帮我", "记一下", "记录一下", "记录",
                  "提醒我一下", "提醒我", "提醒一下", "提醒", "我要", "我想",
                  "请", "日程", "一下"]:
            t = t.replace(w, "")
        # 循环描述
        t = re.sub(r"(每天|每日|天天|工作日|周末)", "", t)
        t = re.sub(r"(?:每\s*)?(?:周|星期|礼拜)\s*[一二三四五六日天和、,，及]+", "", t)
        t = re.sub(r"循环", "", t)
        # 日期词
        t = re.sub(r"(今天|明天|后天|大后天|昨天|当天)", "", t)
        # 时间段
        t = re.sub(r"(凌晨|清晨|早晨|早上|上午|中午|正午|下午|午后|傍晚|晚上|今晚|夜里|夜晚|晚间|一早)", "", t)
        # 时间点：X 点 Y 分 / HH:MM
        t = re.sub(
            r"(十[一二]?|二十[一二三四]?|两|[一二三四五六七八九]|\d{1,2})\s*[点时]\s*"
            r"(半|过|[一二三四五六七八九十]+\s*分?|\d{1,2}\s*分?)?", "", t,
        )
        t = re.sub(r"([01]?\d|2[0-3])\s*[:：]\s*[0-5]?\d", "", t)
        # 量词（"一个/一条..."），避免把"设置一个提醒"误当成内容
        t = re.sub(r"(一个|一条|一项|一次|个|条)", "", t)
        # 收尾清理（含残留的「的」）
        t = t.strip(" \t，。,.!！？?、~:：的")
        return t or None

    def _parse_direct(self, text: str) -> Optional[Tuple[str, str, str]]:
        """解析用户直接输入的工具格式 HH:MM|YYYYYYY|内容。"""
        m = DIRECT_RE.match(text)
        if not m:
            return None
        hour, minute = int(m.group(1)), int(m.group(2))
        if hour > 23 or minute > 59:
            return None
        return f"{hour:02d}:{minute:02d}", m.group(3), m.group(4).strip()

    # ===================== 工具：set_schedule =====================

    def set_schedule(self, tool_str: str) -> Dict[str, str]:
        """
        工具函数：按标准格式 HH:MM|YYYYYYY|事项内容 写入数据库。
        这是智能体唯一的「写日程」入口，确保任何添加都落库。
        """
        m = re.match(r"^(\d{2}):(\d{2})\|([01]{7})\|(.+)$", tool_str)
        if not m:
            raise ValueError(f"工具格式非法：{tool_str}")
        time = f"{m.group(1)}:{m.group(2)}"
        recurrence = m.group(3)
        content = m.group(4)
        # 一次性日程绑定到「今天」；循环日程不绑定具体日期
        schedule_date = None if recurrence != NO_RECUR else datetime.now().strftime("%Y-%m-%d")
        sid = self.db.add(time, recurrence, content, schedule_date)
        return {"id": sid, "time": time, "recurrence": recurrence,
                "content": content, "schedule_date": schedule_date}

    # ===================== 意图识别 =====================

    def detect_intent(self, text: str) -> str:
        if self.pending_delete is not None and any(w in text for w in ["取消", "算了", "不删", "不用了", "先不"]) \
                and "日程" not in text:
            return "cancel"
        if any(w in text for w in ["确认", "确定", "是的", "对", "yes", "嗯", "可以"]) and "日程" not in text:
            return "confirm"
        if any(w in text for w in ["删除", "删掉", "移除", "去掉"]) or re.search(r"取消.*日程|取消第|取消日程", text):
            return "delete"
        if re.search(r"(日程|安排|提醒).*(有哪些|哪些|查看|看看|列表|清单|都有什么|有什么)", text) \
                or any(w in text for w in ["我的日程", "查看日程", "今天的日程", "所有日程", "全部日程"]):
            return "query"
        # 含时间 / 直接格式 / 添加类触发词 -> 添加
        if self._parse_direct(text) or self.parse_time(text) \
                or any(w in text for w in ["添加", "新增", "设置", "创建", "提醒", "日程", "安排"]):
            return "add"
        return "unknown"

    # ===================== 各意图处理 =====================

    def _handle_add(self, text: str) -> str:
        direct = self._parse_direct(text)
        if direct:
            time, recurrence, content = direct
        else:
            time = self.parse_time(text)
            recurrence = self.parse_recurrence(text)
            content = self._extract_content(text)
            # 完整性引导：缺字段就追问，不记录残缺数据
            missing: List[str] = []
            if not time:
                missing.append("时间（例如：下午 5 点 / 17:00）")
            if not content:
                missing.append("事项内容（例如：开会）")
            if missing:
                return "好的~ 还差一点信息，请补充：" + "；".join(missing) + "。"

        # 调用工具落库
        info = self.set_schedule(f"{time}|{recurrence}|{content}")
        recur_desc = self._describe_recurrence(recurrence)
        return (
            f"已为您记录日程（编号 {info['id']}）：\n"
            f"  时间：{time}    {recur_desc}\n"
            f"  事项：{content}\n"
            f"  标准格式：{time}|{recurrence}|{content}\n"
            f"到点我会准时提醒您~"
        )

    def _handle_query(self, text: str) -> str:
        if any(w in text for w in ["所有", "全部"]):
            items = self.db.list_all()
            title = "您目前的全部日程："
        else:
            date = datetime.now().strftime("%Y-%m-%d")
            day_word = "今天"
            if "明天" in text:
                date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                day_word = "明天"
            items = self.db.list_for_date(date)
            title = f"您{day_word}的日程包括："

        if not items:
            return "您当前还没有相关日程哦~ 可以对我说「添加日程：下午 5 点开会」。"

        # 建立 序号 -> ID 映射，供"删除日程N"使用
        self.index_map = {i: s["id"] for i, s in enumerate(items, 1)}
        lines = [title]
        for i, s in enumerate(items, 1):
            recur = self._describe_recurrence(s["recurrence"])
            tag = f"（{recur}）" if s["recurrence"] != NO_RECUR else ""
            lines.append(f"{i}. {s['time']} {s['content']}{tag}")
        return "\n".join(lines)

    def _handle_delete(self, text: str) -> str:
        # 提取序号：阿拉伯数字或中文数字
        idx: Optional[int] = None
        m = re.search(r"(\d+)", text)
        if m:
            idx = int(m.group(1))
        else:
            m = re.search(r"日程\s*([一二三四五六七八九十]+)", text)
            if m:
                idx = cn_to_int(m.group(1))

        if idx is None:
            return "请告诉我要删除第几个日程，例如「删除日程 1」。可以先问「我今天的日程有哪些？」查看编号。"

        # 若还没有列表映射，先按今天的日程建立
        if not self.index_map:
            items = self.db.list_for_date(datetime.now().strftime("%Y-%m-%d"))
            self.index_map = {i: s["id"] for i, s in enumerate(items, 1)}

        sid = self.index_map.get(idx)
        if sid is None:
            return f"没有找到编号为 {idx} 的日程，请先用「我今天的日程有哪些？」确认编号。"

        sched = self.db.get(sid)
        if not sched:
            return f"编号 {idx} 的日程已不存在。"

        # 二次确认（满足「删除内容提前确认」的验收要求）
        self.pending_delete = sid
        self._pending_index = idx
        return (
            f"您确定要删除日程 {idx} 吗？内容是：{sched['time']} {sched['content']}。\n"
            f"回复「确认」执行删除，回复「取消」放弃。"
        )

    def _handle_confirm(self) -> str:
        if self.pending_delete is None:
            return "当前没有需要确认的操作哦~"
        sched = self.db.get(self.pending_delete)
        idx = getattr(self, "_pending_index", "")
        self.db.delete(self.pending_delete)
        self.pending_delete = None
        if not sched:
            return "操作已完成，但日程已不存在。"
        return f"已经删除日程 {idx}，删除的日程内容是：{sched['time']} {sched['content']}"

    def _handle_cancel(self) -> str:
        self.pending_delete = None
        return "好的，已为您取消该操作~"

    # ===================== 辅助 =====================

    @staticmethod
    def _describe_recurrence(recurrence: str) -> str:
        """把循环规则翻译成自然语言。"""
        if recurrence == NO_RECUR:
            return "一次性"
        if recurrence == "1111111":
            return "每天循环"
        if recurrence == "1111100":
            return "工作日循环"
        if recurrence == "0000011":
            return "周末循环"
        names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        days = [names[i] for i, c in enumerate(recurrence) if c == "1"]
        return "每" + "、".join(days) + " 循环" if days else "一次性"

    # ===================== 对话入口 =====================

    def chat(self, user_input: str) -> str:
        text = user_input.strip()
        if not text:
            return "您想说点什么呢？"
        intent = self.detect_intent(text)
        logger.info(f"用户输入：{text} -> 意图：{intent}")
        if intent == "add":
            return self._handle_add(text)
        if intent == "query":
            return self._handle_query(text)
        if intent == "delete":
            return self._handle_delete(text)
        if intent == "confirm":
            return self._handle_confirm()
        if intent == "cancel":
            return self._handle_cancel()
        return (
            "抱歉我没太理解~ 您可以这样说：\n"
            "  · 添加日程：下午 5 点开会\n"
            "  · 我今天的日程有哪些？\n"
            "  · 删除日程 1"
        )
