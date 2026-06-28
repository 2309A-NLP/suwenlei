"""
家庭记账本智能体 - 主程序
工单编号：人工智能 NLP-Agent 数字人项目 - 记账本任务

优化：
- 添加相对日期解析（上周、上个月等）
- 添加预算检查功能
- 添加数据导出功能
- 改进错误处理和日志
- 代码精简和类型注解
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from database_tool import (
    add_income, add_expense, query_expenses, get_monthly_stats,
    delete_expense, confirm_delete, export_data, check_budget
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BookkeepingAgent:
    """家庭记账本智能体"""

    def __init__(self):
        self.pending_delete = None
        self.pending_export = None
        self.opening_message = (
            "您好，欢迎使用咱们小家专属记账本！请按照\"x 年 x 月 x 日，谁做什么事收入/支出多少钱\"的格式来输入。"
            "请告诉我你的账目需求吧~"
        )
        self.member_aliases = {
            "爸爸": ["爸爸", "老公", "丈夫", "父亲", "老爸"],
            "妈妈": ["妈妈", "老婆", "媳妇", "妻子", "母亲", "老妈", "我家领导"],
            "女儿": ["女儿", "孩子", "闺女"],
        }
        self.category_keywords = [
            ("登山鞋", "服装"), ("衣服", "服装"), ("鞋", "服装"),
            ("买书", "买书"), ("三体", "买书"), ("书", "买书"),
            ("下馆子", "餐饮"), ("吃饭", "餐饮"), ("外卖", "餐饮"),
            ("打车", "交通"), ("地铁", "交通"), ("公交", "交通"),
            ("旅游团", "旅游"), ("旅游", "旅游"), ("出去玩", "旅游"),
            ("报销", "报销"), ("工资", "工资"), ("奖金", "奖金"),
            ("购物", "购物"), ("买东西", "购物"),
            ("补习班", "教育"), ("辅导班", "教育"), ("学费", "教育"),
            ("化妆品", "美容"), ("美容", "美容"),
            ("加油", "交通"), ("电影", "娱乐"), ("玩具", "娱乐"),
        ]

    def parse_date(self, text: str) -> Optional[str]:
        """解析日期字符串为标准格式 YYYY-MM-DD 或 YYYY-MM"""
        today = datetime.now()

        # 绝对日期
        if any(word in text for word in ["今天", "今日"]):
            return today.strftime("%Y-%m-%d")
        if any(word in text for word in ["明天", "明日"]):
            return (today + timedelta(days=1)).strftime("%Y-%m-%d")
        if any(word in text for word in ["昨天", "昨日"]):
            return (today - timedelta(days=1)).strftime("%Y-%m-%d")
        if "前天" in text:
            return (today - timedelta(days=2)).strftime("%Y-%m-%d")
        if any(word in text for word in ["这个月", "本月", "当月"]):
            return today.strftime("%Y-%m")
        
        # 相对日期
        if "上周" in text:
            # 上周一
            last_week = today - timedelta(days=today.weekday() + 7)
            return last_week.strftime("%Y-%m-%d")
        if "上个月" in text or "上月" in text:
            first_day = today.replace(day=1) - timedelta(days=1)
            return first_day.replace(day=1).strftime("%Y-%m-%d")
        
        # 月日格式
        match = re.search(r'(\d{1,2})\s*月\s*(\d{1,2})\s*日', text)
        if match:
            month, day = int(match.group(1)), int(match.group(2))
            return f"{today.year}-{month:02d}-{day:02d}"

        # 完整日期格式
        match = re.search(r'(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?', text)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            return f"{year:04d}-{month:02d}-{day:02d}"

        return None

    def parse_member(self, text: str) -> Optional[str]:
        """解析家庭成员"""
        # 先检查别名映射（包含"我"的别名如"我家领导"要先匹配）
        for member, aliases in self.member_aliases.items():
            if any(alias in text for alias in aliases):
                return member
        
        # 再检查单独的"我"（没有别名时返回 None）
        if any(word in text for word in ["我", "自己"]):
            return None
        
        return None

    def parse_amount(self, text: str) -> Optional[float]:
        """解析金额"""
        match = re.search(r'(\d+(?:\.\d+)?)\s*[元块]', text)
        if match:
            return float(match.group(1))
        return None

    def normalize_item(self, item: str) -> str:
        return re.sub(r'\s+', ' ', item).strip(' ，。,.!！？：:')

    def parse_category_and_item(self, text: str) -> Tuple[str, str]:
        """解析类别和项目"""
        category = "其他"
        item = ""

        for keyword, mapped_category in self.category_keywords:
            if keyword in text:
                category = mapped_category
                item = keyword
                break

        match = re.search(r'["“](.+?)["”]', text)
        if match:
            item = self.normalize_item(match.group(1))
            return category, item

        patterns = [
            r'(?:买了 | 购买 | 买)([^\d，。,]+)',
            r'(?:收到 | 收入)([^\d，。,]+)',
            r'(?:报了 | 报名 | 报)([^\d，。,]+)',
            r'(?:花了)([^\d，。,]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                candidate = self.normalize_item(match.group(1))
                if candidate:
                    item = candidate
                    break

        return category, item

    def extract_delete_keyword(self, text: str) -> Optional[str]:
        """提取删除关键词"""
        for keyword, _ in self.category_keywords:
            if keyword in text:
                return keyword

        quoted_match = re.search(r'["“](.+?)["”]', text)
        if quoted_match:
            return self.normalize_item(quoted_match.group(1))

        text = re.sub(r'^(删除 | 删掉 | 移除)', '', text)
        text = re.sub(r'(的)?(费用 | 记录 | 账目 | 这笔账 | 这条记录)$', '', text)

        member = self.parse_member(text)
        if member:
            text = text.replace(member, '', 1)
            for alias in self.member_aliases[member]:
                text = text.replace(alias, '', 1)

        text = re.sub(r'(今天 | 昨天 | 前天 | 这个月 | 本月)', '', text)
        text = self.normalize_item(text)
        return text or None

    def detect_intent(self, text: str) -> str:
        """检测用户意图"""
        if "确认删除" in text:
            return "confirm_delete"
        if "确认导出" in text:
            return "confirm_export"
        if any(word in text for word in ["删除", "删掉", "移除"]):
            return "delete"
        if any(word in text for word in ["导出", "输出", "保存"]):
            return "export"
        if any(word in text for word in ["预算", "超支"]):
            return "budget"
        if any(word in text for word in ["花了多少钱", "总共", "一共", "统计", "汇总"]):
            return "stats"
        if any(word in text for word in ["查询", "查看", "看看", "明细", "记录"]):
            return "query"
        if self.parse_amount(text) is not None:
            if any(word in text for word in ["收入", "收到", "赚", "报销", "工资"]):
                return "income"
            return "expense"
        return "unknown"

    def process_expense(self, text: str) -> str:
        date_str = self.parse_date(text) or datetime.now().strftime("%Y-%m-%d")
        member = self.parse_member(text)
        amount = self.parse_amount(text)
        category, item = self.parse_category_and_item(text)

        missing: List[str] = []
        if not member:
            missing.append("成员（爸爸/妈妈/女儿）")
        if amount is None:
            missing.append("金额")
        if missing:
            return f"请问{', '.join(missing)}是多少？"
        return add_expense(date_str, member, category, amount, item, text)

    def process_income(self, text: str) -> str:
        date_str = self.parse_date(text) or datetime.now().strftime("%Y-%m-%d")
        member = self.parse_member(text)
        amount = self.parse_amount(text)
        category, item = self.parse_category_and_item(text)

        if not member:
            return "请问是家里谁的收入？（爸爸/妈妈/女儿）"
        if amount is None:
            return "请问收入多少钱？"
        return add_income(date_str, member, category, amount, item, text)

    def process_query(self, text: str) -> str:
        member = self.parse_member(text)
        month = self.parse_date(text) if any(word in text for word in ["这个月", "本月", "当月"]) else None
        if month or "明细" in text:
            return query_expenses(member, month or datetime.now().strftime("%Y-%m"))
        if member:
            return query_expenses(member, None)
        return query_expenses(None, None)

    def process_stats(self, text: str) -> str:
        member = self.parse_member(text)
        month = self.parse_date(text) if any(word in text for word in ["这个月", "本月", "当月"]) else None
        return get_monthly_stats(member, month)

    def process_delete(self, text: str) -> str:
        member = self.parse_member(text)
        keyword = self.extract_delete_keyword(text)
        if not keyword:
            return "请问要删除什么记录？请提供关键词（如'旅游团'）"

        self.pending_delete = {"keyword": keyword, "member": member}
        return delete_expense(keyword, member)

    def process_confirm_delete(self) -> str:
        if not self.pending_delete:
            return "没有待确认的删除操作"
        result = confirm_delete(self.pending_delete["keyword"], self.pending_delete.get("member"))
        self.pending_delete = None
        return result

    def process_export(self, text: str) -> str:
        """处理导出请求"""
        member = self.parse_member(text)
        month = self.parse_date(text) if any(word in text for word in ["这个月", "本月", "当月"]) else None
        
        # 生成默认文件名
        today = datetime.now().strftime("%Y%m%d")
        filename = f"记账_{today}"
        if member:
            filename += f"_{member}"
        if month:
            filename += f"_{month}"
        filename += ".csv"
        
        self.pending_export = {"filename": filename, "member": member, "month": month}
        return f"📥 准备导出数据到 {filename}，请回复'确认导出'执行操作。"

    def process_confirm_export(self) -> str:
        """确认导出操作"""
        if not self.pending_export:
            return "没有待确认的导出操作"
        
        result = export_data(
            self.pending_export["filename"],
            member=self.pending_export.get("member"),
            month=self.pending_export.get("month")
        )
        self.pending_export = None
        return result

    def process_budget(self, text: str) -> str:
        """处理预算检查"""
        amount = self.parse_amount(text)
        if amount is None:
            return "请问预算金额是多少？"
        
        month = self.parse_date(text) if any(word in text for word in ["这个月", "本月", "当月"]) else None
        return check_budget(amount, month)

    def chat(self, user_input: str) -> str:
        intent = self.detect_intent(user_input)
        if intent == "expense":
            return self.process_expense(user_input)
        if intent == "income":
            return self.process_income(user_input)
        if intent == "query":
            return self.process_query(user_input)
        if intent == "stats":
            return self.process_stats(user_input)
        if intent == "delete":
            return self.process_delete(user_input)
        if intent == "confirm_delete":
            return self.process_confirm_delete()
        if intent == "export":
            return self.process_export(user_input)
        if intent == "confirm_export":
            return self.process_confirm_export()
        if intent == "budget":
            return self.process_budget(user_input)
        return (
            "抱歉，我没太理解。您可以这样说：\n"
            "  - '今天女儿买了双登山鞋 499 元'（记账）\n"
            "  - '这个月女儿花了多少钱？'（查询）\n"
            "  - '看下这个月家里花钱明细'（统计）\n"
            "  - '删除女儿报旅游团的费用'（删除）\n"
            "  - '导出这个月记账数据'（导出 CSV）\n"
            "  - '这个月预算 5000 元'（预算检查）"
        )


def run_tests(agent: BookkeepingAgent):
    test_cases = [
        ("今天女儿买了双登山鞋 499 元", "支出记账"),
        ("7 月 5 日妈妈收到报销 1000 元", "收入记账"),
        ("看下这个月家里花钱明细", "查询明细"),
        ("这个月女儿花了多少钱？", "成员统计"),
        ("删除女儿报旅游团的费用", "删除预览"),
        ("上周爸爸加油花了 300 元", "相对日期"),
        ("导出这个月记账数据", "数据导出"),
        ("这个月预算 5000 元", "预算检查"),
    ]

    print("\n开始执行测试用例...\n")
    for index, (input_text, description) in enumerate(test_cases, 1):
        print(f"测试 {index} - {description}:")
        print(f"  输入：{input_text}")
        response = agent.chat(input_text)
        print(f"  输出：{response}\n")


def main():
    agent = BookkeepingAgent()

    print("=" * 60)
    print(agent.opening_message)
    print("=" * 60)
    print("\n输入'退出'结束程序\n")

    while True:
        try:
            user_input = input("您：").strip()
            if not user_input:
                continue
            if user_input in ["退出", "exit", "quit"]:
                print("再见！")
                break
            if user_input == "测试":
                run_tests(agent)
                continue

            response = agent.chat(user_input)
            print(f"\n智能体：{response}\n")
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as exc:
            logger.error(f"错误：{exc}")
            print(f"\n出错了：{exc}\n")


if __name__ == "__main__":
    main()
