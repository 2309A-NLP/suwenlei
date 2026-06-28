# -*- coding: utf-8 -*-
"""
家庭记账本智能体 - 单元测试

测试覆盖：
- 基本功能测试（记账、查询、统计、删除）
- 日期解析测试（绝对日期、相对日期）
- 成员识别测试（标准称呼、别名）
- 边界测试（空输入、特殊字符）
- 异常测试（错误格式、缺失字段）
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta

# 源码位于 研发/ 目录，将其加入导入路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "研发"))

from agent import BookkeepingAgent
import database_tool
from database_tool import MoneyNotesDB


class BookkeepingAgentTestCase(unittest.TestCase):
    """智能体基本功能测试"""
    
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_money_notes.db")
        self.test_db = MoneyNotesDB(self.db_path)

        # 保存原始的全局 db 实例
        self.original_db = database_tool.db
        # 替换为测试数据库
        database_tool._db_instance = self.test_db
        database_tool.db = self.test_db
        
        self.agent = BookkeepingAgent()

    def tearDown(self):
        # 恢复原始 db 实例
        database_tool.db = self.original_db
        database_tool._db_instance = self.original_db
        self.temp_dir.cleanup()

    def test_expense_record(self):
        """测试支出记账"""
        response = self.agent.chat("今天女儿买了双登山鞋 499 元")
        self.assertIn("已记录支出", response)
        records = self.test_db.get_all_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["member"], "女儿")
        self.assertEqual(records[0]["category"], "服装")
        self.assertIn("登山鞋", records[0]["item"])
        self.assertEqual(records[0]["amount"], -499)

    def test_income_record(self):
        """测试收入记账"""
        response = self.agent.chat("7 月 5 日妈妈收到报销 1000 元")
        self.assertIn("已记录收入", response)
        records = self.test_db.get_all_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["member"], "妈妈")
        self.assertEqual(records[0]["amount"], 1000)
        self.assertEqual(records[0]["date"], "2026-07-05")

    def test_query_monthly_records(self):
        """测试查询月度记录"""
        with patch("agent.datetime") as mock_datetime:
            mock_now = mock_datetime.now.return_value
            mock_now.strftime.return_value = "2026-06"
            self.test_db.add_record("2026-06-10", "女儿", "服装", -499, "登山鞋", "测试")
            response = self.agent.chat("看下这个月家里花钱明细")
        self.assertIn("找到 1 条记录", response)
        self.assertIn("登山鞋", response)

    def test_stats_member_monthly(self):
        """测试成员统计"""
        self.test_db.add_record("2026-06-10", "女儿", "服装", -499, "登山鞋", "测试")
        response = self.agent.chat("这个月女儿花了多少钱？")
        self.assertIn("统计结果", response)
        self.assertIn("女儿", response)
        self.assertIn("499", response)

    def test_confirm_delete_priority(self):
        """测试删除确认流程"""
        self.test_db.add_record("2026-06-10", "女儿", "旅游", -500, "旅游团", "测试")
        preview = self.agent.chat("删除女儿报旅游团的费用")
        self.assertIn("待删除记录", preview)
        result = self.agent.chat("确认删除")
        self.assertIn("已删除 1 条记录", result)
        self.assertEqual(len(self.test_db.get_all_records()), 0)

    def test_delete_extracts_free_text_keyword(self):
        """测试删除关键词提取"""
        self.test_db.add_record("2026-06-10", "女儿", "教育", -120, "数学辅导班", "测试")
        preview = self.agent.chat("删除女儿数学辅导班的费用")
        self.assertIn("待删除记录", preview)
        self.assertIn("数学辅导班", preview)


class DateParsingTestCase(unittest.TestCase):
    """日期解析测试"""
    
    def setUp(self):
        self.agent = BookkeepingAgent()

    def test_today(self):
        """测试'今天'解析"""
        parsed = self.agent.parse_date("今天")
        expected = datetime.now().strftime("%Y-%m-%d")
        self.assertEqual(parsed, expected)

    def test_yesterday(self):
        """测试'昨天'解析"""
        parsed = self.agent.parse_date("昨天")
        expected = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        self.assertEqual(parsed, expected)

    def test_tomorrow(self):
        """测试'明天'解析"""
        parsed = self.agent.parse_date("明天")
        expected = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        self.assertEqual(parsed, expected)

    def test_day_before_yesterday(self):
        """测试'前天'解析"""
        parsed = self.agent.parse_date("前天")
        expected = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        self.assertEqual(parsed, expected)

    def test_this_month(self):
        """测试'这个月'解析"""
        parsed = self.agent.parse_date("这个月")
        expected = datetime.now().strftime("%Y-%m")
        self.assertEqual(parsed, expected)

    def test_last_week(self):
        """测试'上周'解析"""
        parsed = self.agent.parse_date("上周")
        expected = (datetime.now() - timedelta(days=datetime.now().weekday() + 7)).strftime("%Y-%m-%d")
        self.assertEqual(parsed, expected)

    def test_last_month(self):
        """测试'上个月'解析"""
        parsed = self.agent.parse_date("上个月")
        first_day = datetime.now().replace(day=1) - timedelta(days=1)
        expected = first_day.replace(day=1).strftime("%Y-%m-%d")
        self.assertEqual(parsed, expected)

    def test_month_day_format(self):
        """测试'7 月 5 日'格式"""
        parsed = self.agent.parse_date("7 月 5 日")
        self.assertEqual(parsed, "2026-07-05")

    def test_month_day_without_spaces(self):
        """测试'7 月 5 日'无空格格式"""
        parsed = self.agent.parse_date("7 月 5 日")
        self.assertEqual(parsed, "2026-07-05")

    def test_full_date_format(self):
        """测试完整日期格式"""
        # 年份用当前年份，避免跨年问题
        current_year = datetime.now().year
        parsed = self.agent.parse_date(f"{current_year}年 1 月 15 日")
        self.assertEqual(parsed, f"{current_year}-01-15")

    def test_iso_date_format(self):
        """测试 ISO 日期格式"""
        parsed = self.agent.parse_date("2025-01-15")
        self.assertEqual(parsed, "2025-01-15")

    def test_invalid_date(self):
        """测试无效日期返回 None"""
        parsed = self.agent.parse_date("随便一个日期")
        self.assertIsNone(parsed)


class MemberParsingTestCase(unittest.TestCase):
    """成员识别测试"""
    
    def setUp(self):
        self.agent = BookkeepingAgent()

    def test_standard_members(self):
        """测试标准成员称呼"""
        self.assertEqual(self.agent.parse_member("爸爸"), "爸爸")
        self.assertEqual(self.agent.parse_member("妈妈"), "妈妈")
        self.assertEqual(self.agent.parse_member("女儿"), "女儿")

    def test_father_aliases(self):
        """测试爸爸别名"""
        for alias in ["老公", "丈夫", "父亲", "老爸"]:
            with self.subTest(alias=alias):
                self.assertEqual(self.agent.parse_member(alias), "爸爸")

    def test_mother_aliases(self):
        """测试妈妈别名"""
        for alias in ["老婆", "媳妇", "妻子", "母亲", "老妈", "我家领导"]:
            with self.subTest(alias=alias):
                self.assertEqual(self.agent.parse_member(alias), "妈妈")

    def test_daughter_aliases(self):
        """测试女儿别名"""
        for alias in ["孩子", "闺女"]:
            with self.subTest(alias=alias):
                self.assertEqual(self.agent.parse_member(alias), "女儿")

    def test_first_person(self):
        """测试'我'返回 None"""
        self.assertIsNone(self.agent.parse_member("我"))

    def test_unknown_member(self):
        """测试未知成员返回 None"""
        self.assertIsNone(self.agent.parse_member("陌生人"))


class AmountParsingTestCase(unittest.TestCase):
    """金额解析测试"""
    
    def setUp(self):
        self.agent = BookkeepingAgent()

    def test_integer_amount(self):
        """测试整数金额"""
        self.assertEqual(self.agent.parse_amount("100 元"), 100.0)

    def test_yuan_unit(self):
        """测试'元'单位"""
        self.assertEqual(self.agent.parse_amount("500 元"), 500.0)

    def test_kuai_unit(self):
        """测试'块'单位"""
        self.assertEqual(self.agent.parse_amount("500 块"), 500.0)

    def test_invalid_amount(self):
        """测试无效金额返回 None"""
        self.assertIsNone(self.agent.parse_amount("一百元"))
        self.assertIsNone(self.agent.parse_amount("abc 元"))


class IntentDetectionTestCase(unittest.TestCase):
    """意图识别测试"""
    
    def setUp(self):
        self.agent = BookkeepingAgent()

    def test_expense_intent(self):
        """测试支出意图"""
        self.assertEqual(self.agent.detect_intent("今天花了 100 元"), "expense")

    def test_income_intent(self):
        """测试收入意图"""
        self.assertEqual(self.agent.detect_intent("今天收到工资 5000 元"), "income")

    def test_query_intent(self):
        """测试查询意图"""
        self.assertEqual(self.agent.detect_intent("查看这个月明细"), "query")

    def test_stats_intent(self):
        """测试统计意图"""
        self.assertEqual(self.agent.detect_intent("这个月花了多少钱"), "stats")

    def test_delete_intent(self):
        """测试删除意图"""
        self.assertEqual(self.agent.detect_intent("删除这条记录"), "delete")

    def test_confirm_delete_intent(self):
        """测试确认删除意图"""
        self.assertEqual(self.agent.detect_intent("确认删除"), "confirm_delete")

    def test_export_intent(self):
        """测试导出意图"""
        self.assertEqual(self.agent.detect_intent("导出数据"), "export")

    def test_budget_intent(self):
        """测试预算意图"""
        self.assertEqual(self.agent.detect_intent("这个月预算 5000 元"), "budget")

    def test_unknown_intent(self):
        """测试未知意图"""
        self.assertEqual(self.agent.detect_intent("随便说点什么"), "unknown")


class EdgeCaseTestCase(unittest.TestCase):
    """边界和异常测试"""
    
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_money_notes.db")
        self.test_db = MoneyNotesDB(self.db_path)
        self.original_db = database_tool.db
        database_tool.db = self.test_db
        self.agent = BookkeepingAgent()

    def tearDown(self):
        database_tool.db = self.original_db
        self.temp_dir.cleanup()

    def test_empty_input(self):
        """测试空输入"""
        response = self.agent.chat("")
        self.assertIn("没太理解", response)

    def test_missing_member(self):
        """测试缺失成员"""
        response = self.agent.chat("今天花了 100 元")
        # 没有成员时应该返回引导信息
        self.assertIn("请问", response)
        self.assertIn("成员", response)

    def test_missing_amount(self):
        """测试缺失金额"""
        response = self.agent.chat("今天女儿买东西")
        # 没有金额时应该返回引导或未知（取决于实现）
        # 当前实现会返回 unknown 因为没有检测到金额
        self.assertIn("没太理解", response)

    def test_special_characters(self):
        """测试特殊字符"""
        response = self.agent.chat("今天女儿买了\"三体\"书 50 元")
        # 有特殊字符但能正确解析
        self.assertIn("已记录", response)

    def test_multiple_amounts(self):
        """测试多个金额（取第一个）"""
        response = self.agent.chat("今天女儿花了 100 元又花了 200 元")
        # 有多个金额时取第一个
        self.assertIn("已记录", response)

    def test_delete_nonexistent(self):
        """测试删除不存在的记录"""
        response = self.agent.chat("删除不存在的记录")
        self.assertIn("未找到", response)

    def test_confirm_without_pending(self):
        """测试无待确认时确认删除"""
        response = self.agent.chat("确认删除")
        self.assertIn("没有待确认", response)


class CategoryMappingTestCase(unittest.TestCase):
    """类别映射测试"""
    
    def setUp(self):
        self.agent = BookkeepingAgent()

    def test_book_category(self):
        """测试买书类别"""
        for keyword in ["买书", "书", "三体"]:
            category, item = self.agent.parse_category_and_item(f"买了{keyword}")
            self.assertEqual(category, "买书")

    def test_clothing_category(self):
        """测试服装类别"""
        for keyword in ["衣服", "鞋", "登山鞋"]:
            category, item = self.agent.parse_category_and_item(f"买了{keyword}")
            self.assertEqual(category, "服装")

    def test_food_category(self):
        """测试餐饮类别"""
        for keyword in ["吃饭", "外卖", "下馆子"]:
            category, item = self.agent.parse_category_and_item(f"{keyword}")
            self.assertEqual(category, "餐饮")

    def test_transport_category(self):
        """测试交通类别"""
        for keyword in ["打车", "地铁", "公交", "加油"]:
            category, item = self.agent.parse_category_and_item(f"{keyword}")
            self.assertEqual(category, "交通")

    def test_education_category(self):
        """测试教育类别"""
        for keyword in ["补习班", "辅导班", "学费"]:
            category, item = self.agent.parse_category_and_item(f"上了{keyword}")
            self.assertEqual(category, "教育")


if __name__ == "__main__":
    unittest.main(verbosity=2)
