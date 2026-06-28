# -*- coding: utf-8 -*-
"""
家庭记账本智能体 - 功能演示 (简化版)
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

# 源码位于 研发/ 目录，将其加入导入路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "研发"))

from agent import BookkeepingAgent

agent = BookkeepingAgent()

print("=" * 70)
print("家庭记账本智能体 - 更多可用表达方式演示")
print("=" * 70)

# 日期表达多样性
print("\n【日期表达多样性】")
print("-" * 70)
date_examples = [
    "今天老公吃饭花了 200 元",
    "昨天老婆买衣服花了 800 元",
    "前天孩子买玩具花了 150 元",
    "1 月 15 日爸爸加油花了 300 元",
    "2025-02-14 妈妈美容花了 500 元",
]

for example in date_examples:
    response = agent.chat(example)
    print(f"输入：{example}")
    print(f"输出：{response}\n")

# 成员称呼多样性
print("\n【成员称呼多样性】")
print("-" * 70)
member_examples = [
    "我老公今天工资发了 15000 元",
    "我老婆昨天买化妆品花了 600 元",
    "我家孩子买书花了 100 元",
]

for example in member_examples:
    response = agent.chat(example)
    print(f"输入：{example}")
    print(f"输出：{response}\n")

# 收支类别多样性
print("\n【收支类别多样性】")
print("-" * 70)
category_examples = [
    "今天爸爸工资收入 8000 元",
    "妈妈今天收到奖金 3000 元",
    "今天家里买菜花了 150 元",
    "老公今天请客吃饭花了 500 元",
    "女儿上补习班花了 1000 元",
    "家里买电视花了 3000 元",
    "全家旅游花了 5000 元",
]

for example in category_examples:
    response = agent.chat(example)
    print(f"输入：{example}")
    print(f"输出：{response}\n")

# 查询方式多样性
print("\n【查询方式多样性】")
print("-" * 70)
query_examples = [
    "看看这个月家里花钱明细",
    "查询爸爸的消费记录",
    "统计一下家里总收入",
    "女儿总共花了多少钱",
]

for example in query_examples:
    response = agent.chat(example)
    print(f"输入：{example}")
    print(f"输出：{response[:150]}...\n")

print("=" * 70)
print("演示完成！智能体支持丰富的自然语言表达方式")
print("=" * 70)
