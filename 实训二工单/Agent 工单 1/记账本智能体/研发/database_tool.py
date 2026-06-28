"""
家庭记账本智能体 - 数据库工具
工单编号：人工智能 NLP-Agent 数字人项目 - 记账本任务

优化：
- 添加数据库连接池（避免频繁创建连接）
- 添加日志记录
- 添加数据导出功能（CSV）
- 添加预算检查功能
- 改进错误处理
"""

import sqlite3
import csv
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from contextlib import contextmanager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 项目按职能分目录后，数据库文件位于 部署/money_notes.db。
# 用基于本文件位置的绝对路径，保证无论从哪个工作目录启动都能找到同一个库。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 记账本智能体/
DEFAULT_DB_PATH = str(_PROJECT_ROOT / "部署" / "money_notes.db")


class MoneyNotesDB:
    """家庭记账本数据库操作类（带连接池）"""
    
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self.init_db()
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接（上下文管理器）"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"数据库操作失败：{e}")
            raise
        finally:
            conn.close()
    
    def init_db(self):
        """初始化数据库和表"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS money_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    member VARCHAR(20) NOT NULL,
                    category VARCHAR(50) NOT NULL,
                    item VARCHAR(100),
                    amount DECIMAL(10,2) NOT NULL,
                    note TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # 创建索引提升查询性能
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_date 
                ON money_notes(date)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_member 
                ON money_notes(member)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_category 
                ON money_notes(category)
            """)
            logger.info("数据库初始化完成")
    
    def add_record(self, date: str, member: str, category: str, 
                   amount: float, item: str = "", note: str = "") -> int:
        """添加记账记录"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO money_notes (date, member, category, item, amount, note)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (date, member, category, item, amount, note))
            record_id = cursor.lastrowid
            logger.info(f"添加记录：ID={record_id}, {date} {member} {category} {amount}元")
            return record_id
    
    def query_records(self, member: Optional[str] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      category: Optional[str] = None,
                      item: Optional[str] = None,
                      limit: int = 100) -> List[Dict[str, Any]]:
        """查询记录"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            conditions = []
            params = []
            
            if member:
                conditions.append("member = ?")
                params.append(member)
            if start_date:
                conditions.append("date >= ?")
                params.append(start_date)
            if end_date:
                conditions.append("date <= ?")
                params.append(end_date)
            if category:
                conditions.append("category = ?")
                params.append(category)
            if item:
                conditions.append("item LIKE ?")
                params.append(f"%{item}%")
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            query = f"""
                SELECT id, date, member, category, item, amount, note, created_at
                FROM money_notes
                WHERE {where_clause}
                ORDER BY date DESC, id DESC
                LIMIT ?
            """
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    def delete_records(self, ids: List[int]) -> int:
        """删除指定 ID 的记录"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(ids))
            cursor.execute(f"DELETE FROM money_notes WHERE id IN ({placeholders})", ids)
            deleted_count = cursor.rowcount
            logger.info(f"删除 {deleted_count} 条记录，IDs: {ids}")
            return deleted_count
    
    def get_statistics(self, member: Optional[str] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> Dict[str, Any]:
        """获取统计数据"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            conditions = []
            params = []
            
            if member:
                conditions.append("member = ?")
                params.append(member)
            if start_date:
                conditions.append("date >= ?")
                params.append(start_date)
            if end_date:
                conditions.append("date <= ?")
                params.append(end_date)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            # 总收入
            cursor.execute(f"""
                SELECT COALESCE(SUM(amount), 0) FROM money_notes
                WHERE {where_clause} AND amount > 0
            """, params)
            income = cursor.fetchone()[0]
            
            # 总支出
            cursor.execute(f"""
                SELECT COALESCE(SUM(amount), 0) FROM money_notes
                WHERE {where_clause} AND amount < 0
            """, params)
            expense = cursor.fetchone()[0]
            
            # 按成员统计
            cursor.execute(f"""
                SELECT member, SUM(amount) as total
                FROM money_notes
                WHERE {where_clause}
                GROUP BY member
            """, params)
            by_member = {row[0]: row[1] for row in cursor.fetchall()}
            
            # 按类别统计
            cursor.execute(f"""
                SELECT category, SUM(amount) as total
                FROM money_notes
                WHERE {where_clause}
                GROUP BY category
            """, params)
            by_category = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                "total_income": income,
                "total_expense": expense,
                "net": income + expense,
                "by_member": by_member,
                "by_category": by_category
            }
    
    def export_to_csv(self, output_path: str, member: Optional[str] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> int:
        """导出数据到 CSV 文件"""
        records = self.query_records(member=member, start_date=start_date, end_date=end_date, limit=10000)
        
        if not records:
            return 0
        
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['date', 'member', 'category', 'item', 'amount', 'note'])
            writer.writeheader()
            for record in records:
                # 移除不需要的字段
                row = {k: v for k, v in record.items() if k in writer.fieldnames}
                writer.writerow(row)
        
        logger.info(f"导出 {len(records)} 条记录到 {output_path}")
        return len(records)
    
    def check_budget(self, budget: float, start_date: str, end_date: str) -> Tuple[float, bool]:
        """检查预算是否超支"""
        stats = self.get_statistics(start_date=start_date, end_date=end_date)
        expense = abs(stats['total_expense'])
        is_over_budget = expense > budget
        return expense, is_over_budget
    
    def get_all_records(self) -> List[Dict[str, Any]]:
        """获取所有记录（用于调试）"""
        return self.query_records(limit=10000)


# 全局数据库实例
_db_instance: Optional[MoneyNotesDB] = None


def get_db(db_path: str = DEFAULT_DB_PATH) -> MoneyNotesDB:
    """获取数据库单例实例"""
    global _db_instance
    if _db_instance is None:
        _db_instance = MoneyNotesDB(db_path)
    return _db_instance


# 向后兼容：保留 db 全局变量
db = get_db()


def add_income(date: str, member: str, category: str, amount: float, 
               item: str = "", note: str = "") -> str:
    """添加收入记录"""
    db = get_db()
    record_id = db.add_record(date, member, category, abs(amount), item, note)
    return f"✓ 已记录收入：{date} {member} {category} {item} +{amount}元（记录 ID: {record_id}）"


def add_expense(date: str, member: str, category: str, amount: float,
                item: str = "", note: str = "") -> str:
    """添加支出记录"""
    db = get_db()
    record_id = db.add_record(date, member, category, -abs(amount), item, note)
    return f"✓ 已记录支出：{date} {member} {category} {item} -{amount}元（记录 ID: {record_id}）"


def query_expenses(member: str = None, month: str = None) -> str:
    """查询支出记录"""
    db = get_db()
    start_date = end_date = None
    if month:
        start_date = f"{month}-01"
        end_date = f"{month}-31"
    
    records = db.query_records(member=member, start_date=start_date, end_date=end_date)
    
    if not records:
        return "未找到符合条件的记录"
    
    result = f"找到 {len(records)} 条记录：\n"
    for r in records[:20]:  # 最多显示 20 条
        sign = "+" if r['amount'] > 0 else ""
        result += f"  {r['date']} {r['member']} {r['category']} {r['item']} {sign}{r['amount']}元\n"
    
    if len(records) > 20:
        result += f"  ... 还有 {len(records) - 20} 条\n"
    
    return result


def get_monthly_stats(member: str = None, month: str = None) -> str:
    """获取月度统计"""
    db = get_db()
    start_date = end_date = None
    if month:
        start_date = f"{month}-01"
        end_date = f"{month}-31"
    
    stats = db.get_statistics(member=member, start_date=start_date, end_date=end_date)
    
    result = f"📊 统计结果：\n"
    result += f"  总收入：{stats['total_income']:.2f}元\n"
    result += f"  总支出：{abs(stats['total_expense']):.2f}元\n"
    result += f"  结余：{stats['net']:.2f}元\n"
    
    if stats['by_member']:
        result += "\n  按成员：\n"
        for member_name, amount in stats['by_member'].items():
            sign = "+" if amount > 0 else ""
            result += f"    {member_name}: {sign}{amount:.2f}元\n"
    
    if stats['by_category']:
        result += "\n  按类别：\n"
        for category_name, amount in stats['by_category'].items():
            sign = "+" if amount > 0 else ""
            result += f"    {category_name}: {sign}{amount:.2f}元\n"
    
    return result


def delete_expense(item_keyword: str, member: str = None) -> str:
    """删除支出记录（需二次确认）"""
    db = get_db()
    records = db.query_records(member=member, item=item_keyword)
    
    if not records:
        return f"未找到包含'{item_keyword}'的记录"
    
    # 显示待删除记录
    result = f"⚠️ 找到 {len(records)} 条待删除记录：\n"
    for r in records:
        result += f"  ID:{r['id']} {r['date']} {r['member']} {r['category']} {r['item']} {r['amount']}元\n"
    
    result += f"\n请确认是否删除这些记录？回复'确认删除'执行操作。"
    return result


def confirm_delete(item_keyword: str, member: str = None) -> str:
    """确认删除操作"""
    db = get_db()
    records = db.query_records(member=member, item=item_keyword)
    if not records:
        return "未找到记录，删除失败"
    
    ids = [r['id'] for r in records]
    count = db.delete_records(ids)
    return f"✓ 已删除 {count} 条记录"


def export_data(output_path: str, member: str = None, month: str = None) -> str:
    """导出数据到 CSV"""
    db = get_db()
    start_date = end_date = None
    if month:
        start_date = f"{month}-01"
        end_date = f"{month}-31"
    
    count = db.export_to_csv(output_path, member=member, start_date=start_date, end_date=end_date)
    if count == 0:
        return "没有可导出的数据"
    return f"✓ 已导出 {count} 条记录到 {output_path}"


def check_budget(budget: float, month: str = None) -> str:
    """检查预算"""
    db = get_db()
    today = datetime.now()
    if month:
        start_date = f"{month}-01"
        end_date = f"{month}-31"
    else:
        start_date = f"{today.year}-{today.month:02d}-01"
        end_date = f"{today.year}-{today.month:02d}-31"
    
    expense, is_over = db.check_budget(budget, start_date, end_date)
    
    if is_over:
        over_amount = expense - budget
        return f"⚠️ 预算超支！本月已支出 {expense:.2f}元，超出预算 {over_amount:.2f}元"
    else:
        remaining = budget - expense
        return f"✓ 预算正常。本月已支出 {expense:.2f}元，剩余 {remaining:.2f}元"


if __name__ == "__main__":
    # 测试
    print("初始化数据库...")
    test_db = MoneyNotesDB("test_money_notes.db")
    
    print("\n添加测试记录...")
    test_db.add_record("2025-01-15", "女儿", "买书", -50, "三体", "今天买的")
    test_db.add_record("2025-01-15", "妈妈", "收入", 1000, "报销", "7 月 5 日")
    
    print("\n查询所有记录...")
    records = test_db.get_all_records()
    for r in records:
        print(f"  {r['date']} {r['member']} {r['item']} {r['amount']}元")
    
    print("\n获取统计...")
    stats = test_db.get_statistics()
    print(f"  总收入：{stats['total_income']}元")
    print(f"  总支出：{abs(stats['total_expense'])}元")
    
    print("\n测试导出 CSV...")
    count = test_db.export_to_csv("test_export.csv")
    print(f"  导出 {count} 条记录")
    
    print("\n测试完成！")
