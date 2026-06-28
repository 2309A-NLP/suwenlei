"""
家庭记账本智能体 - Web 服务器
工单编号：人工智能 NLP-Agent 数字人项目 - 记账本任务
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from agent import BookkeepingAgent
from database_tool import MoneyNotesDB
import json

app = Flask(__name__)
CORS(app)

# 初始化智能体和数据库
agent = BookkeepingAgent()
db = MoneyNotesDB()


@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """聊天接口"""
    data = request.json
    user_input = data.get('message', '').strip()
    
    if not user_input:
        return jsonify({'success': False, 'message': '请输入内容'})
    
    response = agent.chat(user_input)
    return jsonify({'success': True, 'message': response})


@app.route('/api/records', methods=['GET'])
def get_records():
    """获取所有记录"""
    records = db.get_all_records()
    return jsonify({'success': True, 'records': records})


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """获取统计数据"""
    stats = db.get_statistics()
    return jsonify({'success': True, 'stats': stats})


@app.route('/api/delete', methods=['POST'])
def delete_records():
    """删除记录"""
    data = request.json
    ids = data.get('ids', [])
    if not ids:
        return jsonify({'success': False, 'message': '请指定要删除的记录 ID'})
    
    count = db.delete_records(ids)
    return jsonify({'success': True, 'message': f'已删除 {count} 条记录'})


if __name__ == '__main__':
    print("=" * 60)
    print("家庭记账本智能体 Web 服务启动中...")
    print("=" * 60)
    print("访问地址：http://localhost:5000")
    print("按 Ctrl+C 停止服务")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
