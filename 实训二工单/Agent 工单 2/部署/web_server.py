"""
日程提醒智能体 - Web 服务器
工单编号：人工智能NLP-Agent数字人项目-智能体编排任务

为工单7的 Dify 编排提供 HTTP 接口：与记账本服务同构（POST /api/chat），
Dify 的 HTTP 请求节点统一以 {"message": "..."} 调用、取返回的 message 字段。
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from schedule_db import ScheduleDB
from schedule_agent import ScheduleAgent

app = Flask(__name__)
CORS(app)

# 初始化日程数据库与智能体（纯规则，无需 LLM key）
db = ScheduleDB()
agent = ScheduleAgent(db)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """日程对话接口：增删查日程，返回自然语言结果。"""
    data = request.get_json(force=True, silent=True) or {}
    user_input = (data.get('message') or '').strip()

    if not user_input:
        return jsonify({'success': False, 'message': '请输入内容'})

    response = agent.chat(user_input)
    return jsonify({'success': True, 'message': response})


@app.route('/api/schedules', methods=['GET'])
def list_schedules():
    """获取全部启用中的日程（便于调试核对真实写入）。"""
    return jsonify({'success': True, 'schedules': db.list_all(enabled_only=True)})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'ok': True, 'service': 'schedule', 'port': 5002})


if __name__ == '__main__':
    print("=" * 60)
    print("日程提醒智能体 Web 服务启动中...")
    print("访问地址：http://localhost:5002")
    print("接口：POST /api/chat  {\"message\": \"明天下午3点提醒我开会\"}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5002, debug=False)
