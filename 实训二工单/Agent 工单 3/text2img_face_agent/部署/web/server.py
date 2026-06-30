# -*- coding: utf-8 -*-
# 工单编号: 人工智能NLP-Agent数字人项目-文生图智能体任务
"""文生图智能体 Web 前端 (Flask)。

  上传人脸图 -> 调用 FaceViewAgent.run_pipeline -> 返回三视图 + 最终对比图。

运行:
  /d/an/envs/nlp_1/python.exe web/server.py          # 默认 http://127.0.0.1:7860
依赖: flask (已装); 需在项目根目录有 .env 提供 DASHSCOPE_API_KEY, 或已设环境变量。
"""
from __future__ import annotations

import os
import sys
import uuid

# 让 web/ 下的脚本能 import 上层的 text2img_agent 包
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _load_dotenv() -> None:
    """把项目根目录 .env 中的 KEY=VALUE 注入环境变量(不覆盖已有值)。"""
    env_path = os.path.join(ROOT, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

from flask import Flask, request, jsonify, send_from_directory, Response  # noqa: E402

from text2img_agent import FaceViewAgent, AgentConfig  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024   # 单文件上限 32MB

UPLOAD_DIR = os.path.join(ROOT, "web", "_uploads")
OUTPUT_ROOT = os.path.join(ROOT, "outputs_web")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

_ALLOWED = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>文生图智能体 · 面部三视图 + 扩图</title>
<style>
  :root{
    --brand:#6d5efc; --brand2:#22d3ee; --ink:#10131a; --sub:#7a8194;
    --card:rgba(255,255,255,.82); --line:rgba(17,19,26,.08); --ok:#16a34a;
  }
  *{box-sizing:border-box}
  html,body{margin:0;min-height:100%}
  body{
    font-family:"Microsoft YaHei","PingFang SC",system-ui,sans-serif;color:var(--ink);
    background:
      radial-gradient(1100px 600px at 12% -8%,#e8e6ff 0,transparent 55%),
      radial-gradient(900px 520px at 100% 0,#d6f7ff 0,transparent 50%),
      linear-gradient(180deg,#f7f8fc 0,#eef1f8 100%);
    background-attachment:fixed;
  }
  .wrap{max-width:1040px;margin:0 auto;padding:40px 20px 64px}
  header{text-align:center;margin-bottom:28px}
  .badge{display:inline-block;font-size:12px;letter-spacing:.5px;color:var(--brand);
    background:rgba(109,94,252,.10);border:1px solid rgba(109,94,252,.22);
    padding:5px 12px;border-radius:999px;margin-bottom:14px}
  h1{margin:0 0 8px;font-size:30px;font-weight:800;letter-spacing:.5px;
    background:linear-gradient(90deg,var(--brand),var(--brand2));-webkit-background-clip:text;
    background-clip:text;color:transparent}
  header p{margin:0;color:var(--sub);font-size:14px}
  .card{background:var(--card);backdrop-filter:blur(14px);border:1px solid var(--line);
    border-radius:20px;padding:26px;margin-bottom:22px;
    box-shadow:0 18px 50px -22px rgba(40,44,80,.35)}
  #drop{position:relative;border:2px dashed rgba(109,94,252,.35);border-radius:16px;
    padding:40px 20px;text-align:center;cursor:pointer;transition:.18s;background:rgba(255,255,255,.45)}
  #drop:hover{border-color:var(--brand);background:rgba(109,94,252,.06)}
  #drop.hover{border-color:var(--brand);background:rgba(109,94,252,.10);transform:translateY(-1px)}
  .ic{width:54px;height:54px;margin:0 auto 12px;border-radius:14px;display:grid;place-items:center;
    background:linear-gradient(135deg,var(--brand),var(--brand2));color:#fff;font-size:26px;
    box-shadow:0 10px 24px -8px rgba(109,94,252,.6)}
  #drop b{font-size:16px}
  #drop small{display:block;color:var(--sub);margin-top:6px;font-size:12.5px}
  #preview{max-width:230px;max-height:230px;margin:16px auto 0;display:none;border-radius:14px;
    box-shadow:0 10px 30px -12px rgba(0,0,0,.35)}
  .row{display:flex;gap:12px;align-items:center;justify-content:center;margin-top:20px;flex-wrap:wrap}
  button{font-size:15px;font-weight:600;padding:12px 26px;border:none;border-radius:12px;cursor:pointer;transition:.15s}
  .primary{color:#fff;background:linear-gradient(135deg,var(--brand),#8b7bff);
    box-shadow:0 12px 26px -10px rgba(109,94,252,.7)}
  .primary:hover:not(:disabled){transform:translateY(-1px)}
  .primary:disabled{opacity:.5;cursor:not-allowed;box-shadow:none}
  .ghost{background:rgba(17,19,26,.05);color:var(--ink)}
  .ghost:hover{background:rgba(17,19,26,.09)}
  .hint{text-align:center;color:var(--sub);font-size:12.5px;margin-top:12px}
  #status{display:none;margin-top:18px;text-align:center}
  .bar{height:6px;border-radius:999px;background:rgba(109,94,252,.15);overflow:hidden;max-width:420px;margin:0 auto 12px}
  .bar i{display:block;height:100%;width:40%;border-radius:999px;
    background:linear-gradient(90deg,var(--brand),var(--brand2));animation:slide 1.3s ease-in-out infinite}
  @keyframes slide{0%{margin-left:-40%}100%{margin-left:100%}}
  #stepText{color:var(--brand);font-size:14px;font-weight:600}
  .err{display:none;margin-top:14px;text-align:center;color:#d4380d;font-size:13.5px;
    background:rgba(212,56,13,.07);border:1px solid rgba(212,56,13,.2);padding:10px;border-radius:10px}
  #result{display:none;animation:fade .4s ease}
  @keyframes fade{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
  .rhead{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;flex-wrap:wrap;gap:8px}
  .rhead h3{margin:0;font-size:17px}
  .done{font-size:12.5px;color:var(--ok);background:rgba(22,163,74,.10);border:1px solid rgba(22,163,74,.25);
    padding:4px 10px;border-radius:999px}
  #sheet{width:100%;border:1px solid var(--line);border-radius:14px;display:block}
  .views{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:16px}
  .views figure{margin:0;text-align:center}
  .views img{width:100%;border:1px solid var(--line);border-radius:14px;background:#fff;
    transition:.18s;box-shadow:0 8px 22px -16px rgba(0,0,0,.5)}
  .views img:hover{transform:translateY(-3px);box-shadow:0 14px 30px -16px rgba(0,0,0,.5)}
  .views figcaption{font-size:13px;color:var(--sub);margin-top:8px;font-weight:600}
  footer{text-align:center;color:var(--sub);font-size:12px;margin-top:8px}
  a.dl{color:var(--brand);text-decoration:none;font-size:13px;font-weight:600}
  a.dl:hover{text-decoration:underline}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <span class="badge">Agent 数字人项目 · 文生图智能体</span>
    <h1>面部三视图生成 + 扩图</h1>
    <p>上传一张面部照片，自动生成「左转 / 端正 / 右转」三视图并扩图 · 后端：通义 qwen-image-edit</p>
  </header>

  <div class="card">
    <div id="drop">
      <div class="ic">⬆</div>
      <b>点击选择 · 或拖拽图片到此</b>
      <small>支持 PNG / JPG / JPEG / WEBP，建议清晰正面照</small>
      <img id="preview" alt="预览">
    </div>
    <input id="file" type="file" accept="image/*" hidden>
    <div class="row">
      <button id="go" class="primary" disabled>✨ 生成三视图</button>
      <button id="reset" class="ghost">重选</button>
    </div>
    <div class="hint">生成需调用云端模型，约 2–3 分钟，请保持页面打开。</div>
    <div id="status">
      <div class="bar"><i></i></div>
      <div id="stepText">正在生成…</div>
    </div>
    <div id="err" class="err"></div>
  </div>

  <div class="card" id="result">
    <div class="rhead">
      <h3>最终效果对比图</h3>
      <span class="done">✓ 生成完成 <span id="elapsed"></span></span>
    </div>
    <img id="sheet" alt="对比图">
    <div class="views">
      <figure><img id="v_left"><figcaption>面部左转 · <a class="dl" id="d_left" download>下载</a></figcaption></figure>
      <figure><img id="v_front"><figcaption>面部端正 · <a class="dl" id="d_front" download>下载</a></figcaption></figure>
      <figure><img id="v_right"><figcaption>面部右转 · <a class="dl" id="d_right" download>下载</a></figcaption></figure>
    </div>
  </div>

  <footer>工单编号：人工智能NLP-Agent数字人项目-文生图智能体任务</footer>
</div>

<script>
const $ = s => document.querySelector(s);
const drop=$('#drop'), fileInput=$('#file'), preview=$('#preview');
const goBtn=$('#go'), resetBtn=$('#reset');
const statusBox=$('#status'), stepText=$('#stepText'), errBox=$('#err'), result=$('#result');
let picked=null, stepTimer=null;
const STEPS=['对齐人脸…','生成左转视图…','生成端正视图…','生成右转视图…','三视图扩图…','拼合对比图…'];

drop.addEventListener('click', ()=>fileInput.click());
['dragover','dragenter'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('hover');}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('hover');}));
drop.addEventListener('drop',ev=>{if(ev.dataTransfer.files.length)setFile(ev.dataTransfer.files[0]);});
fileInput.addEventListener('change',()=>{if(fileInput.files.length)setFile(fileInput.files[0]);});

function setFile(f){
  if(!f.type.startsWith('image/')){showErr('请选择图片文件');return;}
  picked=f; errBox.style.display='none';
  preview.src=URL.createObjectURL(f); preview.style.display='block';
  goBtn.disabled=false;
}
resetBtn.addEventListener('click',()=>{
  picked=null; fileInput.value=''; preview.style.display='none';
  goBtn.disabled=true; result.style.display='none'; errBox.style.display='none';
});
function showErr(m){errBox.textContent=m; errBox.style.display='block';}

function startSteps(){
  let i=0; stepText.textContent=STEPS[0];
  stepTimer=setInterval(()=>{ i=(i+1)%STEPS.length; stepText.textContent=STEPS[i]; }, 1800);
}
function stopSteps(){ clearInterval(stepTimer); }

goBtn.addEventListener('click', async ()=>{
  if(!picked) return;
  goBtn.disabled=true; resetBtn.disabled=true;
  statusBox.style.display='block'; errBox.style.display='none'; result.style.display='none';
  startSteps();
  const fd=new FormData(); fd.append('image',picked);
  try{
    const r=await fetch('/api/generate',{method:'POST',body:fd});
    const data=await r.json();
    if(!r.ok || !data.ok) throw new Error(data.error || ('HTTP '+r.status));
    const bust='?t='+Date.now();
    $('#sheet').src   = data.contact_sheet+bust;
    $('#v_left').src  = data.views.left+bust;  $('#d_left').href  = data.views.left;
    $('#v_front').src = data.views.front+bust; $('#d_front').href = data.views.front;
    $('#v_right').src = data.views.right+bust; $('#d_right').href = data.views.right;
    $('#elapsed').textContent = data.elapsed ? ('· '+data.elapsed+'s') : '';
    result.style.display='block';
    result.scrollIntoView({behavior:'smooth'});
  }catch(e){ showErr('生成失败：'+e.message); }
  finally{ stopSteps(); statusBox.style.display='none'; goBtn.disabled=false; resetBtn.disabled=false; }
});
</script>
</body>
</html>"""


@app.route("/")
def index() -> Response:
    return Response(PAGE, mimetype="text/html")


@app.route("/api/generate", methods=["POST"])
def generate():
    f = request.files.get("image")
    if f is None or not f.filename:
        return jsonify(ok=False, error="未收到图片文件"), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in _ALLOWED:
        return jsonify(ok=False, error=f"不支持的格式: {ext}"), 400

    job = uuid.uuid4().hex[:12]
    src_path = os.path.join(UPLOAD_DIR, f"{job}{ext}")
    f.save(src_path)

    out_dir = os.path.join(OUTPUT_ROOT, job)
    try:
        agent = FaceViewAgent(AgentConfig())          # 读取 DASHSCOPE_API_KEY
        res = agent.run_pipeline(src_path, out_dir=out_dir)
    except Exception as e:  # noqa: BLE001
        return jsonify(ok=False, error=f"{type(e).__name__}: {e}"), 500

    def url(p: str) -> str:
        return "/outputs/" + job + "/" + os.path.basename(p)

    return jsonify(
        ok=True,
        contact_sheet=url(res.contact_sheet),
        views={k: url(res.outpainted[k]) for k in ("left", "front", "right")},
        elapsed=res.elapsed_sec,
    )


@app.route("/outputs/<job>/<name>")
def serve_output(job: str, name: str) -> Response:
    return send_from_directory(os.path.join(OUTPUT_ROOT, job), name)


if __name__ == "__main__":
    host = os.environ.get("WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("WEB_PORT", "7860"))
    print(f"[Web] 文生图智能体前端: http://{host}:{port}")
    app.run(host=host, port=port, threaded=True)
