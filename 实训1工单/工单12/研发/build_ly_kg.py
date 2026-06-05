# -*- coding: utf-8 -*-
"""构建力源信息知识图谱"""
import os, sys, nest_asyncio
nest_asyncio.apply()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR_LY = os.path.join(BASE_DIR, 'lightrag_storage_ly')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

sys.path.insert(0, BASE_DIR)
from llm import DEEPSEEK_API_KEY
print(f"API Key: {DEEPSEEK_API_KEY[:10]}...{DEEPSEEK_API_KEY[-4:]}")

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc

async def llm_func(prompt, system_prompt='', **kw):
    import requests
    msgs = []
    if system_prompt:
        msgs.append({'role': 'system', 'content': system_prompt})
    msgs.append({'role': 'user', 'content': prompt})
    proxies = {'http': '', 'https': ''}
    resp = requests.post(
        'https://api.deepseek.com/v1/chat/completions',
        headers={'Authorization': f'Bearer {DEEPSEEK_API_KEY}', 'Content-Type': 'application/json'},
        json={'model': 'deepseek-chat', 'messages': msgs, 'temperature': 0.1, 'max_tokens': 4096},
        timeout=120, proxies=proxies
    )
    resp.raise_for_status()
    return resp.json()['choices'][0]['message']['content']

_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        model_path = os.path.join(MODEL_DIR, 'bge-m3')
        try:
            _embedder = SentenceTransformer(model_path, device='cuda')
        except:
            _embedder = SentenceTransformer(model_path, device='cpu')
            print("bge-m3 CPU")
    return _embedder

async def embed_func(texts, **kw):
    import torch, numpy as np
    emb = get_embedder().encode(texts, batch_size=32, show_progress_bar=False)
    return emb.cpu().numpy() if isinstance(emb, torch.Tensor) else np.array(emb, dtype=np.float32)

# 读取PDF
pdf_path = os.path.join(UPLOAD_DIR, '招股说明书2.pdf')
import fitz
doc = fitz.open(pdf_path)
text = '\n'.join([page.get_text() for page in doc if page.get_text().strip()])
doc.close()
print(f"文本长度: {len(text)}")

# 构建知识图谱
embedding_func = EmbeddingFunc(embedding_dim=1024, func=embed_func, max_token_size=2048)
rag = LightRAG(
    working_dir=STORAGE_DIR_LY,
    llm_model_func=llm_func,
    embedding_func=embedding_func,
    chunk_token_size=3000,
    chunk_overlap_token_size=200,
    llm_model_name='deepseek-chat',
)

import asyncio
asyncio.run(rag.ainsert(text))
print("力源信息知识图谱构建完成!")

# 统计
import networkx as nx
graph_path = os.path.join(STORAGE_DIR_LY, 'graph_chunk_entity_relation.graphml')
if os.path.exists(graph_path):
    G = nx.read_graphml(graph_path)
    entities = [n for n, d in G.nodes(data=True) if d.get('entity_type')]
    relations = [(u, v, d) for u, v, d in G.edges(data=True)]
    print(f"实体数: {len(entities)}, 关系数: {len(relations)}")
