# -*- coding: utf-8 -*-
"""
LightRAG知识图谱引擎 — 基于图结构的检索增强生成
"""
import os
import json
import logging
import asyncio
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(__file__)
LIGHTRAG_WORKING_DIR = os.path.join(BASE_DIR, 'lightrag_storage_v2')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
# 优先用环境变量，fallback到llm.py里配好的key
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', 'sk-9573...718')
DEEPSEEK_BASE_URL = 'https://api.deepseek.com'

os.makedirs(LIGHTRAG_WORKING_DIR, exist_ok=True)


# ---- DeepSeek LLM ----
def _deepseek_llm_sync(prompt: str = '', system_prompt: str = '') -> str:
    import requests
    from requests.adapters import HTTPAdapter
    msgs = []
    if system_prompt:
        msgs.append({'role': 'system', 'content': system_prompt})
    msgs.append({'role': 'user', 'content': prompt})
    # 直连 DeepSeek，不走系统代理
    proxies = {'http': '', 'https': ''}
    resp = requests.post(
        f'{DEEPSEEK_BASE_URL}/v1/chat/completions',
        headers={'Authorization': f'Bearer {DEEPSEEK_API_KEY}', 'Content-Type': 'application/json'},
        json={'model': 'deepseek-chat', 'messages': msgs, 'temperature': 0.1, 'max_tokens': 4096},
        timeout=120,
        proxies=proxies
    )
    resp.raise_for_status()
    return resp.json()['choices'][0]['message']['content']


async def lightrag_llm_func(prompt: str = '', system_prompt: str = '', **kwargs) -> str:
    return _deepseek_llm_sync(prompt, system_prompt)


# ---- bge-m3嵌入 ----
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        model_path = os.path.join(MODEL_DIR, 'bge-m3')
        try:
            _embedder = SentenceTransformer(model_path, device='cuda')
            logger.info("bge-m3加载完成 (CUDA)")
        except Exception:
            logger.warning("CUDA不可用，回退到CPU")
            _embedder = SentenceTransformer(model_path, device='cpu')
            logger.info("bge-m3加载完成 (CPU)")
    return _embedder


async def _embedding_impl(texts: List[str], **kwargs) -> 'np.ndarray':
    import torch
    import numpy as np
    embedder = _get_embedder()
    with torch.no_grad():
        emb = embedder.encode(texts, batch_size=32, show_progress_bar=False)
    if isinstance(emb, torch.Tensor):
        return emb.cpu().numpy()
    return np.array(emb, dtype=np.float32)


# ---- LightRAG ----
def _create_lightrag_instance():
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    embedding_func = EmbeddingFunc(
        embedding_dim=1024,
        func=_embedding_impl,
        max_token_size=2048,
    )
    return LightRAG(
        working_dir=LIGHTRAG_WORKING_DIR,
        llm_model_func=lightrag_llm_func,
        embedding_func=embedding_func,
        chunk_token_size=3000,
        chunk_overlap_token_size=200,
        llm_model_name='deepseek-chat',
    )


# ---- PDF导入 ----
def _extract_pdf_text(pdf_path: str) -> str:
    import fitz
    doc = fitz.open(pdf_path)
    texts = [page.get_text() for page in doc if page.get_text().strip()]
    doc.close()
    return '\n'.join(texts)


async def query_lightrag(question: str, mode: str = 'hybrid') -> Dict:
    from lightrag.base import QueryParam
    try:
        rag = _create_lightrag_instance()
        await rag.initialize_storages()
        param = QueryParam(mode=mode, hl_keywords=[question], ll_keywords=[''])
        result = await rag.aquery(question, param=param)
        await rag.finalize_storages()
    except Exception as e:
        logger.error(f"LightRAG查询失败: {e}")
        import traceback
        traceback.print_exc()
        result = f"查询失败: {e}"
    return {'question': question, 'mode': mode, 'answer': str(result), 'source': 'LightRAG'}


def export_knowledge_graph() -> Dict:
    try:
        graph_path = os.path.join(LIGHTRAG_WORKING_DIR, 'graph_chunk_entity_relation.graphml')
        if not os.path.exists(graph_path):
            return {'success': False, 'error': 'graphml文件不存在'}
        import networkx as nx
        G = nx.read_graphml(graph_path)
        entities = [n for n, d in G.nodes(data=True) if d.get('entity_type')]
        relations = [(u, v, d) for u, v, d in G.edges(data=True)]
        return {
            'success': True,
            'entities_count': len(entities),
            'relations_count': len(relations),
            'entities': entities[:100],
            'relations': [{'source': u, 'target': v, **d} for u, v, d in relations[:100]],
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    async def main():
        print("构建知识图谱...")
        result = await build_knowledge_graph()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get('success'):
            print("\n知识图谱统计:")
            stat = export_knowledge_graph()
            print(json.dumps(stat, ensure_ascii=False, indent=2))

    asyncio.run(main())
