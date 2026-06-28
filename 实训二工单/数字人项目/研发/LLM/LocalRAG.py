import requests

class LocalRAG:
    def __init__(self, url="http://127.0.0.1:8889/api/ask"):
        self.url = url
        self.prefix_prompt = ''
        self.history = []

    def generate(self, question, system_prompt=""):
        try:
            resp = requests.post(
                self.url,
                json={"question": question, "top_k": 5},
                timeout=120
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("answer", str(data))
        except Exception as e:
            print(f"[LocalRAG] 失败: {e}")
            return "本地RAG请求出错,请检查隧道和RAG服务是否正常。"

    def chat(self, system_prompt, message, history):
        response = self.generate(message, system_prompt)
        history.append((message, response))
        return response, history

    def clear_history(self):
        self.history = []
