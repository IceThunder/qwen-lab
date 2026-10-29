"""Call the local OpenAI-compatible endpoint using only the Python stdlib."""
import argparse
import json
from urllib.request import Request, urlopen

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--prompt", default="请用中文解释什么是神经网络。")
p.add_argument("--max-tokens", type=int, default=128)
p.add_argument("--port", type=int, default=8080)
a = p.parse_args()
body = {"model": "default_model", "messages": [{"role": "user", "content": a.prompt}],
        "temperature": 0, "max_tokens": a.max_tokens, "stream": False}
request = Request(f"http://127.0.0.1:{a.port}/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
with urlopen(request, timeout=180) as response:
    result = json.load(response)
print(json.dumps(result, ensure_ascii=False, indent=2))
