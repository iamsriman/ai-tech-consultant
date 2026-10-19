import os, httpx
from typing import Any, Dict, Optional
from .config import ANTHROPIC_MODEL, GROQ_MODEL

class Provider:
    label="unknown"; model_name="unknown"
    def call(self, prompt: str) -> Dict[str, Any]: raise NotImplementedError

class ClaudeProvider(Provider):
    label="claude"
    def __init__(self, model_name: str | None = None): self.api_key=os.getenv("ANTHROPIC_API_KEY","").strip(); self.model_name=(model_name or os.getenv("ANTHROPIC_MODEL", ANTHROPIC_MODEL)).strip()
    def call(self,prompt):
        if not self.api_key: raise RuntimeError("ANTHROPIC_API_KEY is not configured")
        r=httpx.post("https://api.anthropic.com/v1/messages",headers={"x-api-key":self.api_key,"anthropic-version":"2023-06-01","content-type":"application/json"},json={"model":self.model_name,"max_tokens":2048,"messages":[{"role":"user","content":prompt}]},timeout=45)
        r.raise_for_status(); data=r.json(); text="\n".join(x.get("text","") for x in data.get("content",[]) if x.get("type")=="text"); usage=data.get("usage",{})
        return {"text":text,"input_tokens":usage.get("input_tokens"),"output_tokens":usage.get("output_tokens")}

class GroqProvider(Provider):
    label="groq"
    def __init__(self, model_name: str | None = None): self.api_key=os.getenv("GROQ_API_KEY","").strip(); self.model_name=(model_name or os.getenv("GROQ_MODEL", GROQ_MODEL)).strip()
    def call(self,prompt):
        if not self.api_key: raise RuntimeError("GROQ_API_KEY is not configured")
        r=httpx.post("https://api.groq.com/openai/v1/chat/completions",headers={"Authorization":f"Bearer {self.api_key}"},json={"model":self.model_name,"messages":[{"role":"user","content":prompt}],"max_tokens":2048},timeout=45)
        r.raise_for_status(); data=r.json(); usage=data.get("usage",{}); return {"text":data["choices"][0]["message"]["content"],"input_tokens":usage.get("prompt_tokens"),"output_tokens":usage.get("completion_tokens")}

def get_provider(name: Optional[str]=None, model_name: Optional[str]=None):
    selected=(name or os.getenv("TASK3_LLM_PROVIDER", os.getenv("LLM_PROVIDER","claude"))).lower()
    if selected=="claude": return ClaudeProvider(model_name)
    if selected=="groq": return GroqProvider(model_name)
    raise ValueError(f"Unsupported provider: {selected}")
