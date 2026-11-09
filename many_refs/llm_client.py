"""OpenAI-compatible chat client for the many-references pipeline.

Uses a local vLLM server started by serve_vllm.sh (found through vllm_endpoints/<jobid>.json),
or any remote endpoint given by OPENAI_BASE_URL / OPENAI_API_KEY / LLM_MODEL.
  python many_refs/llm_client.py "Translate into Czech: The city council approved a new regulation."
"""
import glob
import json
import os
import sys

from openai import OpenAI

ENDPOINTS = "/lnet/work/people/jon/predict_split_points/vllm_endpoints"


SECRETS = "/lnet/work/people/jon/predict_split_points/.secrets"
PROVIDERS = {  # name: (base_url, key file in SECRETS)
    "openrouter": ("https://openrouter.ai/api/v1", "openrouter_key"),
    "openai": ("https://api.openai.com/v1", "openai_key"),
}


def connect(model=None, provider=None):
    """Return (client, model name).

    provider "openrouter"/"openai": remote API, key read from .secrets/<provider>_key.
    Otherwise environment overrides (OPENAI_BASE_URL, LLM_MODEL) win, then the newest local
    vLLM server serving `model`."""
    provider = provider or os.environ.get("LLM_PROVIDER")
    if provider in PROVIDERS:
        url, keyfile = PROVIDERS[provider]
        key = open(f"{SECRETS}/{keyfile}").read().strip()
        c = OpenAI(base_url=url, api_key=key, max_retries=5, timeout=900)
        c.provider = provider
        return c, model
    if os.environ.get("OPENAI_BASE_URL"):
        return OpenAI(), model or os.environ["LLM_MODEL"]
    eps = [json.load(open(f)) for f in sorted(glob.glob(f"{ENDPOINTS}/*.json"), key=os.path.getmtime, reverse=True)]
    eps = [e for e in eps if model is None or e["model"] == model]
    if not eps:
        sys.exit(f"no running vLLM server{' for ' + model if model else ''} in {ENDPOINTS}")
    return OpenAI(base_url=eps[0]["base_url"], api_key=eps[0]["api_key"]), eps[0]["model"]


def chat_messages(client, model, messages, think=False, **kw):
    """Multi-turn version of chat(): messages = [{"role": ..., "content": ...}, ...]."""
    if getattr(client, "provider", None):
        kw.pop("temperature", None)
        effort = think if isinstance(think, str) else ("medium" if think else "low")
        kw["extra_body"] = {"reasoning": {"effort": effort}}
    elif think:
        kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
    r = client.chat.completions.create(model=model, messages=messages, **kw)
    return r.choices[0].message.content or ""


def chat(client, model, prompt, think=False, **kw):
    """think: False, True or an effort ("low"/"medium"/"high"); only the answer is returned.

    Local vLLM: chat-template switch. Remote reasoning models: reasoning effort; they take no
    temperature, so it is dropped."""
    if getattr(client, "provider", None):
        kw.pop("temperature", None)
        effort = think if isinstance(think, str) else ("medium" if think else "low")
        kw["extra_body"] = {"reasoning": {"effort": effort}}
    elif think:
        kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
    r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], **kw)
    return r.choices[0].message.content


if __name__ == "__main__":
    client, model = connect()
    print(f"[{model}]")
    print(chat(client, model, sys.argv[1], max_tokens=512, temperature=0.7))
