"""
experiments/check_endpoint.py — Verificación de que ChatOpenAI funciona con
el endpoint OpenAI-compatible de LlamUS y devuelve logprobs correctamente.

Uso:
    python -m experiments.check_endpoint [--model llama3.2:3b]
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


def check_endpoint(model_name: str = "gemma3:1b", base_url: str = "https://llamus.cs.us.es/ollama/v1") -> int:
    api_key = os.getenv("LLAMUS_API_KEY")
    if not api_key:
        print("ERROR: LLAMUS_API_KEY no está en .env")
        return 2

    print(f"Endpoint: {base_url}")
    print(f"Model:    {model_name}")
    print(f"API key:  {api_key[:6]}...{api_key[-4:]}  ({len(api_key)} chars)")

    try:
        llm = ChatOpenAI(
            base_url=base_url,
            model=model_name,
            api_key=api_key,
            temperature=0.0,
            logprobs=True,
            top_logprobs=20,
            max_tokens=50,
        )
    except Exception as e:
        print(f"ERROR al instanciar ChatOpenAI: {e}")
        return 3

    print("\nLlamando al endpoint...")
    try:
        response = llm.invoke([
            SystemMessage(content="You are a helpful assistant. Respond with valid JSON only."),
            HumanMessage(content='Respond with this exact JSON: {"ok": true, "n": 42}'),
        ])
    except Exception as e:
        print(f"ERROR en invoke(): {e}")
        return 4

    print(f"\nContenido: {response.content!r}")

    metadata = getattr(response, "response_metadata", None)
    if not metadata:
        print("ERROR: response no tiene response_metadata")
        return 5

    print(f"\nresponse_metadata keys: {list(metadata.keys())}")
    logprobs = metadata.get("logprobs")
    if not logprobs:
        print("ERROR: response_metadata no contiene 'logprobs'")
        return 6

    print(f"logprobs keys: {list(logprobs.keys()) if isinstance(logprobs, dict) else type(logprobs)}")
    content_tokens = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not content_tokens:
        print("ERROR: logprobs no contiene 'content' con tokens")
        return 7

    print(f"Número de tokens generados: {len(content_tokens)}")
    first = content_tokens[0]
    if not isinstance(first, dict):
        print(f"ERROR: token entries no son dict (tipo {type(first)})")
        return 8
    print(f"Estructura del primer token: {list(first.keys())}")

    if "token" not in first or "top_logprobs" not in first:
        print(f"ERROR: token entry no tiene 'token' o 'top_logprobs'")
        return 9

    print(f"\nPrimer token: {first['token']!r}")
    print(f"  logprob: {first.get('logprob')}")
    print(f"  top_logprobs: {len(first.get('top_logprobs', []))} alternativas")
    for i, alt in enumerate(first.get("top_logprobs", [])[:5]):
        print(f"    [{i}] {alt.get('token')!r}  logprob={alt.get('logprob')}")

    print("\n✓ ChatOpenAI funciona con LlamUS y devuelve logprobs correctamente.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma3:1b")
    parser.add_argument("--base-url", default="https://llamus.cs.us.es/ollama/v1")
    args = parser.parse_args()
    sys.exit(check_endpoint(args.model, args.base_url))
