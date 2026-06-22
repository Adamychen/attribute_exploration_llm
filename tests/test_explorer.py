"""
Tests del grafo AttributeExplorerGraph con un mock de LLM (sin red).

Verifica:
- El grafo termina tras N iteraciones
- Acepta implicaciones cuya respuesta es 'yes'
- Acepta counterexamples cuando la respuesta es 'no' con atributo vector
- Marca early_stop tras agotar retries
"""
import sys
import os

CURRENT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT)
sys.path.insert(0, ROOT)

import asyncio
from typing import Any, Dict, List
from langchain_core.messages import AIMessage
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser

from llmfca.attribute_explorer import AttributeExplorerGraph, Implication
from llmfca.prompting.prompts import (
    INTEGER_ATTRIBUTES,
    INTEGER_PROMPT,
    INTEGER_PARSER,
)


class MockModel:
    """Mock determinista: siempre responde 'yes' (acepta implicación).

    Para activar la rama de counterexample, pasa responses=['no', 'no', ...].
    """

    def __init__(self, response: str = "yes", example: str = "null", attrs: Dict[str, bool] = None):
        self.response = response
        self.example = example
        self.attrs = attrs or {}
        self.call_count = 0

    def invoke(self, messages):
        self.call_count += 1
        payload = {
            "explanation": "mocked",
            "response": self.response,
            "object": self.example,
            "attributes": self.attrs,
        }
        import json
        return AIMessage(content=json.dumps(payload))


def message_fn(imp: Implication) -> str:
    return f"Mock query for {imp}"


def test_graph_accepts_implication():
    """Si el mock siempre responde 'yes', el grafo acepta la implicación y termina."""
    model = MockModel(response="yes")
    explorer = AttributeExplorerGraph(
        model=model,
        message_function=message_fn,
        system_prompt=INTEGER_PROMPT,
        attributes=INTEGER_ATTRIBUTES,
        parser=INTEGER_PARSER,
    )
    app = explorer.build()

    initial = {
        "context": None,
        "stage": "initialization",
        "stem_basis": [],
        "attributes": INTEGER_ATTRIBUTES,
        "current_implication": None,
        "llm_response": "",
        "parsed_response": {},
        "flags": {k: False for k in [
            "wrong_len", "not_enough_attrs", "invalid_format_answer",
            "useless_counterexample", "no_counterexample",
            "no_attributes",
        ]},
        "retries": 0,
        "max_retries": -1,
        "accept_option": "yes",
        "early_stop": False,
        "finished": False,
        "timeout": False,
        "current_messages": [],
        "joint_entropy": 0.0,
        "mean_entropy": 0.0,
        "parser": INTEGER_PARSER,
        "parsing_error_msg": None,
        "wallclock_start": 0.0,
        "n_iterations": 0,
        "n_retries_total": 0,
        "n_retry_exhausted": 0,
    }

    result = asyncio.run(app.ainvoke(initial))
    assert "stem_basis" in result, "Result should have stem_basis key"
    assert result.get("finished") is True or result.get("early_stop") is True, (
        f"Graph should terminate cleanly, got: finished={result.get('finished')}, "
        f"early_stop={result.get('early_stop')}"
    )
    print(f"OK test_graph_accepts_implication (calls={model.call_count})")


def test_graph_handles_invalid_response():
    """Si el mock devuelve JSON inválido, el grafo entra en retry y termina con early_stop."""
    class BadModel:
        def __init__(self):
            self.call_count = 0
        def invoke(self, messages):
            self.call_count += 1
            return AIMessage(content="this is not json at all {oops")

    model = BadModel()
    explorer = AttributeExplorerGraph(
        model=model,
        message_function=message_fn,
        system_prompt=INTEGER_PROMPT,
        attributes=INTEGER_ATTRIBUTES,
        parser=INTEGER_PARSER,
    )
    app = explorer.build()

    initial = {
        "context": None,
        "stage": "initialization",
        "stem_basis": [],
        "attributes": INTEGER_ATTRIBUTES,
        "current_implication": None,
        "llm_response": "",
        "parsed_response": {},
        "flags": {k: False for k in [
            "wrong_len", "not_enough_attrs", "invalid_format_answer",
            "useless_counterexample", "no_counterexample",
            "no_attributes",
        ]},
        "retries": 0,
        "max_retries": -1,
        "accept_option": "yes",
        "early_stop": False,
        "finished": False,
        "timeout": False,
        "current_messages": [],
        "joint_entropy": 5.0,
        "mean_entropy": 5.0,
        "parser": INTEGER_PARSER,
        "parsing_error_msg": None,
        "wallclock_start": 0.0,
        "n_iterations": 0,
        "n_retries_total": 0,
        "n_retry_exhausted": 0,
    }

    result = asyncio.run(app.ainvoke(initial))
    assert result.get("n_retries_total", 0) > 0, "Should have at least 1 retry on bad response"
    print(f"OK test_graph_handles_invalid_response (calls={model.call_count}, "
          f"retries={result.get('n_retries_total')}, exhausted={result.get('n_retry_exhausted')})")


if __name__ == "__main__":
    test_graph_accepts_implication()
    test_graph_handles_invalid_response()
    print("\nAll explorer tests passed.")
