"""Model names used in the paper are translated to the ids served by the endpoint."""
from llmfca.llm.client import ENDPOINT_MODEL_ALIASES, make_model


def test_paper_names_are_translated_to_endpoint_ids():
    for paper_name, endpoint_id in ENDPOINT_MODEL_ALIASES.items():
        assert make_model(paper_name, "http://localhost/v1", "k").model_name == endpoint_id


def test_other_names_are_unchanged():
    assert make_model("llama3.1:8b", "http://localhost/v1", "k").model_name == "llama3.1:8b"
