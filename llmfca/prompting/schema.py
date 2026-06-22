"""
schema.py — Pydantic schema generation for LLM responses.

Builds a dynamic BaseModel that validates the JSON shape returned by the
LLM during Attribute Exploration. The 'attributes' dict is constructed from
the predicate names of the active context.

Used by prompting/prompts.py to instantiate *_PARSER for each domain.
"""
from typing import List, Optional, Tuple, Type

from pydantic import BaseModel, Field, create_model
from langchain_core.output_parsers import PydanticOutputParser


def create_schema(predicate_names: List[str]) -> Type[BaseModel]:
    """Build a Pydantic model with a boolean field per predicate.

    The resulting model has four top-level fields: explanation, response,
    object, attributes (the latter is a nested dict-of-bool).
    """
    attribute_fields = {
        name: (bool, Field(..., description=f"Does the example satisfy {name}?"))
        for name in predicate_names
    }
    DynamicAttributes = create_model("Attributes", **attribute_fields)
    DynamicResponse = create_model(
        "ModelResponse",
        explanation=(str, Field(..., description="Brief reasoning")),
        response=(str, Field(..., description="Must be 'yes' or 'no'")),
        object=(Optional[str], Field(
            description="Provide an example to justify your answer if needed. "
                        "Otherwise, return null."
        )),
        attributes=(DynamicAttributes, Field(
            ..., description="Truth values for EACH predicate defined in the context."
        )),
    )
    return DynamicResponse


def get_parser(attributes: List[str]) -> Tuple[PydanticOutputParser, str]:
    """Return (parser, format_instructions) for a given attribute set."""
    schema = create_schema(attributes)
    parser = PydanticOutputParser(pydantic_object=schema)
    return parser, parser.get_format_instructions()
