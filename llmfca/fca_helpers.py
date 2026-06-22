"""
fca_helpers: utilidades de soporte para Attribute Exploration.

- context_to_dict: serializa un Context a un dict (objects, properties, bool matrix).
- get_message_function: devuelve la función FOL que formatea una Implication
  como string para el LLM, parametrizada por (question_mode, arity).
"""
from itertools import combinations
from typing import Callable, List

from concepts import Context
from llmfca.attribute_explorer import Implication


def context_to_dict(ctx: Context) -> dict:
    return {
        "objects": list(ctx.objects),
        "properties": list(ctx.properties),
        "context": [list(row) for row in ctx.bools],
    }


def _fol_or(arity: int, attrs: List[str]) -> str:
    if arity == 1:
        return " \\lor ".join(f"(\\lnot {x}(x))" for x in attrs)
    if arity == 2:
        return " \\lor ".join(f"(\\lnot {x}(x,y))" for x in attrs)
    raise ValueError(f"Unsupported arity: {arity}")


def _fol_and(arity: int, attrs: List[str]) -> str:
    if arity == 1:
        return " \\land ".join(f"{x}(x)" for x in attrs)
    if arity == 2:
        return " \\land ".join(f"{x}(x,y)" for x in attrs)
    raise ValueError(f"Unsupported arity: {arity}")


def _quantifier(qmode: str, arity: int) -> str:
    q = "forall" if qmode == "implicational" else "exists"
    if arity == 1:
        return f"\\{q} x"
    if arity == 2:
        return f"\\{q} x,y"
    raise ValueError(f"Unsupported arity: {arity}")


def _format_message(imp: Implication, qmode: str, arity: int) -> str:
    """Forma un mensaje FOL para el LLM.

    Implicational: forall x (antecedent -> consequent)
    Existential:   exists x (antecedent and not consequent[items])
    """
    var = _quantifier(qmode, arity)
    consequent_diff = [a for a in imp.consequent if a not in imp.antecedent]
    if not imp.antecedent:
        if qmode == "implicational":
            return f"{var} (\\to {_fol_and(arity, imp.consequent)})?"
        return f"{var} ({_fol_or(arity, consequent_diff)})?"
    ant = _fol_and(arity, imp.antecedent)
    if qmode == "implicational":
        cons = _fol_and(arity, consequent_diff) if consequent_diff else ""
        return f"{var} ({ant} \\to {cons})?"
    cons = _fol_or(arity, consequent_diff) if consequent_diff else ""
    return f"{var} ({ant} \\land {cons})?"


def get_message_function(qmode: str, arity: int) -> Callable[[Implication], str]:
    """Devuelve la función de formateo FOL para (qmode, arity)."""
    if (qmode, arity) not in {("implicational", 1), ("implicational", 2),
                                ("existential", 1), ("existential", 2)}:
        raise ValueError(f"Unsupported (qmode, arity): ({qmode}, {arity})")
    return lambda imp: _format_message(imp, qmode, arity)
