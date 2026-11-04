import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import operator
from typing import List, Dict, Any, Optional, Literal, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from itertools import combinations
import re
import json
from concepts import Context
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from llmfca.entropy_analyzer import EntropyAnalyzer
from llmfca.analysis.metrics import all_attributes_context


class Implication:
    """A pair (antecedent, consequent) representing a candidate attribute implication.

    The id is canonical: sorted-tuple of antecedent sorted-tuple of consequent,
    so that two Implications with the same logical content compare equal.
    """

    def __init__(self, antecedent, consequent):
        self.antecedent = antecedent
        self.consequent = consequent
        self.id = f"{sorted(antecedent)}->{sorted(consequent)}"

    def __str__(self):
        return f"{self.antecedent} => {[x for x in self.consequent if x not in self.antecedent]}"
    
    def __repr__(self):
        return self.__str__()

#ErrorMessage contains the error messages that we send to the model when it generates an invalid response.
class ErrorMessage():

    @staticmethod
    def wrong_len(n):
        return f"\nERROR: Boolean list must have {n} values."
    @staticmethod
    def not_enough_args(imp):
        def and_strings(attrs):
            return " and ".join(attrs)
        return f"\nERROR: Object does not satisfy {and_strings(imp.antecedent)}."
    @staticmethod
    def invalid_format_answer():
        return "\nERROR: Invalid JSON format."
    @staticmethod
    def useless_counterexample(imp):
        return f"\nERROR: Object satisfies consequent, so it is not a counter-example."
    @staticmethod
    def no_counterexample():
        return f"\nERROR: You provided no example."
    @staticmethod
    def no_attributes():
        return f"\nERROR: Missing attributes field."

# --- State Definition ---

class ExplorationState(TypedDict):
    context: Optional[Any]                              #This is the context we are updating
    stem_basis: List[Implication]                       #Store the canonical basis
    attributes: List[str]                               #(same as FCA, set of attrs)
    stage: Literal["initialization", "exploration"]     #Stages indicates if we have to initialize the context or we can add objects directly
    current_implication: Optional[Implication]          #Actual query to send to the model
    llm_response: str
    parsed_response: Dict[str, Any]                     #JSON response
    flags: Dict[str, bool]                              #Flags dictionary with all type of errors encountered in the current_implication
    retries: int
    max_retries: int                                    #Number of max retries after an invalid response, calculated dynamically
    accept_option: str                                  #Due to implicational or existential questions, we need to know which response doesn't need a counterexample
    current_messages : List[str]
    finished: bool
    early_stop:bool
    joint_entropy: float  # For valid case
    mean_entropy: float   # For invalid case
    parser: PydanticOutputParser
    parsing_error_msg: Optional[str]
    wallclock_start: float                              # perf_counter() at run start
    n_iterations: int                                   # pseudo-intent steps processed
    n_retries_total: int                                # cumulative retries across the run
    n_retry_exhausted: int                              # times the retry bound was hit
    iteration_log: Annotated[List[Dict[str, Any]], operator.add]  # per-iteration trace; reducer appends

# --- Nodes ---

class AttributeExplorerGraph:
    """LangGraph state machine for Attribute Exploration with a stochastic oracle.

    The graph walks lexicographically through pseudo-intents of the current
    formal context and asks the LLM to validate each candidate implication.
    On "yes" the implication joins the Stem basis; on "no" the LLM must supply
    a counterexample with a full attribute vector. Recursive validation
    retries handle malformed responses, bounded by an entropy-derived cap.

    Attributes
    ----------
    model : LangChain chat model
        Must expose response.response_metadata['logprobs']['content'].
    message_function : Callable[[Implication], str]
        Formats the current implication into the user prompt (FOL string).
    system_prompt : str
        The system message: domain description + JSON schema + optional
        stem basis / canonical context.
    attributes : List[str]
        The full attribute set of the formal context (the universe M).
    parser : PydanticOutputParser
        Parser for the dynamic ModelResponse schema (see prompting/schema.py).
    """

    def __init__(self, model, message_function, system_prompt, attributes, parser):
        self.model = model
        self.message_function = message_function
        self.system_prompt = system_prompt
        self.attributes = attributes
        self.parser = parser
        self.N_OP = 10
        self.error_funcs = {
           "wrong_len": lambda s, imp: ErrorMessage.wrong_len(len(self.attributes)),
           "not_enough_attrs": lambda s, imp: ErrorMessage.not_enough_args(imp),
           "useless_counterexample": lambda s, imp: ErrorMessage.useless_counterexample(imp),
           "invalid_format_answer": lambda s, imp: ErrorMessage.invalid_format_answer(),
           "no_counterexample": lambda s, imp: ErrorMessage.no_counterexample(),
           "no_attributes": lambda s, imp: ErrorMessage.no_attributes() 
        }
        self.entr_analyzer = EntropyAnalyzer(self.N_OP)

    def next_implication(self,ctx,basis):
        found_imp = None
        attrs = self.attributes
        for tam in range(len(attrs) + 1):
            for subset in combinations(attrs, tam):
                subset_list = list(subset)
                ext = ctx.extension(subset_list)
                inten = ctx.intension(ext)
                imp = Implication(subset_list, list(inten))
                
                if set(imp.antecedent) == set(imp.consequent): continue
                
                is_pseudo = True
                for basis_imp in basis:
                    if set(basis_imp.antecedent).issubset(set(imp.antecedent)):
                        if not set(basis_imp.consequent).issubset(set(imp.antecedent)):
                            is_pseudo = False
                            break
                if not is_pseudo: continue
                
                if any(b.id == imp.id for b in basis): continue
                
                found_imp = imp
                break
            if found_imp: break
        return found_imp

    # NODE 1: Generator
    def find_next_implication(self, state: ExplorationState):
        ctx = state['context']
        basis = state['stem_basis']
        found_imp = None
        if ctx is None:
            found_imp = Implication([], self.attributes)
        else:
            found_imp = self.next_implication(ctx,basis)

        print(f"Next Implication: {found_imp}")
        finished = found_imp is None
        log_entry = {
            "step": state.get("n_iterations", 0) + 1,
            "stage": state.get("stage"),
            "implication": str(found_imp) if found_imp else None,
            "event": "find_next_implication",
            "finished": finished,
        }
        return {
            "current_implication": found_imp,
            "max_retries": -1,
            "retries": 0,
            "flags": {k:False for k in state['flags']},
            "joint_entropy": 0.0,
            "mean_entropy": 0.0,
            "finished": finished,
            "n_iterations": state.get("n_iterations", 0) + 1,
            "iteration_log": [log_entry],
        }


    # NODE 2: Oracle
    # This node builds the corresponding message and appends corresponding errors
    def ask_llm(self, state: ExplorationState):
        imp = state['current_implication']
        msg = self.message_function(imp)
        messages = []

        #If we detect an error, we send the corresponding error message
        active_errors = [k for k, v in state['flags'].items() if v]
        if active_errors:
            print(f"   >>> Retrying ({state['retries']}/{state['max_retries']}) due to: {active_errors}")
            msg=""
            for k in active_errors:
                msg += self.error_funcs[k](state, imp)
            messages = state["current_messages"].copy()
            messages.append(HumanMessage(content=msg))
        else:
            messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=msg)]

        new_flags = {a:False for a,_ in state['flags'].items()}
        joint_ent, mean_ent = 0.0, 0.0
        try:
            response = self.model.invoke(messages)
            messages.append(response)    
            if hasattr(response, 'response_metadata') and 'logprobs' in response.response_metadata:
                
                content = response.response_metadata['logprobs'].get('content', [])
                joint_ent, mean_ent = self.entr_analyzer.analyze_uncertainty(content)
            
        except Exception as e:
            print(f"Error invoking LLM: {e}")
            response = ""
            new_flags["invalid_format_answer"] = True
            return {
                "llm_response": "",
                "flags": {**new_flags, "invalid_format_answer": True},
                "current_messages": messages,
                "joint_entropy": 0.0,
                "mean_entropy": 0.0,
                "parsing_error_msg": str(e),
                "iteration_log": [{
                    "step": state.get("n_iterations", 0),
                    "event": "ask_llm",
                    "error": str(e),
                }],
            }

        return {
            "llm_response": response,
            "flags": new_flags,
            "current_messages": messages,
            "joint_entropy": joint_ent,
            "mean_entropy": mean_ent,
            "iteration_log": [{
                "step": state.get("n_iterations", 0),
                "event": "ask_llm",
                "joint_entropy": joint_ent,
                "mean_entropy": mean_ent,
                "retry_attempt": state.get("retries", 0),
            }],
        }

    # NODE 3: Parser
    def parse_response(self, state: ExplorationState):
        raw = state['llm_response']
        if raw is None or (isinstance(raw, str) and not raw):
            flags = {k: False for k in state['flags']}
            flags["invalid_format_answer"] = True
            return {
                "parsed_response": {},
                "flags": flags,
                "parsing_error_msg": state.get("parsing_error_msg") or "empty LLM response",
            }
        text = raw.content if hasattr(raw, "content") else str(raw)
        flags = state['flags'].copy()

        clean_think_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        clean_text = clean_think_text.replace("```json", "").replace("```", "").strip()
        
        json_match = re.search(r'\{.*\}', clean_text, re.DOTALL)
        if json_match:
            clean_text = json_match.group(0)

        parsed = {}
        error_msg = None

        try:

            try:
                pydantic_obj = self.parser.parse(clean_text)
                parsed = pydantic_obj.model_dump()
            except Exception:
                parsed = json.loads(clean_text)
            
            if "attributes" in parsed and isinstance(parsed["attributes"], dict):
                parsed["attributes"] = {k.lower().replace(" ", ""): v for k, v in parsed["attributes"].items()}
            
            if "response" in parsed:
                parsed["response"] = str(parsed["response"]).lower().strip()

            raw_obj = parsed.get("object")

            if raw_obj is None or raw_obj == "":
                # Si viene null o vacío, generamos un nombre temporal
                parsed["object"] = "null"

            flags["invalid_format_answer"] = False

        except OutputParserException as e:
            #print(f"DEBUG: Parser Error: {e}")
            flags["invalid_format_answer"] = True
            error_msg = str(e)
            
        except Exception as e:
            #print(f"DEBUG: Generic Error: {e}")
            flags["invalid_format_answer"] = True
            error_msg = "Invalid Output Format."

        return {
            "parsed_response": parsed, 
            "flags": flags, 
            "parsing_error_msg": error_msg 
        }
        

    # NODE 4: Validator
    def validate_logic(self, state: ExplorationState):
        flags = state['flags'].copy()
        current_retries = state['retries']
        parsed = state['parsed_response']
        imp = state['current_implication']

        # 1. Check required fields
        if "response" not in parsed:
            flags["invalid_format_answer"] = True

        if "response" in parsed and re.search(fr"({state['accept_option']})", str(parsed["response"]).lower()):
            return {"flags": {k:False for k in flags}} 
        
        if "object" not in parsed:
            flags["no_counterexample"] = True
        if "attributes" not in parsed:
            flags["no_attributes"] = True

        vals = None 
        try:
            # Intentamos crear la lista
            vals = [parsed["attributes"][a.lower()] for a in self.attributes]
            
            if len(vals) != len(self.attributes):
                flags["wrong_len"] = True
                vals = None 
                
        except (KeyError, TypeError):
            flags["wrong_len"] = True
            vals = None 

        if vals is not None and state['stage'] == "exploration":
            antecedent_indices = [self.attributes.index(a) for a in imp.antecedent]
            consequent_indices = [self.attributes.index(c) for c in imp.consequent]

            # Ahora es seguro usar vals[i]
            if not all(vals[i] for i in antecedent_indices):
                flags["not_enough_attrs"] = True

            if all(vals[i] for i in consequent_indices):
                flags["useless_counterexample"] = True

        # ================================================================
        # ROBUST DYNAMIC BOUND STRATEGY (Ec. 5 y 6)
        # ================================================================

        has_error = any(flags.values())
        new_max_retries = state['max_retries']
        n_retries_total = state.get("n_retries_total", 0)
        n_retry_exhausted = state.get("n_retry_exhausted", 0)

        if has_error:
            n_retries_total += 1
            is_valid_json = (not flags.get("invalid_format_answer", False)
                              and not flags.get("no_counterexample", False)
                              and not flags.get("no_attributes", False)
                              and not flags.get("wrong_len", False))
            # Solo calculamos el bound si no se ha fijado ya (-1 indica inicio de intento)
            if state['max_retries'] == -1:
                # Calculamos N_max usando las métricas guardadas en ask_llm
                new_max_retries = self.entr_analyzer.calculate_bound(
                    state['joint_entropy'],
                    state['mean_entropy'],
                    is_valid_json
                )

            early_stop = False
            if state['max_retries'] != -1:
                early_stop = (current_retries + 1) > state["max_retries"]
                if early_stop:
                    n_retry_exhausted += 1

            log_entry = {
                "step": state.get("n_iterations", 0),
                "event": "validate_logic",
                "active_flags": [k for k, v in flags.items() if v],
                "retries_used": current_retries + 1,
                "max_retries": new_max_retries,
                "is_valid_json": is_valid_json,
                "joint_entropy": state.get("joint_entropy", 0.0),
                "mean_entropy": state.get("mean_entropy", 0.0),
            }
            return {
                "flags": flags,
                "retries": current_retries + 1,
                "max_retries": new_max_retries,
                "early_stop": early_stop,
                "n_retries_total": n_retries_total,
                "n_retry_exhausted": n_retry_exhausted,
                "iteration_log": [log_entry],
            }

        return {
            "flags": {k: False for k in flags},
            "iteration_log": [{
                "step": state.get("n_iterations", 0),
                "event": "validate_logic",
                "decision": "accept",
            }],
        }

    # NODE 5: Updater
    def update_state(self, state: ExplorationState):
        parsed = state['parsed_response']
        ctx = state['context']

        if re.search(fr"({state['accept_option']})", parsed["response"]):
            log_entry = {
                "step": state.get("n_iterations", 0),
                "event": "update_state",
                "decision": "accept",
                "implication": str(state.get("current_implication")),
            }
            if state['stage'] == "initialization":
                # The oracle accepted [] -> M: every object has every attribute.
                # The only L-closed set is M, so the exploration ends; the answers
                # define the context with one object that has all attributes.
                return {
                    "context": all_attributes_context(self.attributes),
                    "stem_basis": state['stem_basis'] + [state['current_implication']],
                    "stage": "exploration",
                    "retries": 0,
                    "finished": True,
                    "iteration_log": [log_entry],
                }
            else:
                return {
                    "stem_basis": state['stem_basis'] + [state['current_implication']],
                    "retries": 0,
                    "iteration_log": [log_entry],
                }
        else:
            ex_name = str(parsed["object"])
            attrs_map = parsed["attributes"]
            bool_list = [attrs_map[attr.lower()] for attr in self.attributes]
            while ex_name in self.attributes: ex_name += "_obj"
            log_entry = {
                "step": state.get("n_iterations", 0),
                "event": "update_state",
                "decision": "add_counterexample",
                "object": ex_name,
                "attribute_vector": bool_list,
            }
            if state['stage'] == "initialization":
                return {
                    "context": Context([ex_name], self.attributes, [bool_list]),
                    "stage": "exploration",
                    "retries": 0,
                    "iteration_log": [log_entry],
                }

            objs = [o[0] if isinstance(o, tuple) else o for o in ctx.objects]
            while ex_name in objs: ex_name += "'"

            new_ctx = Context(objs + [ex_name], self.attributes, [list(b) for b in ctx.bools] + [bool_list])
            return {"context": new_ctx, "retries": 0, "iteration_log": [log_entry]}

    # --- Edges ---
    def route_after_validate(self, state: ExplorationState):
        if any(state['flags'].values()):
            # Dynamic Bound Check
            if state['retries'] > state['max_retries']:
                #print(f"!!! Max retries ({state['max_retries']}) reached. Aborting.")
                return END 
            return "ask_llm"
        return "update_state"

    def check_finished(self, state: ExplorationState):
        if state['current_implication'] is None:
            return END
        return "ask_llm"
    
    def check_finished_initial_question(self, state: ExplorationState):
        if state["finished"]:
            return END
        return "find_next_implication"

    # --- Build ---
    def build(self):
        graph = StateGraph(ExplorationState)
        graph.add_node("find_next_implication", self.find_next_implication)
        graph.add_node("ask_llm", self.ask_llm)
        graph.add_node("parse_response", self.parse_response)
        graph.add_node("validate_logic", self.validate_logic)
        graph.add_node("update_state", self.update_state)

        graph.set_entry_point("find_next_implication") # Initialization starts here

        graph.add_conditional_edges("find_next_implication", self.check_finished, {"ask_llm": "ask_llm", END: END})
        graph.add_edge("ask_llm", "parse_response")
        graph.add_edge("parse_response", "validate_logic")
        graph.add_conditional_edges("validate_logic", self.route_after_validate, {"ask_llm": "ask_llm", "update_state": "update_state", END: END})
        graph.add_conditional_edges("update_state", self.check_finished_initial_question, {"find_next_implication": "find_next_implication", END:END})
        
        
        return graph.compile()