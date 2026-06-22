import math
from typing import List, Dict, Tuple


class EntropyAnalyzer:
    """Token-level entropy over the LLM response, used to size the retry bound.

    The dynamic bound for a single query is
        N_max = min(N_op, ceil(2 ** H*(r)))
    where H* is the joint entropy of semantic tokens (or the mean entropy per
    token if the JSON is malformed).
    """

    def __init__(self, n_op: int = 10):
        self.n_op = n_op

    def calculate_token_entropy(self, top_logprobs: List[Dict]) -> float:
        """
        Calculate Shannon Entropy for a specific token based on its top_logprobs.
        """
        if not top_logprobs:
            return 0.0
            
        # 1. Logprobs to linear probs
        probs = [math.exp(item["logprob"]) for item in top_logprobs]
        
        # 2. Renormalization
        total_mass = sum(probs)
        if total_mass == 0: return 0.0
        
        normalized_probs = [p / total_mass for p in probs]
        
        # 3. Shannon Entropy
        entropy = 0.0
        for p in normalized_probs:
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy

    def analyze_uncertainty(self, logprobs_content: List[Dict]) -> Tuple[float, float]:
        """
        Calculate entropy based on JSON structure                
        """
        semantic_entropies = []
        general_entropies = [] # Para la Mean Entropy (todo lo que no sea sintaxis pura)
        all_entropies = []     # Para depuración si es necesario

        # --- Definición de Estados y Sintaxis ---
        JSON_SYNTAX = {"{", "}", "[", "]", ":", ",", '"', "\n", " "}
        
        current_section = None  
        expecting_value = False
        inside_attributes = False

        for token_data in logprobs_content:
            raw_token = token_data.get("token", "")
            token_str = raw_token.strip().lower()
            top_k = token_data.get("top_logprobs", [])
            
            if not token_str: continue

            # 1. Calcular H_t para el token actual
            h_t = self.calculate_token_entropy(top_k)
            all_entropies.append(h_t)

            # 2. Lógica de Navegación JSON (State Machine)
            
            # Detectar claves principales
            # Nota: Los tokens pueden venir con comillas, ej: '"response"' o 'response'
            clean_key = token_str.replace('"', '')
            
            if clean_key == "response":
                current_section = "response"
                expecting_value = False
            elif clean_key == "attributes":
                current_section = "attributes"
                inside_attributes = True
                expecting_value = False
            
            if ":" in token_str:
                expecting_value = True
                continue 
            
            # Detect object's end
            if "}" in token_str:
                if inside_attributes and current_section == "attributes":
                    inside_attributes = False # Salimos de attributes
                expecting_value = False
            if "," in token_str:
                expecting_value = False

            # 3. Clasificación de Entropía
            
            is_syntax = (token_str in JSON_SYNTAX) or \
                        (all(char in '{}[],:" \n' for char in raw_token))

            if not is_syntax:
                general_entropies.append(h_t)
                
                if expecting_value:
                    if current_section == "response":
                        if token_str in ["yes", "no"]:
                            semantic_entropies.append(h_t)

                    elif inside_attributes:
                        if token_str in ["true", "false"]:
                            semantic_entropies.append(h_t)

        joint_entropy = sum(semantic_entropies)
        mean_entropy = sum(general_entropies) / len(general_entropies) if general_entropies else 0.0

        return joint_entropy, mean_entropy

    def calculate_bound(self, joint_entropy: float, mean_entropy: float, is_json_valid: bool) -> int:
        """Cap retries at min(N_op, ceil(2 ** H*)). Low entropy → fewer retries.

        The bound is operationally capped at N_op = 10, so this function
        serves primarily as a *quantitative* signal (the printed H* and
        perplexity) rather than a discriminator that changes behaviour.
        """
        h_star = joint_entropy if is_json_valid else mean_entropy

        # Evitar overflow o números irracionales
        try:
            perplexity = 2 ** h_star
        except OverflowError:
            perplexity = float('inf')

        n_calc = math.ceil(perplexity)
        n_max = min(getattr(self, 'n_op', 5), n_calc)

        print(f"   [Math] Valid: {is_json_valid} | H*: {h_star:.4f} | PP: {perplexity:.2f} | Bound: {n_max}")
        return n_max