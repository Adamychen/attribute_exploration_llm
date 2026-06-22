from typing import Optional, Dict, List, Type
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from llmfca.prompting.schema import get_parser


# ==========================================
# INTEGER CONTEXT
# ==========================================

PREDICATES_INTEGER_FOL = {
    "even(x)": "x is divisible by 2",
    "odd(x)": "x is not divisible by 2",
    "prime(x)": "x has exactly two distinct divisors: 1 and x",
    "compound(x)": "x has at least one divisor other than 1 and itself",
    "quadratic(x)": "x is a perfect square"
}

INTEGER_ATTRIBUTES = ["even", "odd", "prime", "compound", "quadratic"]

# ==========================================
# RCC CONTEXT
# ==========================================

PREDICATES_RCC_FOL = {
    "C(x,y)": "Connected. Regions x and y share at least a single point.",
    "DC(x,y)": "Disconnected. Regions x and y have no points in common.",
    "O(x,y)": "Overlapped. Regions x and y share some interior points, but neither is fully a subset of the other.",
    "P(x,y)": "Part of. Region x is part of y if every point of x is also a point in y.",
    "PP(x,y)": "Proper Part - x is a part of y, but x is not equal to y, this means P(x,y) \\land \\lnot Pi(x,y)",
    "EQ(x,y)": "x and y are equal. Regions x and y are exactly the same region and share all interior points, EQ(x,y) if P(x,y) \\land Pi(x,y), x is part of y and y is part of x.",
    "DR(x,y)": "Discrete From - x and y are not overlapped, (\\lnot O(x,y)).",
    "PO(x,y)": "Partially Overlapped. Regions x and y are partially overlapped if they are overlapped and x is not part of y and y is not part of x. PO(x,y) if O(x,y) \\land \\lnot P(x,y) \\land \\lnot Pi(x,y)",
    "TPP(x,y)": "Tangential Proper Part. Region x is entirely contained within y (x is proper part of y), and their boundaries touch in at least one point.",
    "EC(x,y)": "Externally Connected. Regions x and y are connected but share no interior points (they touch only at their boundaries). EC(x,y) if C(x,y) \\land \\lnot O(x,y)",
    "NTPP(x,y)": "Non-Tangential Proper Part. Region x is entirely contained within y (x is proper part of y), but their boundaries do not touch.",
    "Pi(x,y)": "Part Inverse - The inverse of P(x,y): in this case, y is a part of x.",
    "PPi(x,y)": "Proper Part Inverse - The inverse of PP(x,y): y is a proper part of x.",
    "NTPPi(x,y)": "Non-Tangential Proper Part Inverse - The inverse of NTPP(x,y): y is a non-tangential proper part of x.",
    "TPPi(x,y)": "Tangential Proper Part Inverse - The inverse of TPP(x,y): y is a tangential proper part of x"
}

RCC_ATTRIBUTES = ["C","DC","O","P", "PP", "EQ", "DR", "PO", "TPP", "EC", "NTPP", "Pi", "PPi", "NTPPi", "TPPi"]

# ==========================================
# TOP LEVEL CATEGORIES
# ==========================================

PREDICATES_TOP_FOL = {
    "Independent(x)": "An entity x characterized by some inherent Firstness, independent of any relationships it may have to other entities. If x is an independent entity, it is not necessary that there exists an entity y such that x has y or y has x",
    "Physical(x)": "An entity x that has a location in space-time. Formally, Physical is a primitive that satisfies the following axioms: Anything physical is located in some place and Anything physical occurs at some point in time.",
    "Relative(x)": "An entity in a relationship to some other entity. Formally, for any relative x, there must exist some y such that x has y or y has x.",
    "Abstract(x)": "Pure information as distinguished from any particular encoding of the information in a physical medium. Formally, Abstract is a primitive that satisfies the following axioms: No abstraction as a location in space and No abstraction occurs at a point in time.",
    "Mediating(x)": "An entity characterized by some Thirdness that brings other entities into a relationship. An independent entity need not have any relationship to anything else, a relative entity must have some relationship to something else, and a mediating entity creates a relationship between two other entities. An example of a mediating entity is a marriage, which creates a relationship between a husband and a wife.",
    "Actuality(x)": "A physical entity whose existence is independent of any other entity. As instances, the category Actuality includes both objects and processes. The term is taken from Whitehead, who used it as a synonym for actual entity, which he considered the equivalent of Aristotle's ousia and Descartes's res vera",
    "Form(x)": "Abstract information independent of any encoding or embodiment. Forms can be said to exist in the same sense as mathematical objects such as sets and relations, but instances of forms cannot exist at a particular place and time without some physical encoding or embodiment. Whitehead called them 'eternal objects' because they are independent of space and time.",
    "Prehension(x)": "A physical entity relative to some entity or entities. The has-test is used to check whether an entity x prehends an entity y. If so, the prehension may be expressed has(x,y).",
    "Proposition(x)": "An abstraction that relates some entity or entities. In logic, the assertion of a proposition is a claim that the abstraction corresponds to some aspect or configuration of the entity or entities involved. As an example, the statement cat(Yojo) expresses a proposition that the form labeled Cat characterizes the entity named Yojo. According to Peirce and Whitehead, more complex propositions are asserted by constructing a compound predicate, such as a mathematical expression or a diagram, and using it to characterize the prehensions that relate multiple entities.",
    "Nexus(x)": "A physical entity mediating two or more other entities. Each nexus is a bundle of prehensions, which may be the junctures of an object or the participants of a process. Examples include an arch that consists of junctures of stones or an action that consists of what one participant called an agent is doing to another participant called a patient.",
    "Intention(x)": "Abstraction considered as mediating other entities. Examples of intentions include the hopes, fears, wishes, and purposes that mediate some agent's actions.",
    "Continuant(x)": "An entity whose identity continues to be recognizable over some extended interval of time. Formally, Continuant is a primitive that satisfies the following axioms: - A continuant x has only spatial parts and no temporal parts. At any time t when x exists, all of x exists at the same time t. New parts of a continuant x may be acquired and old parts may be lost, as when a snake sheds its skin. Parts that have been lost may cease to exist, but everything that remains a part of x continues to exist at the same time as x. - The identity conditions for a continuant are independent of time. If c is a subtype of Continuant, then the identity predicate Idc(x,y) for identifying two instances x and y of type c does not depend on time. A physical continuant is an object, and an abstract continuant is a schema that may be used to characterize some object.",
    "Occurrent(x)": "An entity that does not have a stable identity during any interval of time. Formally, Occurrent is a primitive that satisfies the following axioms: - The temporal parts of an occurrent, which are called stages, exist at different times. - The spatial parts of an occurrent, which are called participants, may exist at the same time, but an occurrent may have different participants at different stages. - There are no identity conditions that can be used to identify two occurrents that are observed in nonoverlapping space-time regions. A person's lifetime, for example, is an occurrent. Different stages of a life cannot be reliably identified unless some continuant, such as the person's fingerprints or DNA, is recognized by suitable identity conditions at each stage. Even then, the identification depends on an inference that presupposes the uniqueness of the identity conditions.",
    "Object(x)": "Actuality considered as a continuant, which retains its identity over some interval of time. Although no physical entity is ever permanent, an object can be recognized by identity conditions that remain stable during its lifetime. The type Object includes ordinary physical objects as well as the instantiations of classes in object-oriented programming languages.",
    "Process(x)": "Actuality considered as an occurrent during the interval of interest. Depending on the time scale and level of detail, the same actual entity may be viewed as a stable object or a dynamic process. Even an entity as stable as a diamond could be considered a process when viewed over a long time period or at the atomic level of vibrating particles.",
    "Schema(x)": "A form that has the structure of a continuant. A schema is an abstract form whose structure does not specify time or timelike relationships. Examples include geometric forms, the syntactic structures of sentences in some language, or the encodings of pictures in a multimedia system.",
    "Script(x)": "A form that has the structure of an occurrent. A script is an abstract form that represents time sequences. Examples include computer programs, a recipe for baking a cake, a sheet of music to be played on a piano, or a differential equation that governs the evolution of a physical process. A movie can be described by several different kinds of scripts: the first is a specification of the actions and dialog to be acted out by humans; but the sequence of frames in a reel of film is also a script that determines a process carried out by a projector that generates flickering images on a screen.",
    "Juncture(x)": "A prehension considered as a continuant during some time interval. The prehending entity is an object in a stable relationship to some prehended entity during that interval. An example of a juncture is the relationship between two adjacent stones in an arch. The arch itself is a nexus that both mediates and consists of the multiple junctures.",
    "Participation(x)": "A prehension considered as an occurrent during the interval of interest. The prehending entity is a process, and the prehended entity is called a participant.",
    "Description(x)": "A proposition about a continuant. A description is a proposition that states how some schema characterizes some aspect or configuration of a continuant.",
    "History(x)": "A proposition about an occurrent. A history is a proposition that relates some script to the stages of some occurrent. A computer program, for example, is a script; a computer executing the program is a process; and the abstract information encoded in a trace of the instructions executed is a history. Like any proposition, a history need not be true, and it need not be predicated of the past: a myth is a history of an imaginary past; a prediction is a history of an expected future; and a scenario is a history of some hypothetical occurrent.",
    "Structure(x)": "A nexus considered as a continuant. A structure mediates multiple objects whose junctures constitute the structure.",
    "Situation(x)": "A nexus considered as an occurrent. A situation mediates the participants of some process, whose stages may involve different participants at different times.",
    "Reason(x)": "Intention that has the form of a continuant. Unlike a simple description (Secondness), a reason explains an entity in terms of an intention (Thirdness). For a birthday party, a description might list the presents, but a reason would explain why the presents are relevant to the party.",
    "Purpose(x)": "Intention that has the form of an occurrent. As an example, the words and notes of the song 'Happy Birthday' constitute a script; a description of how people at a party sang the song is history; and the intention that explains the situation is a purpose. The basic axioms for Purpose are inherited from its supertypes Mediating, Abstract, and Occurrent. Lower-level axioms relate purposes to actions and agents: Time sequence. If an agent x performs an act y whose purpose is a situation z, the start of y occurs before the start of z. Contingency. If an agent x performs an act y whose purpose is a situation z described by a proposition p, then it is possible that z might not occur or that p might not be true of z. Success or failure. If an agent x performs an act y whose purpose is a situation z described by a proposition p, then x is said to be successful if z occurs and p is true of z; otherwise, x is said to have failed."
}

ATTRIBUTES_TOP = ["Independent", "Physical", "Relative", "Abstract", "Mediating", "Continuant", "Actuality", "Form", "Prehension", "Proposition", "Nexus", "Intention", "Occurrent", "Object", "Process", "Schema", "Script", "Juncture", "Participation", "Description", "History", "Structure", "Situation", "Reason", "Purpose"]

# ==========================================
# GRAPHS CONTEXT
# ==========================================

PREDICATES_GRAPHS_FOL = {
    "acyclic(g)": "g is an acyclic graph; is a graph that does not contain any cycles.",
    "connected(g)": "g is a graph where there is a path between any two vertices.",
    "2-connected(g)": "g is a connected graph with at least three vertices, where the removal of any single vertex (and its incident edges) does not disconnect the graph.",
    "geodetic(g)": "g is a graph where every pair of vertices has a unique shortest path between them.",
    "bipartite(g)": "g is a graph whose vertices can be divided into two disjoint sets such that every edge connects a vertex in the first set to a vertex in the second set.",
    "nonseparable(g)": "g is a connected graph that remains connected even after removing any single vertex.",
    "planar(g)": "g is a graph that can be drawn on a plane (a flat surface) without any edges crossing each other.",
    "tree(g)": "g is a graph that is connected(g) and acyclic(g)."
}

ATTRIBUTES_GRAPHS = ["acyclic", "connected", "2-connected", "geodetic", "bipartite", "nonseparable", "planar", "tree"]

EVALUATION_RULES_FOL_IMPLICATION = """
Analyze the logical formula's main quantifier and apply the corresponding rule:

1. **CASE: UNIVERSAL QUANTIFIER (∀x ...)**
   - Form: \\forall x (P(x) \\to Q(x))
   - **Goal:** Search for a COUNTEREXAMPLE.
   - **Logic:**
     - A counterexample is an object 'x' where P(x) is TRUE and Q(x) is FALSE.
     - If you find at least one counterexample: The statement is **FALSE**. (Provide the counterexample).
     - If P(x) is never true for any object (vacuously true): The statement is **TRUE**. (No counterexample).
     - If P(x) is true for some objects and Q(x) is always true for those specific objects: The statement is **TRUE**.
     - If antecedent is empty (\\forall x \\to ...), then we are asking if all objects satisfy consequent.

2. **CASE: EXISTENTIAL QUANTIFIER (∃x ...)**
   - Form: \\exists x (P(x) \\land Q(x)) \\land (\\lnot K(x) \\lor \\lnot J(x))
   - **Goal:** Search for a WITNESS (valid example).
   - **Logic:**
     - A witness is an object 'x' where BOTH P(x) and Q(x) are TRUE but K(x) or J(x) are FALSE (at least one).
     - If you find at least one witness: The statement is **TRUE**. (Provide the witness).
     - If you check all objects and none satisfy the condition: The statement is **FALSE**.
"""

# ==========================================
# PROMPT DEFINITIONS
# ==========================================


PROMPT_TEMPLATE = """
Welcome, Assistant!

Context name: {context_name}

Context description: {context_description}

Attributes description:
{predicates_description}

{evaluation_rules}

You must respond with a SINGLE valid JSON object.
Do NOT return the JSON Schema definition.
Do NOT output explanations outside the JSON.

{format_instructions}
"""

PROMPT_TEMPLATE_STEM_BASIS = """
Welcome, Assistant!

Context name: {context_name}

Context description: {context_description}

Attributes description:
{predicates_description}

{evaluation_rules}

You must respond with a SINGLE valid JSON object.
Do NOT return the JSON Schema definition.
Do NOT output explanations outside the JSON.
{format_instructions}

Here you have the sets of rules (pseudo intents) (canonical basis), use these rules as an expert knowledge about the context:
{stem_basis}
"""

PROMPT_TEMPLATE_FULL_CONTEXT = """
Welcome, Assistant!

Context name: {context_name}

Context description: {context_description}

Attributes description:
{predicates_description}

{evaluation_rules}

You must respond with a SINGLE valid JSON object.
Do NOT return the JSON Schema definition.
Do NOT output explanations outside the JSON.
{format_instructions}

Here you have the complete formal context, use this information to answer my questions:
{complete_formal_context}
"""

generic_prompt = PromptTemplate(
    template = PROMPT_TEMPLATE,
    input_variables = [
        "context_name",
        "context_description",
        "predicates_description",
        "evaluation_rules",
        "format_instructions"
    ]
)

generic_prompt_stem_basis = PromptTemplate(
    template=PROMPT_TEMPLATE_STEM_BASIS,
    input_variables = [
        "context_name",
        "context_description",
        "predicates_description",
        "evaluation_rules",
        "format_instructions",
        "stem_basis"
    ]
)

generic_prompt_full_context = PromptTemplate(
    template=PROMPT_TEMPLATE_FULL_CONTEXT,
    input_variables = [
        "context_name",
        "context_description",
        "predicates_description",
        "evaluation_rules",
        "format_instructions",
        "complete_formal_context"
    ]
)

def get_instructions(attributes):
    return get_parser(attributes)

# Integer PROMPTS

INTEGER_PARSER, inst = get_instructions(INTEGER_ATTRIBUTES)
INTEGER_PROMPT = generic_prompt.format(
    context_name = "Integer Numbers",
    context_description = "The context is the Integer Numbers set.",
    predicates_description = PREDICATES_INTEGER_FOL,
    evaluation_rules = EVALUATION_RULES_FOL_IMPLICATION,
    format_instructions = inst
)


INTEGER_PROMPT_STEM_BASIS = generic_prompt_stem_basis.format(
    context_name = "Integer Numbers",
    context_description = "The context is the Integer Numbers set.",
    predicates_description = PREDICATES_INTEGER_FOL,
    evaluation_rules = EVALUATION_RULES_FOL_IMPLICATION,
    format_instructions = inst,
    stem_basis = "1. \\nexists x (prime(x) \\land compound(x))." \
    "2. \\nexists x (prime(x) \\land quadratic(x))." \
    "3. \\nexists x (odd(x) \\land even(x))." \
    "4. \\forall x (even(x) \\land quadratic(x) \\to compound(x))."
)

INTEGER_PROMPT_FULL_CONTEXT = generic_prompt_full_context.format(
    context_name = "Integer Numbers",
    context_description = "The context is the Integer Numbers set.",
    predicates_description = PREDICATES_INTEGER_FOL,
    evaluation_rules = EVALUATION_RULES_FOL_IMPLICATION,
    format_instructions = inst,
    complete_formal_context = "- Objects: {1,2,3,4,6,15,25,9}" \
    "- Attributes: {prime,odd,even,compound,quadratic}" \
    "- Object-Attribute Relations:" \
    "   1: \\lnot  prime(1), odd(1), \\lnot  even(1), \\lnot  compound(1), quadratic(1)." \
    "   2: prime(2), \\lnot  odd(2), even(2), \\lnot compound(2), \\lnot quadratic(2)." \
    "   3: prime(3), odd(3), \\lnot even(3), \\lnot compound(3), \\lnot quadratic(3)." \
    "   4: \\lnot prime(4), \\lnot odd(4), even(4), compound(4), quadratic(4)." \
    "   6: \\lnot prime(6), \\lnot odd(6), even(6), compound(6),\\lnot  quadratic(6)." \
    "   15: \\lnot prime(15), \\lnot even(15), compound(15), odd(15), \\lnot quadratic(15)." \
    "   25: \\lnot prime(25), \\lnot even(25), odd(25), compound(25), quadratic(25)." \
    "   9: \\lnot prime(9), odd(9), \\lnot even(9), compound(9), quadratic(9)."
)




# RCC PROMPTS

rcc_desc = "You are to operate within the framework of Region Connection Calculus (RCC), a formalism for Qualitative Spatial Reasoning (QSR)." \
    "Primitives: The domain consists of spatial regions (non-empty regular closed sets), not points or coordinates." \
    "Topology over Metric: Reasoning is invariant to topological deformation (stretching, twisting). Metric properties (size, precise shape, distance) are irrelevant unless specified."

rcc_stem_basis = """
    1. \\forall x,y DC(x,y) \\to  DR(x,y).
    2. \\forall x,y O(x,y) \\to C(x,y).
    3. \\forall x,y P(x,y) \\to C(x,y) \\land O(x,y).
    4. \\forall x,y PP(x,y) \\to C(x,y) \\land O(x,y) \\land P(x,y).
    5. \\forall x,y EQ(x,y) \\to  C(x,y) \\land O(x,y) \\land P(x,y) \\land Pi(x,y) .
    6. \\forall x,y C(x,y) \\land DR(x,y) \\to  EC(x,y) .
    7. \\forall x,y PO(x,y) \\to  C(x,y) \\land O(x,y) .
    8. \\forall x,y TPP(x,y) \\to  C(x,y) \\land O(x,y) \\land P(x,y) \\land PP(x,y) .
    9. \\forall x,y EC(x,y) \\to  C(x,y) \\land DR(x,y) .
    10. \\forall x,y NTPP(x,y) \\to  C(x,y) \\land O(x,y) \\land P(x,y) \\land PP(x,y) .
    11. \\forall x,y Pi(x,y) \\to  C(x,y) \\land O(x,y) .
    12. \\forall x,y C(x,y) \\land O(x,y) \\land P(x,y) \\land Pi(x,y) \\to  EQ(x,y) .
    13. \\forall x,y PPi(x,y) \\to  C(x,y) \\land O(x,y) \\land Pi(x,y) .
    14. \\forall x,y NTPPi(x,y) \\to  C(x,y) \\land O(x,y) \\land Pi(x,y) \\land PPi(x,y) .
    15. \\forall x,y TPPi(x,y) \\to  C(x,y) \\land O(x,y) \\land Pi(x,y) \\land PPi(x,y) .
    16. \\nexists x,y C(x,y) \\land O(x,y) \\land Pi(x,y) \\land PPi(x,y) \\land NTPPi(x,y) \\land TPPi(x,y) .
    17. \\nexists x,y C(x,y) \\land O(x,y) \\land P(x,y) \\land PO(x,y) .
    18. \\nexists x,y C(x,y) \\land O(x,y) \\land DR(x,y) \\land EC(x,y) .
    19. \\nexists x,y C(x,y) \\land DC(x,y) \\land DR(x,y) \\land EC(x,y) .
    20. \\nexists x,y C(x,y) \\land O(x,y) \\land P(x,y) \\land PP(x,y) \\land TPP(x,y) \\land NTPP(x,y) .
    21. \\nexists x,y C(x,y) \\land O(x,y) \\land PO(x,y) \\land Pi(x,y) .
    22. \\nexists x,y C(x,y) \\land O(x,y) \\land P(x,y) \\land PP(x,y) \\land EQ(x,y) \\land Pi(x,y) .
    23. \\nexists x,y C(x,y) \\land O(x,y) \\land P(x,y) \\land EQ(x,y) \\land Pi(x,y) \\land PPi(x,y) .
"""

rcc_full_context = """
- Objects: {(A,B), (C,D), (E,F), (G,H), (I,J), (K,L), (M,N), (P,Q)}
- Attributes: {C,DC,O,P,PP,EQ,DR,PO,TPP,EC,NTPP,Pi,PPi,NTPPi,TPPi}
- Object-Attribute Relations:
    (A,B): \\lnot C(A,B), DC(A,B), \\lnot O(A,B), \\lnot P(A,B), \\lnot PP(A,B), \\lnot EQ(A,B), DR(A,B), \\lnot PO(A,B), \\lnot TPP(A,B), \\lnot EC(A,B), \\lnot NTPP(A,B), \\lnot Pi(A,B), \\lnot PPi(A,B), \\lnot NTPPi(A,B), \\lnot TPPi(A,B) .
    (C,D): C(C,D), \\lnot DC(C,D), O(C,D), \\lnot P(C,D), \\lnot PP(C,D), \\lnot EQ(C,D), \\lnot DR(C,D), PO(C,D), \\lnot TPP(C,D), \\lnot EC(C,D), \\lnot NTPP(C,D), \\lnot Pi(C,D), \\lnot PPi(C,D), \\lnot NTPPi(C,D), \\lnot TPPi(C,D) . 
    (E,F): C(E,F), \\lnot DC(E,F), \\lnot O(E,F), \\lnot P(E,F), \\lnot PP(E,F), \\lnot EQ(E,F), DR(E,F), \\lnot PO(E,F), \\lnot TPP(E,F), EC(E,F), \\lnot NTPP(E,F), \\lnot Pi(E,F), \\lnot PPi(E,F), \\lnot NTPPi(E,F), \\lnot TPPi(E,F) . 
    (G,H): C(G,H), \\lnot DC(G,H), O(G,H), P(G,H), \\lnot PP(G,H), EQ(G,H), \\lnot DR(G,H), \\lnot PO(G,H), \\lnot TPP(G,H), \\lnot EC(G,H), \\lnot NTPP(G,H), Pi(G,H), \\lnot PPi(G,H), \\lnot NTPPi(G,H), \\lnot TPPi(G,H) . 
    (I,J): C(I,J), \\lnot DC(I,J), O(I,J), P(I,J), PP(I,J), \\lnot EQ(I,J), \\lnot DR(I,J), \\lnot PO(I,J), \\lnot TPP(I,J), \\lnot EC(I,J), NTPP(I,J), \\lnot Pi(I,J), \\lnot PPi(I,J), \\lnot NTPPi(I,J), \\lnot TPPi(I,J) . 
    (K,L): C(K,L), \\lnot DC(K,L), O(K,L), P(K,L), PP(K,L), \\lnot EQ(K,L), \\lnot DR(K,L), \\lnot PO(K,L), TPP(K,L), \\lnot EC(K,L), \\lnot NTPP(K,L), \\lnot Pi(K,L), \\lnot PPi(K,L), \\lnot NTPPi(K,L), \\lnot TPPi(K,L) . 
    (M,N): C(M,N), \\lnot DC(M,N), O(M,N), \\lnot P(M,N), \\lnot PP(M,N), \\lnot EQ(M,N), \\lnot DR(M,N), \\lnot PO(M,N), \\lnot TPP(M,N), \\lnot EC(M,N), \\lnot NTPP(M,N), Pi(M,N), PPi(M,N), \\lnot NTPPi(M,N), TPPi(M,N) . 
    (P,Q): C(P,Q), \\lnot DC(P,Q), O(P,Q), \\lnot P(P,Q), \\lnot PP(P,Q), \\lnot EQ(P,Q), \\lnot DR(P,Q), \\lnot PO(P,Q), \\lnot TPP(P,Q), \\lnot EC(P,Q), \\lnot NTPP(P,Q), Pi(P,Q), PPi(P,Q), NTPPi(P,Q), \\lnot TPPi(G,H) . 

"""

RCC_PARSER, inst = get_instructions(RCC_ATTRIBUTES)
RCC_PROMPT = generic_prompt.format(
    context_name = "Region Connection Calculus Context (RCC)",
    context_description = rcc_desc,
    predicates_description = PREDICATES_RCC_FOL,
    evaluation_rules = EVALUATION_RULES_FOL_IMPLICATION,
    format_instructions = inst
)

RCC_PROMPT_STEM_BASIS = generic_prompt_stem_basis.format(
    context_name = "Region Connection Calculus Context (RCC)",
    context_description = rcc_desc,
    predicates_description = PREDICATES_RCC_FOL,
    evaluation_rules = EVALUATION_RULES_FOL_IMPLICATION,
    format_instructions = inst,
    stem_basis = rcc_stem_basis
)

RCC_PROMPT_FULL_CONTEXT = generic_prompt_full_context.format(
    context_name = "Region Connection Calculus Context (RCC)",
    context_description = rcc_desc,
    predicates_description = PREDICATES_RCC_FOL,
    evaluation_rules = EVALUATION_RULES_FOL_IMPLICATION,
    format_instructions = inst,
    complete_formal_context = rcc_full_context
)

#TOP LEVEL CATEGORIES PROMPTS

top_desc = "This ontology represents the top levels of the KR Ontology (Knowledge Representation), defined by John F. Sowa in his book Knowledge Representation. The system is a synthesis of various philosophical and logical sources, but it is primarily influenced by two major streams of thought: " \
    "   - The Semiotics of Charles Sanders Peirce: From whom Sowa adopts the notions of 'Firstness, Secondness, and Thirdness' (using the terms Independent, Relative, and Mediating). " \
    "   - The Categories of Existence of Alfred North Whitehead: From whom he adopts the distinction between entities that endure through time and dynamic processes."

top_stem_basis = """
\\forall x (Independent(x) \\land Physical(x)) \\to (Actuality(x))
\\forall x (Physical(x) \\land Relative(x)) \\to (Prehension(x))
\\forall x (Independent(x) \\land Abstract(x)) \\to (Form(x))
\\forall x (Relative(x) \\land Abstract(x)) \\to (Proposition(x))
\\forall x (Physical(x) \\land Mediating(x)) \\to (Nexus(x))
\\forall x (Abstract(x) \\land Mediating(x)) \\to (Intention(x))
\\forall x (Actuality(x)) \\to (Independent(x) \\land Physical(x))
\\forall x (Independent(x) \\land Physical(x) \\land Continuant(x) \\land Actuality(x)) \\to (Object(x))
\\forall x (Form(x)) \\to (Independent(x) \\land Abstract(x))
\\forall x (Independent(x) \\land Abstract(x) \\land Continuant(x) \\land Form(x)) \\to (Schema(x))
\\forall x (Prehension(x)) \\to (Physical(x) \\land Relative(x))
\\forall x (Physical(x) \\land Relative(x) \\land Continuant(x) \\land Prehension(x)) \\to (Juncture(x))
\\forall x (Proposition(x)) \\to (Relative(x) \\land Abstract(x))
\\forall x (Relative(x) \\land Abstract(x) \\land Continuant(x) \\land Proposition(x)) \\to (Description(x))
\\forall x (Nexus(x)) \\to (Physical(x) \\land Mediating(x))
\\forall x (Physical(x) \\land Mediating(x) \\land Continuant(x) \\land Nexus(x)) \\to (Structure(x))
\\forall x (Intention(x)) \\to (Abstract(x) \\land Mediating(x))
\\forall x (Abstract(x) \\land Mediating(x) \\land Continuant(x) \\land Intention(x)) \\to (Reason(x))
\\forall x (Independent(x) \\land Physical(x) \\land Actuality(x) \\land Occurrent(x)) \\to (Process(x))
\\forall x (Independent(x) \\land Abstract(x) \\land Form(x) \\land Occurrent(x)) \\to (Script(x))
\\forall x (Physical(x) \\land Relative(x) \\land Prehension(x) \\land Occurrent(x)) \\to (Participation(x))
\\forall x (Relative(x) \\land Abstract(x) \\land Proposition(x) \\land Occurrent(x)) \\to (History(x))
\\forall x (Physical(x) \\land Mediating(x) \\land Nexus(x) \\land Occurrent(x)) \\to (Situation(x))
\\forall x (Object(x)) \\to (Independent(x) \\land Physical(x) \\land Continuant(x) \\land Actuality(x))
\\forall x (Process(x)) \\to (Independent(x) \\land Physical(x) \\land Actuality(x) \\land Occurrent(x))
\\forall x (Schema(x)) \\to (Independent(x) \\land Abstract(x) \\land Continuant(x) \\land Form(x))
\\forall x (Script(x)) \\to (Independent(x) \\land Abstract(x) \\land Form(x) \\land Occurrent(x))
\\forall x (Juncture(x)) \\to (Physical(x) \\land Relative(x) \\land Continuant(x) \\land Prehension(x))
\\forall x (Participation(x)) \\to (Physical(x) \\land Relative(x) \\land Prehension(x) \\land Occurrent(x))
\\forall x (Description(x)) \\to (Relative(x) \\land Abstract(x) \\land Continuant(x) \\land Proposition(x))
\\forall x (History(x)) \\to (Relative(x) \\land Abstract(x) \\land Proposition(x) \\land Occurrent(x))
\\forall x (Structure(x)) \\to (Physical(x) \\land Mediating(x) \\land Continuant(x) \\land Nexus(x))
\\forall x (Situation(x)) \\to (Physical(x) \\land Mediating(x) \\land Nexus(x) \\land Occurrent(x))
\\forall x (Reason(x)) \\to (Abstract(x) \\land Mediating(x) \\land Continuant(x) \\land Intention(x))
\\forall x (Purpose(x)) \\to (Abstract(x) \\land Mediating(x) \\land Intention(x) \\land Occurrent(x))
\\nexists x (Independent(x) \\land Relative(x))
\\nexists x (Physical(x) \\land Abstract(x))
\\nexists x (Independent(x) \\land Mediating(x))
\\nexists x (Relative(x) \\land Mediating(x))
\\nexists x (Continuant(x) \\land Occurrent(x))
\\forall x (Abstract(x) \\land Mediating(x) \\land Intention(x) \\land Occurrent(x)) \\to (Purpose(x))
"""

top_full_context = """
- Objects: {object,process,schema,script,juncture,participation,description,history,structure,situation,reason,purpose}
- Attributes: {Independent,Physical,Relative,Abstract,Mediating,Continuant,Actuality,Form,Prehension,Proposition,Nexus,Intention,Occurrent,Object,Process,Schema,Script,Juncture,Participation,Description,History,Structure,Situation,Reason,Purpose}
- Object-Attribute Relations:
    object: Independent(object), Physical(object), \\lnot Relative(object), \\lnot Abstract(object), \\lnot Mediating(object), Continuant(object), Actuality(object), \\lnot Form(object), \\lnot Prehension(object), \\lnot Proposition(object), \\lnot Nexus(object), \\lnot Intention(object), \\lnot Occurrent(object), Object(object), \\lnot Process(object), \\lnot Schema(object), \\lnot Script(object), \\lnot Juncture(object), \\lnot Participation(object), \\lnot Description(object), \\lnot History(object), \\lnot Structure(object), \\lnot Situation(object), \\lnot Reason(object), \\lnot Purpose(object).
    process: Independent(process), Physical(process), \\lnot Relative(process), \\lnot Abstract(process), \\lnot Mediating(process), \\lnot Continuant(process), Actuality(process), \\lnot Form(process), \\lnot Prehension(process), \\lnot Proposition(process), \\lnot Nexus(process), \\lnot Intention(process), Occurrent(process), \\lnot Object(process), Process(process), \\lnot Schema(process), \\lnot Script(process), \\lnot Juncture(process), \\lnot Participation(process), \\lnot Description(process), \\lnot History(process), \\lnot Structure(process), \\lnot Situation(process), \\lnot Reason(process), \\lnot Purpose(process).
    schema: Independent(schema), \\lnot Physical(schema), \\lnot Relative(schema), Abstract(schema), \\lnot Mediating(schema), Continuant(schema), \\lnot Actuality(schema), Form(schema), \\lnot Prehension(schema), \\lnot Proposition(schema), \\lnot Nexus(schema), \\lnot Intention(schema), \\lnot Occurrent(schema), \\lnot Object(schema), \\lnot Process(schema), Schema(schema), \\lnot Script(schema), \\lnot Juncture(schema), \\lnot Participation(schema), \\lnot Description(schema), \\lnot History(schema), \\lnot Structure(schema), \\lnot Situation(schema), \\lnot Reason(schema), \\lnot Purpose(schema).
    script: Independent(script), \\lnot Physical(script), \\lnot Relative(script), Abstract(script), \\lnot Mediating(script), \\lnot Continuant(script), \\lnot Actuality(script), Form(script), \\lnot Prehension(script), \\lnot Proposition(script), \\lnot Nexus(script), \\lnot Intention(script), Occurrent(script), \\lnot Object(script), \\lnot Process(script), \\lnot Schema(script), Script(script), \\lnot Juncture(script), \\lnot Participation(script), \\lnot Description(script), \\lnot History(script), \\lnot Structure(script), \\lnot Situation(script), \\lnot Reason(script), \\lnot Purpose(script).
    juncture: \\lnot Independent(juncture), Physical(juncture), Relative(juncture), \\lnot Abstract(juncture), \\lnot Mediating(juncture), Continuant(juncture), \\lnot Actuality(juncture), \\lnot Form(juncture), Prehension(juncture), \\lnot Proposition(juncture), \\lnot Nexus(juncture), \\lnot Intention(juncture), \\lnot Occurrent(juncture), \\lnot Object(juncture), \\lnot Process(juncture), \\lnot Schema(juncture), \\lnot Script(juncture), Juncture(juncture), \\lnot Participation(juncture), \\lnot Description(juncture), \\lnot History(juncture), \\lnot Structure(juncture), \\lnot Situation(juncture), \\lnot Reason(juncture), \\lnot Purpose(juncture).
    participation: \\lnot Independent(participation), Physical(participation), Relative(participation), \\lnot Abstract(participation), \\lnot Mediating(participation), \\lnot Continuant(participation), \\lnot Actuality(participation), \\lnot Form(participation), Prehension(participation), \\lnot Proposition(participation), \\lnot Nexus(participation), \\lnot Intention(participation), Occurrent(participation), \\lnot Object(participation), \\lnot Process(participation), \\lnot Schema(participation), \\lnot Script(participation), \\lnot Juncture(participation), Participation(participation), \\lnot Description(participation), \\lnot History(participation), \\lnot Structure(participation), \\lnot Situation(participation), \\lnot Reason(participation), \\lnot Purpose(participation).
    description: \\lnot Independent(description), \\lnot Physical(description), Relative(description), Abstract(description), \\lnot Mediating(description), Continuant(description), \\lnot Actuality(description), \\lnot Form(description), \\lnot Prehension(description), Proposition(description), \\lnot Nexus(description), \\lnot Intention(description), \\lnot Occurrent(description), \\lnot Object(description), \\lnot Process(description), \\lnot Schema(description), \\lnot Script(description), \\lnot Juncture(description), \\lnot Participation(description), Description(description), \\lnot History(description), \\lnot Structure(description), \\lnot Situation(description), \\lnot Reason(description), \\lnot Purpose(description).
    history: \\lnot Independent(history), \\lnot Physical(history), Relative(history), Abstract(history), \\lnot Mediating(history), \\lnot Continuant(history), \\lnot Actuality(history), \\lnot Form(history), \\lnot Prehension(history), Proposition(history), \\lnot Nexus(history), \\lnot Intention(history), Occurrent(history), \\lnot Object(history), \\lnot Process(history), \\lnot Schema(history), \\lnot Script(history), \\lnot Juncture(history), \\lnot Participation(history), \\lnot Description(history), History(history), \\lnot Structure(history), \\lnot Situation(history), \\lnot Reason(history), \\lnot Purpose(history).
    structure: \\lnot Independent(structure), Physical(structure), \\lnot Relative(structure), \\lnot Abstract(structure), Mediating(structure), Continuant(structure), \\lnot Actuality(structure), \\lnot Form(structure), \\lnot Prehension(structure), \\lnot Proposition(structure), Nexus(structure), \\lnot Intention(structure), \\lnot Occurrent(structure), \\lnot Object(structure), \\lnot Process(structure), \\lnot Schema(structure), \\lnot Script(structure), \\lnot Juncture(structure), \\lnot Participation(structure), \\lnot Description(structure), \\lnot History(structure), Structure(structure), \\lnot Situation(structure), \\lnot Reason(structure), \\lnot Purpose(structure).
    situation: \\lnot Independent(situation), Physical(situation), \\lnot Relative(situation), \\lnot Abstract(situation), Mediating(situation), \\lnot Continuant(situation), \\lnot Actuality(situation), \\lnot Form(situation), \\lnot Prehension(situation), \\lnot Proposition(situation), Nexus(situation), \\lnot Intention(situation), Occurrent(situation), \\lnot Object(situation), \\lnot Process(situation), \\lnot Schema(situation), \\lnot Script(situation), \\lnot Juncture(situation), \\lnot Participation(situation), \\lnot Description(situation), \\lnot History(situation), \\lnot Structure(situation), Situation(situation), \\lnot Reason(situation), \\lnot Purpose(situation).
    reason: \\lnot Independent(reason), \\lnot Physical(reason), \\lnot Relative(reason), Abstract(reason), Mediating(reason), Continuant(reason), \\lnot Actuality(reason), \\lnot Form(reason), \\lnot Prehension(reason), \\lnot Proposition(reason), \\lnot Nexus(reason), Intention(reason), \\lnot Occurrent(reason), \\lnot Object(reason), \\lnot Process(reason), \\lnot Schema(reason), \\lnot Script(reason), \\lnot Juncture(reason), \\lnot Participation(reason), \\lnot Description(reason), \\lnot History(reason), \\lnot Structure(reason), \\lnot Situation(reason), Reason(reason), \\lnot Purpose(reason).
    purpose: \\lnot Independent(purpose), \\lnot Physical(purpose), \\lnot Relative(purpose), Abstract(purpose), Mediating(purpose), \\lnot Continuant(purpose), \\lnot Actuality(purpose), \\lnot Form(purpose), \\lnot Prehension(purpose), \\lnot Proposition(purpose), \\lnot Nexus(purpose), Intention(purpose), Occurrent(purpose), \\lnot Object(purpose), \\lnot Process(purpose), \\lnot Schema(purpose), \\lnot Script(purpose), \\lnot Juncture(purpose), \\lnot Participation(purpose), \\lnot Description(purpose), \\lnot History(purpose), \\lnot Structure(purpose), \\lnot Situation(purpose), \\lnot Reason(purpose), Purpose(purpose).
"""
TOP_PARSER, inst = get_instructions(ATTRIBUTES_TOP)
TOP_PROMPT = generic_prompt.format(
    context_name = "Top-Level Categories",
    context_description = top_desc,
    predicates_description = PREDICATES_TOP_FOL,
    evaluation_rules = EVALUATION_RULES_FOL_IMPLICATION,
    format_instructions = inst
)

TOP_PROMPT_STEM_BASIS = generic_prompt_stem_basis.format(
    context_name = "Top-Level Categories",
    context_description = top_desc,
    predicates_description = PREDICATES_TOP_FOL,
    evaluation_rules = EVALUATION_RULES_FOL_IMPLICATION,
    format_instructions = inst,
    stem_basis = top_stem_basis
)

TOP_PROMPT_FULL_CONTEXT = generic_prompt_full_context.format(
    context_name = "Top-Level Categories",
    context_description = top_desc,
    predicates_description = PREDICATES_TOP_FOL,
    evaluation_rules = EVALUATION_RULES_FOL_IMPLICATION,
    format_instructions = inst,
    complete_formal_context = top_full_context
)

# GRAPHS PROMPTS

graphs_stem_basis = """
    1. \\\forall x (acyclic(x) \\to bipartite(x) \\land planar(x)).
    2. \\forall x (2-connected(x) \\to connected(x)).
    3. \\forall x (geodetic(x) \\to connected(x)).
    4. \\forall x (nonseparable(x) \\to connected(x) \\land 2-connected(x)).
    5. \\forall x (tree(x) \\to acyclic(x) \\land connected(x) \\land geodetic(x) \\land bipartite(x) \\land planar(x)).
    6. \\forall x (connected(x) \\land geodetic(x) \\land bipartite(x) \\to acyclic(x) \\land planar(x) \\land tree(x)).
    7. \\forall x (acyclic(x) \\land connected(x) \\land bipartite(x) \\land planar(x) \\to geodetic(x) \\land tree(x)).
    8. \\lnot \\exists x (acyclic(x) \\land connected(x) \\land 2-connected(x) \\land geodetic(x) \\land bipartite(x) \\land planar(x) \\land tree(x)).
"""

graphs_full_context = """
- Objects: {a,c,d,g,b,n,p,t}
- Attributes: {acyclic,connected,2-connected,geodetic,bipartite,nonseparable,planar,tree}
- Object-Attribute Relations:
    a: acyclic(a), connected(a), \\lnot 2-connected(a), geodetic(a), bipartite(a), \\lnot nonseparable(a), planar(a), tree(a).
    c: \\lnot acyclic(c), connected(c), 2-connected(c), \\lnot geodetic(c), bipartite(c), nonseparable(c), planar(c), \\lnot tree(c).
    d: \\lnot acyclic(d), connected(d), 2-connected(d), geodetic(d), \\lnot bipartite(d), nonseparable(d), \\lnot planar(d), \\lnot  tree(d).
    g: acyclic(g), \\lnot connected(g), \\lnot 2-connected(g), \\lnot geodetic(g), bipartite(g), \\lnot nonseparable(g), planar(g), \\lnot tree(g).
    b: \\lnot acyclic(b), connected(b), 2-connected(b), geodetic(b), \\lnot bipartite(b), nonseparable(b), planar(b), \\lnot tree(b).
    n: \\lnot acyclic(n), connected(n), 2-connected(n), \\lnot geodetic(n), bipartite(n), nonseparable(n), \\lnot planar(n), \\lnot tree(n).
    p: \\lnot acyclic(p), connected(p), 2-connected(p), \\lnot geodetic(p), bipartite(p), \\lnot nonseparable(p), planar(p), \\lnot tree(p).
    t: \\lnot acyclic(t), connected(t), 2-connected(t), geodetic(t), \\lnot bipartite(t), \\lnot nonseparable(t), planar(t), \\lnot tree(t).
"""
GRAPHS_PARSER, inst = get_instructions(ATTRIBUTES_GRAPHS)
GRAPHS_PROMPT = generic_prompt.format(
    context_name = "Graph properties",
    context_description = "You are to operate within the framework of Graph Theory.",
    predicates_description = PREDICATES_GRAPHS_FOL,
    evaluation_rules = EVALUATION_RULES_FOL_IMPLICATION,
    format_instructions = inst
)

GRAPHS_PROMPT_STEM_BASIS = generic_prompt_stem_basis.format(
    context_name = "Graph properties",
    context_description = "You are to operate within the framework of Graph Theory.",
    predicates_description = PREDICATES_GRAPHS_FOL,
    evaluation_rules = EVALUATION_RULES_FOL_IMPLICATION,
    format_instructions = inst,
    stem_basis = graphs_stem_basis
)

GRAPHS_PROMPT_FULL_CONTEXT = generic_prompt_full_context.format(
    context_name = "Graph properties",
    context_description = "You are to operate within the framework of Graph Theory.",
    predicates_description = PREDICATES_GRAPHS_FOL,
    evaluation_rules = EVALUATION_RULES_FOL_IMPLICATION,
    format_instructions = inst,
    complete_formal_context = graphs_full_context
)