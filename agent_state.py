from typing import TypedDict, Callable, Optional, List, Tuple, Dict, Any, Set, Literal

class AgentState(TypedDict, total=False):
    question: str
    logger: Optional[Callable[[str], None]]
    queries: List[str]
    search_results: List[Dict[str, str]]
    links_to_fetch: List[str]
    context: str
    answer: str
    search_engine: str
    sources: List[str]
    used_sources: List[str]
    history: List[dict]

    original_question: str
    analysis: str
    needs_clarification: bool
    clarification_question: str
    user_feedback: str
    troubleshooting_report: str
    clarification_rounds: int

    root_causes: List[str]
    evidence: List[str]
    fix_steps: List[str]
    confidence: float
    commands_to_run: List[str]
    additional_queries: List[str]
    performed_gap_search: bool
    clarification_history: List[str]
    missing_info: List[str]
    last_missing_info: List[str]
    last_confidence: float
    
    plan_action: Literal["search", "clarify", "answer"]
    pre_clarification_question: Optional[str]

    max_queries: int
    skip_search: bool
    search_attempts: int

    context_quality: float
    context_gaps: List[str]
    weighted_context: str

    max_total_searches: int
    max_total_clarifications: int 