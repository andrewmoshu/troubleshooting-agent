from langgraph.graph import StateGraph, END, START

from agent_state import AgentState
from graph_nodes import (
    plan_initial_step,
    determine_query_strategy,
    reflect_and_rewrite,
    execute_search_queries,
    execute_validate_search_results, # Node that calls util
    fetch_and_aggregate,
    assess_context_value,
    synthesize_answer,
    analyze_findings,
    prepare_clarification_request,
    synthesize_troubleshooting_report
)

# --- Conditional Edge Logic / Helper functions for graphs ---

def decide_next_step(state: AgentState) -> str:
    logger = state.get("logger", lambda msg: print(f"DEBUG: {msg}"))
    search_attempts = state.get("search_attempts", 0)
    max_searches = state.get("max_total_searches", 3)
    clarification_rounds = state.get("clarification_rounds", 0)
    max_clarification_rounds = state.get("max_total_clarifications", 3)
    context_quality = state.get("context_quality", 0.0)
    confidence = state.get("confidence", 0.0)
    performed_gap_search = state.get("performed_gap_search", False)
    additional_queries = state.get("additional_queries", [])
    logger(f"Decision state: search_attempts={search_attempts}/{max_searches}, clarification_rounds={clarification_rounds}/{max_clarification_rounds}, "
           f"context_quality={context_quality:.2f}, confidence={confidence:.2f}, "
           f"performed_gap_search={performed_gap_search}, #additional_queries={len(additional_queries)}, "
           f"needs_clarification_flag={state.get('needs_clarification')}")
    if confidence >= 0.85:
        logger("Decision: High confidence (>=0.85). Synthesizing report.")
        return "synthesize_report"
    if additional_queries and not performed_gap_search and search_attempts <= max_searches:
        logger(f"Decision: Additional queries available and no gap search just performed. Injecting gap queries. (Main search attempt {search_attempts})")
        return "inject_gap_queries"
    if state.get("needs_clarification", False) and \
       state.get("clarification_question", "").strip() and \
       confidence < 0.85 and \
       clarification_rounds < max_clarification_rounds:
        logger(f"Decision: Analysis identified need for clarification (question: \"{state.get('clarification_question')[:70]}...\") and confidence ({confidence:.2f}) is not high. Asking user. (Clarification round {clarification_rounds + 1})")
        return "ask_user"
    if clarification_rounds >= max_clarification_rounds:
        logger(f"Decision: Max clarification rounds ({max_clarification_rounds}) reached. Synthesizing report with current information.")
        return "synthesize_report"
    if search_attempts >= max_searches:
        logger(f"Decision: Max main search attempts ({search_attempts}) reached.")
        if confidence < 0.5 and clarification_rounds < max_clarification_rounds and not state.get("needs_clarification"):
            logger("Confidence very low after max searches, attempting a final user clarification.")
            if not state.get("clarification_question"):
                state["clarification_question"] = "I've completed my searches but my confidence in the solution is low. Can you provide any more specific details, error messages, or logs?"
            state["needs_clarification"] = True
            return "ask_user"
        logger("Proceeding to synthesize report after max main searches.")
        return "synthesize_report"
    # If we failed to fetch any meaningful context, try another search iteration first
    if context_quality == 0.0 and search_attempts < max_searches:
        logger("Decision: No usable context retrieved. Rewriting queries and continuing search before asking the user.")
        return "continue_search_main_cycle"
    if context_quality < 0.4 and search_attempts < max_searches:
        logger("Decision: Low context quality (<0.4). Attempts remaining.")
        if clarification_rounds < max_clarification_rounds and not state.get("needs_clarification"):
            logger("Attempting user clarification due to persistent low context quality.")
            if not state.get("clarification_question"):
                 state["clarification_question"] = "The information I found is not very clear. Could you describe the issue in more detail or provide specific error messages?"
            state["needs_clarification"] = True
            return "ask_user"
        else:
            logger("Cannot improve low context via clarification OR already pending. Moving to next search cycle if available.")
    if confidence < 0.6 and context_quality >= 0.4 and search_attempts < max_searches:
        logger("Decision: Confidence low (<0.6) despite decent context (>=0.4). Attempts remaining.")
        if clarification_rounds < max_clarification_rounds and not state.get("needs_clarification"):
            logger("Attempting user clarification to boost low confidence.")
            if not state.get("missing_info") and not state.get("clarification_question"):
                 state["clarification_question"] = "I have some information, but I'm not fully confident. Can you provide more specific details or logs related to your issue?"
            state["needs_clarification"] = True
            return "ask_user"
        else:
            logger("Cannot improve low confidence via clarification OR already pending. Moving to next search cycle if available.")
    if search_attempts < max_searches:
        logger(f"Decision: Default path. Confidence ({confidence:.2f}) not yet optimal. Proceeding to next main search cycle (current main attempts done: {search_attempts}, max: {max_searches}).")
        return "continue_search_main_cycle"
    else:
        logger(f"Decision: Fallback. Max main search attempts ({search_attempts}) reached. Synthesizing report.")
        return "synthesize_report"

# Standard Workflow
workflow_graph = StateGraph(AgentState)
workflow_graph.add_node("plan", plan_initial_step)
workflow_graph.add_node("determine_strategy", determine_query_strategy)
workflow_graph.add_node("reflect", reflect_and_rewrite)
workflow_graph.add_node("search", execute_search_queries) # Changed from original search to execute_search_queries
workflow_graph.add_node("validate_results", execute_validate_search_results) # Added validation step for standard workflow too
workflow_graph.add_node("aggregate", fetch_and_aggregate)
workflow_graph.add_node("synthesize", synthesize_answer)

def branch_after_plan_standard(state: AgentState) -> str:
    logger = state.get("logger", lambda msg: print(f"DEBUG: {msg}"))
    action = state.get("plan_action", "search")
    logger(f"Standard Workflow branching after plan: {action}")
    if action == "answer":
        return "synthesize"
    elif action == "clarify":
        logger("Clarify chosen in standard mode, but no HITL. Proceeding to search strategy.")
        return "determine_strategy"
    else: # "search"
        return "determine_strategy"

workflow_graph.set_entry_point("plan")
workflow_graph.add_conditional_edges(
    "plan",
    branch_after_plan_standard,
    {
        "determine_strategy": "determine_strategy",
        "synthesize": "synthesize"
    }
)
workflow_graph.add_edge("determine_strategy", "reflect")
workflow_graph.add_edge("reflect", "search")
workflow_graph.add_edge("search", "validate_results") # search -> validate_results
workflow_graph.add_edge("validate_results", "aggregate") # validate_results -> aggregate
workflow_graph.add_edge("aggregate", "synthesize")
workflow_graph.add_edge("synthesize", END)
workflow = workflow_graph.compile()

# Troubleshooting Workflow
troubleshooting_graph = StateGraph(AgentState)
troubleshooting_graph.add_node("plan", plan_initial_step)
troubleshooting_graph.add_node("determine_strategy", determine_query_strategy)
troubleshooting_graph.add_node("reflect", reflect_and_rewrite)
troubleshooting_graph.add_node("prepare_main_search", lambda st: {**st, "search_attempts": st.get("search_attempts", 0) + 1, "performed_gap_search": False})
troubleshooting_graph.add_node("actual_search", execute_search_queries) # Uses the new execute_search_queries
troubleshooting_graph.add_node("validate_results_tr", execute_validate_search_results) # Added validation for troubleshoot
troubleshooting_graph.add_node("aggregate", fetch_and_aggregate)
troubleshooting_graph.add_node("assess_context", assess_context_value)
troubleshooting_graph.add_node("analyze", analyze_findings)
troubleshooting_graph.add_node("ask_user", prepare_clarification_request)
troubleshooting_graph.add_node("inject_gap_queries", lambda st: ({
    **st,
    "queries": st.get("additional_queries", []),
    "performed_gap_search": True,
    "question": st.get("original_question"),
    "additional_queries": []
    }))
troubleshooting_graph.add_node("synthesize_report", synthesize_troubleshooting_report)

def branch_after_plan_tr(state: AgentState) -> str:
    logger = state.get("logger", lambda msg: print(f"DEBUG: {msg}"))
    action = state.get("plan_action", "search")
    logger(f"Troubleshooting Workflow branching after plan: {action}")
    if action == "clarify":
        logger(f"Plan leads to user clarification: {state.get('clarification_question')}")
        return "ask_user"
    elif action == "answer":
        logger("Plan leads to direct answer. Routing to determine_strategy.")
        return "determine_strategy"
    else: # "search"
        logger("Plan leads to search.")
        return "determine_strategy"

troubleshooting_graph.set_entry_point("plan")
troubleshooting_graph.add_conditional_edges(
    "plan",
    branch_after_plan_tr,
    {
        "ask_user": "ask_user",
        "determine_strategy": "determine_strategy",
    }
)
troubleshooting_graph.add_edge("determine_strategy", "reflect")
troubleshooting_graph.add_edge("reflect", "prepare_main_search")
troubleshooting_graph.add_edge("prepare_main_search", "actual_search")
troubleshooting_graph.add_edge("actual_search", "validate_results_tr") # actual_search -> validate_results_tr
troubleshooting_graph.add_edge("validate_results_tr", "aggregate")    # validate_results_tr -> aggregate
troubleshooting_graph.add_edge("aggregate", "assess_context")
troubleshooting_graph.add_edge("assess_context", "analyze")
troubleshooting_graph.add_conditional_edges(
    "analyze",
    decide_next_step,
    {
        "ask_user": "ask_user",
        "inject_gap_queries": "inject_gap_queries",
        "synthesize_report": "synthesize_report",
        "continue_search_main_cycle": "determine_strategy"
    }
)
troubleshooting_graph.add_edge("inject_gap_queries", "actual_search") # Gap search also goes to actual_search then validate
troubleshooting_graph.add_edge("synthesize_report", END)

troubleshooting_workflow = troubleshooting_graph.compile(interrupt_after=["ask_user"]) 