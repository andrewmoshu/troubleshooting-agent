import os
import openai
import json
from pathlib import Path
from typing import Callable, List, Dict, Any # Minimal necessary typing for this file

from langchain_openai import ChatOpenAI

# Project-specific imports
from agent_state import AgentState
from pydantic_models import InitialPlan, QueryComplexityScores, ContextEvaluation, RCA # LinkRelevance is used in search_utils
from search_utils import (
    # google_search, # Not directly called by nodes, but by search_queries_util
    search_queries as search_queries_util, # Renaming to avoid confusion if a node is also named search_queries
    validate_search_results,
    extract_main_text,
    summarize_text,
    chunked_summarize
)

# Client for OpenAI API calls within nodes (if not passed through state or already configured globally)
# Ensure API key is loaded, e.g., via load_dotenv() in the main script or direct os.getenv here if needed.
# It's often better to initialize clients once and pass them or use a global instance.
# For now, assuming client used is the one from the main script or ChatOpenAI handles its own key.
# The `client` variable for direct openai.OpenAI calls needs to be available if used.
# The `openai.OpenAI` client instance from the main script:
main_openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Node Functions ---

def plan_initial_step(state: AgentState) -> AgentState:
    logger = state.get("logger", lambda msg: print(f"DEBUG: {msg}"))
    user_feedback_present = bool(state.get("user_feedback"))
    history = state.get("history", [])
    original_question_from_state = state.get("original_question", state.get("question", ""))
    input_for_planner: str

    # --- Early Exit to Avoid Repeated Clarification Loops ---
    # If the agent is resuming after receiving user feedback (i.e., the previous turn
    # asked for clarification and the user has responded), we should typically move
    # forward with a search rather than immediately asking for clarification again.
    # This prevents an infinite loop where the planner keeps choosing the "clarify"
    # action even after the user has provided additional information.
    has_prior_clar_rounds = state.get("clarification_rounds", 0) > 0
    if (user_feedback_present or has_prior_clar_rounds) and not state.get("needs_clarification", False):
        logger("Detected that the agent has already requested clarification and is now proceeding with the troubleshooting search phase.")
        # Leave query generation to the reflect_and_rewrite node; just indicate we intend to search.
        return {
            **state,
            "plan_action": "search",
            # Preserve existing queries if any; they will be regenerated/refined later if empty.
            "queries": state.get("queries", []),
            "clarification_question": state.get("clarification_question", ""),
            "needs_clarification": False
        }

    if user_feedback_present and len(history) >=1 :
        last_user_msg = history[-1].get("content", "") if history else ""
        last_assistant_msg = ""
        if len(history) >=2:
            for i in range(len(history) - 2, -1, -1):
                if history[i].get("role") == "assistant":
                    last_assistant_msg = history[i].get("content", "")
                    break
        situation_summary = (
            f"Original Problem: '{original_question_from_state}'. "
            f"Last Assistant Q: '{last_assistant_msg[:150]}...'. "
            f"Last User A: '{last_user_msg[:150]}...'. "
        )
        logger(f"Planning (user_feedback present). Situation summary: {situation_summary}")
        input_for_planner = f"Original Problem: {original_question_from_state}\nConversation History (last few turns relevant to current decision point):\n"
        for turn in history[-5:]:
            input_for_planner += f"{turn.get('role')}: {turn.get('content')}\n"
        input_for_planner += "\nBased on the LATEST user message in the history, what is the best next step?"
    else:
        input_for_planner = f"User Request: \"{original_question_from_state}\""
        logger(f"Planning (first pass / no user_feedback for this cycle) using: {input_for_planner}")

    logger(f"Full input for planner LLM:\n---\n{input_for_planner}\n---")
    prompt = f"""You are an expert AI assistant determining the next step in a troubleshooting process.

Review the following situation carefully (it might be an initial request or an ongoing conversation with user feedback):
--- --- --- Beginning of Situation --- --- ---
{input_for_planner}
--- --- --- End of Situation --- --- ---

Based *only* on the LATEST user message in the provided situation:
1.  If the user's latest message provides enough new, actionable information to proceed with searching for solutions or causes, set "action": "search".
2.  If the user's latest message is still insufficient, or if more specific details are now critically needed, set "action": "clarify". Formulate a *new, precise, and non-repetitive* clarification question. Avoid re-asking something the user just addressed unless their response was completely off-topic or missed the core of your last question.
3.  If the user's latest message (or the initial request) is clear enough to attempt synthesizing an answer or a report directly, set "action": "answer".

Output a JSON object with the exact schema:
{{
  "action": "search" | "clarify" | "answer",
  "reasoning": "string",
  "queries": ["string"],
  "clarification_question": "string"
}}
"""
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.1, openai_api_key=os.getenv("OPENAI_API_KEY"))
        structured_llm = llm.with_structured_output(InitialPlan, method="json_mode")
        plan_obj: InitialPlan = structured_llm.invoke(prompt)
        action = plan_obj.action
        reasoning = plan_obj.reasoning
        result = {
            **state,
            "plan_action": action,
            "queries": state.get("queries", []),
            "clarification_question": state.get("clarification_question"),
            "needs_clarification": state.get("needs_clarification", False)
        }
        logger(f"Planner LLM decision: {action}. Reasoning: {reasoning}")
        if action == "search":
            result["queries"] = plan_obj.queries if plan_obj.queries else [state.get("original_question", state.get("question"))]
            logger(f"Planned queries: {result['queries']}")
        elif action == "clarify":
            result["clarification_question"] = plan_obj.clarification_question or "Could you please provide more details about your issue?"
            result["needs_clarification"] = True
            logger(f"Planned new clarification: {result['clarification_question']}")
        elif action == "answer":
            logger("Planned to answer/synthesize directly.")
        return result
    except Exception as e:
        logger(f"Error during initial planning: {e}. Defaulting to search using original question.")
        return {**state, "plan_action": "search", "queries": [state.get("original_question", state.get("question"))]}

def determine_query_strategy(state: AgentState) -> AgentState:
    logger = state.get("logger", lambda msg: print(f"DEBUG: {msg}"))
    question = state.get("problem_statement", state["question"])
    context = state.get("context", "")
    if state.get("skip_search"):
        logger("Skipping search as per state.")
        return {**state, "max_queries": 0}
    if len(context) > 1000 and state.get("clarification_rounds", 0) > 0:
        logger("Sufficient context or follow-up clarification, skipping new detailed query strategy.")
        return {**state, "max_queries": state.get("max_queries", 1), "skip_search": True}
    logger("Determining query strategy...")
    prompt = f"""Analyze the following user question:
User Question: "{question}"

Rate the question on the following criteria:
1.  Complexity (1=very simple, 5=very complex, requiring multiple sub-questions)
2.  Specificity (1=very general, 5=very specific, mentioning exact product names, error codes, etc.)
3.  Technical Detail (1=layman terms, 5=highly technical, using jargon)

Output a JSON object with your scores, following this schema:
{{
  "complexity": integer,
  "specificity": integer,
  "technical_detail": integer
}}
"""
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.1, openai_api_key=os.getenv("OPENAI_API_KEY"))
        structured_llm = llm.with_structured_output(QueryComplexityScores, method="json_mode")
        scores_obj: QueryComplexityScores = structured_llm.invoke(prompt)
        complexity = scores_obj.complexity
        specificity = scores_obj.specificity
        logger(f"Query analysis scores - Complexity: {complexity}, Specificity: {specificity}")
        max_q = 1
        if complexity >= 4:
            max_q = 3
        elif complexity >= 2:
            max_q = 2
        search_engine_choice = "google" if specificity >= 3 else "duckduckgo"
        # Respect an explicit empty list from the caller (meaning: perform broad web search).
        # Only fall back to NVIDIA defaults when the `sources` key was absent or set to None.
        sources_in_state = state.get("sources", None)
        current_sources: List[str] = [] if sources_in_state is None else sources_in_state

        if current_sources:  # Only manipulate when user actually provided some sources
            if specificity <= 2 and "docs.nvidia.com" in current_sources:
                expanded_sources = list(set(current_sources + ["forums.developer.nvidia.com", "nvidia.custhelp.com"]))
            else:
                expanded_sources = current_sources
        else:
            # Broad search (no site restriction)
            expanded_sources = []
        logger(f"New query strategy: max_queries={max_q}, engine={search_engine_choice}, sources={expanded_sources}")
        return {
            **state,
            "max_queries": max_q,
            "search_engine": search_engine_choice,
            "sources": expanded_sources,
            "skip_search": False
        }
    except Exception as e:
        logger(f"Error in determine_query_strategy: {e}. Using defaults.")
        return {**state, "max_queries": 1, "search_engine": "duckduckgo", "skip_search": False, "sources": state.get("sources", ["docs.nvidia.com"])}

def reflect_and_rewrite(state: AgentState) -> AgentState:
    logger = state.get("logger", lambda msg: print(f"DEBUG: {msg}"))
    original_question = state.get("original_question", state.get("question", ""))
    history = state.get("history", [])
    user_feedback = state.get("user_feedback", "")
    current_input_for_reflection: str
    if user_feedback:
        last_assistant_q = "previous question I asked"
        if len(history) >= 2 and history[-2].get("role") == "assistant":
            last_assistant_q = history[-2].get("content", "previous question I asked")
            prefixes_to_strip = [
                "🌼 Daisy needs more information:", "**🌼 Daisy needs more information:**",
                "<span style='color:#76B900;font-size:1.2em'><b>🌼 Daisy's Report:</b></span>",
                "<span style='color:#76B900;font-size:1.2em'><b>🌼 Daisy's Answer:</b></span>",
            ]
            for prefix in prefixes_to_strip:
                if last_assistant_q.startswith(prefix):
                    last_assistant_q = last_assistant_q[len(prefix):].strip()
        current_input_for_reflection = (
            f"Original problem: '{original_question}'. "
            f"My last question to user was: '{last_assistant_q}'. "
            f"User's latest answer/clarification to that question: '{user_feedback}'. "
            f"Based on this specific new information from the user, what precise details or solutions should I search for now? Focus on queries that directly address the user's latest answer."
        )
        logger(f"Reflecting on refined context (due to user_feedback) for query generation: '{current_input_for_reflection[:300]}...'")
    else:
        current_input_for_reflection = original_question
        logger(f"Reflecting on initial question (no user_feedback for this cycle) for query generation: '{current_input_for_reflection[:300]}...'")
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    num_queries_to_gen = state.get("max_queries", 1)
    prompt = f"""You are an expert at query generation for information retrieval and troubleshooting NVIDIA products.
Analyze the user's latest message/context provided below, considering the overall conversation history.

Conversation History (if any):
{history_str if len(history_str) < 3000 else history_str[:3000] + "... (history truncated)"}

User's Latest Message/Current Focus for Search Query Generation (this is the most important part for query formulation):
{current_input_for_reflection}

Determine the *intent* of the User's Latest Message/Current Focus:
1. **Troubleshooting:** User is describing a problem, error, or providing details to solve an issue.
2. **Information Seeking:** User is asking a general question or seeking specific information based on prior discussion.

Based on the intent and the User's Latest Message/Current Focus, generate a list of {num_queries_to_gen} optimized search engine queries.
Focus on generating queries that will find information directly relevant to the *User's Latest Message/Current Focus*.
Keep each query concise (ideally 3-10 words) and avoid filler phrases like "details about" or "guidance on".
If fewer than {num_queries_to_gen} distinct, high-quality queries can be formed, generate as many as are appropriate.

*   **If Troubleshooting:** Generate queries focused on finding causes, solutions, diagnostics, or known issues related to the specific product, components, and symptoms mentioned in the *User's Latest Message/Current Focus*.
*   **If Information Seeking (after clarification):** Generate queries that directly target the specifics provided by the user in their latest answer.

Output only the list of queries, each on a new line starting with '- '.
List of queries:
"""
    try:
        response = main_openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
        response_text = response.choices[0].message.content.strip()
        generated_queries_raw = [q.strip('- ').strip() for q in response_text.split('\n') if q.strip().startswith('-') and q.strip('- ').strip()]
        # Post-process for brevity: remove trailing phrases after common fillers and trim length
        generated_queries: List[str] = []
        for q in generated_queries_raw:
            for filler in [" details about", " guidance on", " recommendations for", " how to", " specific troubleshooting steps for"]:
                if filler in q.lower():
                    q = q.split(filler, 1)[0].strip()
            # Remove trailing punctuation
            q = q.rstrip(' .?!')
            # Deduplicate multiple spaces
            q = ' '.join(q.split())
            generated_queries.append(q)
        final_queries = []
        if generated_queries:
            final_queries.extend(generated_queries)
        if not user_feedback and (not final_queries or len(final_queries) < num_queries_to_gen):
            if original_question not in final_queries:
                final_queries.append(original_question)
        final_queries = list(dict.fromkeys(final_queries))[:num_queries_to_gen]
        if not final_queries and original_question:
            final_queries = [original_question]
        logger(f"Generated {len(final_queries)} queries: {final_queries}")
    except Exception as e:
        logger(f"Error during query reflection: {e}. Using original question: {original_question}")
        final_queries = [original_question] if original_question else []
    return {**state, "queries": final_queries}

def execute_search_queries(state: AgentState) -> AgentState:
    logger = state.get("logger", lambda msg: print(f"DEBUG: {msg}"))
    queries = state.get("queries", [state.get("question", "")])
    search_engine = state.get("search_engine", "duckduckgo")
    sources = state.get("sources", [])
    max_queries_to_run = state.get("max_queries", len(queries))
    # perform_site_search can be derived or set, true if sources are present
    perform_site_search = bool(sources) 

    # Call the utility function from search_utils.py
    all_results_info = search_queries_util(
        queries=queries,
        search_engine=search_engine,
        sources=sources,
        logger=logger,
        max_queries_to_run=max_queries_to_run,
        perform_site_search=perform_site_search,
        max_results_per_query=state.get("max_results_per_source", 3) # from original state.get in old search_queries
    )
    return {**state, "search_results": all_results_info}

def execute_validate_search_results(state: AgentState) -> AgentState:
    logger = state.get("logger", lambda msg: print(f"DEBUG: {msg}"))
    search_results_info = state.get("search_results", [])
    original_question = state.get("original_question", state.get("question"))
    history = state.get("history", [])
    # Determine latest_user_input_for_validation based on state
    # The original logic used state.get("is_resuming_after_clarification") which was removed.
    # Now, it should directly use state.get("user_feedback") if it's meant for the current validation context.
    # This implies user_feedback should be set appropriately before this node if it's post-clarification.
    latest_user_input = state.get("user_feedback", "") 

    validated_urls = validate_search_results(
        search_results_info=search_results_info,
        original_question=original_question,
        logger=logger,
        history=history,
        latest_user_input_for_validation=latest_user_input,
        max_results_to_validate=state.get("max_results_to_validate", 10)
    )
    return {**state, "links_to_fetch": validated_urls}

def fetch_and_aggregate(state: AgentState) -> AgentState:
    logger = state.get("logger", lambda msg: print(f"DEBUG: {msg}"))
    links_to_process = state.get("links_to_fetch", [])
    if not links_to_process:
        logger("No validated links found to fetch content from.")
        return {**state, "context": "[No relevant documents found after validation or search yielded no results]", "used_sources": []}

    logger(f"Fetching and extracting content from {len(links_to_process)} validated links...")
    docs = []
    used_sources = []
    max_links_to_process = 10
    links_to_process = links_to_process[:max_links_to_process]
    if len(links_to_process) > max_links_to_process:
        logger(f"Warning: Too many links ({len(links_to_process)}). Processing the first {max_links_to_process}.")

    SUMMARY_CACHE_FILE = Path("summary_cache.json")
    _summary_cache_data: Dict[str, str] = {}
    # Simplified module-level cache handling for fetch_and_aggregate if it becomes a static attribute
    # For now, this replicates the previous logic found in the original file for fetch_and_aggregate.
    # This might be better managed if fetch_and_aggregate was a class or had a more robust cache injection.
    module_cache_attr_name = "_summary_cache_module_level_for_fetch_node"
    if not hasattr(fetch_and_aggregate, module_cache_attr_name): 
        if SUMMARY_CACHE_FILE.exists():
            try:
                setattr(fetch_and_aggregate, module_cache_attr_name, json.loads(SUMMARY_CACHE_FILE.read_text()))
            except Exception:
                setattr(fetch_and_aggregate, module_cache_attr_name, {})
        else:
            setattr(fetch_and_aggregate, module_cache_attr_name, {})
    _summary_cache_data = getattr(fetch_and_aggregate, module_cache_attr_name)

    for i, url in enumerate(links_to_process):
        logger(f"Fetching ({i+1}/{len(links_to_process)}): {url}")
        RAW_MAX_CHARS = 120000
        text = extract_main_text(url, max_chars=RAW_MAX_CHARS, logger=logger)
        SUMMARY_TRIGGER = 8000
        if text and len(text) > SUMMARY_TRIGGER and not text.startswith("["):
            if url in _summary_cache_data:
                logger(f"Using cached summary for {url}.")
                text = _summary_cache_data[url]
            else:
                logger(f"Text from {url} is very large ({len(text)} chars). Performing chunked summarization…")
                focus_question = state.get("original_question", state.get("question", ""))
                text = chunked_summarize(text, logger, focus_question)
                _summary_cache_data[url] = text
        if text and not text.startswith("[Could not") and not text.startswith("[Skipped") and len(text) > 50:
            docs.append(f"Source: {url}\n{text}\n")
            used_sources.append(url)
        elif text:
             logger(f"Skipping content from {url} due to extraction issue or lack of text: {text[:100]}...")
        else:
             logger(f"Skipping content from {url} due to extraction issue (empty text).")
    if not docs:
         logger("Warning: Failed to extract meaningful content from any link.")
         context_msg = "[Could not extract content from provided links or no relevant documents found.]"
         return {**state, "context": context_msg, "used_sources": []}
    logger(f"Aggregated content from {len(docs)} sources.")
    MAX_CONTEXT_CHARS = 35000
    combined = '\n---\n'.join(docs)
    if len(combined) > MAX_CONTEXT_CHARS:
        logger(f"Combined context is large ({len(combined)} chars). Running reduce summarization…")
        focus_question = state.get("original_question", state.get("question", ""))
        combined = chunked_summarize(combined, logger, focus_question)
    try:
        SUMMARY_CACHE_FILE.write_text(json.dumps(_summary_cache_data))
    except Exception:
        logger("Warning: Failed to write summary cache to disk.")
    return {**state, "context": combined, "used_sources": used_sources}

def assess_context_value(state: AgentState) -> AgentState:
    logger = state.get("logger", lambda msg: print(f"DEBUG: {msg}"))
    context = state.get("context", "")
    question = state.get("original_question", state.get("question"))
    if not context or len(context) < 100:
        logger("Context is too short or missing. Setting quality to 0.")
        return {**state, "context_quality": 0.0, "context_gaps": ["No relevant information found or context is too short."]}
    logger("Assessing context value...")
    context_snippet = context[:15000] + "... [truncated]" if len(context) > 15000 else context
    prompt = f"""You are a technical analyst. Evaluate how well the provided context answers the user's question.

User's Question: "{question}"

Provided Context:
---
{context_snippet}
---

Based ONLY on the provided context and question:
1.  Rate the sufficiency of the information to answer the question (from 0.0 for not at all sufficient, to 1.0 for completely sufficient).
2.  List up to 3 specific, key pieces of information that are MISSING from the context, which, if present, would help provide a more complete and accurate answer. If nothing significant is missing, provide an empty list.

Output a JSON object with the following schema:
{{
  "sufficiency_score": float,
  "information_gaps": ["string"] 
}}
"""
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.1, openai_api_key=os.getenv("OPENAI_API_KEY"))
        structured_llm = llm.with_structured_output(ContextEvaluation, method="json_mode")
        eval_obj: ContextEvaluation = structured_llm.invoke(prompt)
        sufficiency = eval_obj.sufficiency_score
        gaps = eval_obj.information_gaps
        logger(f"Context assessment: Sufficiency={sufficiency:.2f}, Gaps={gaps}")
        new_queries_from_gaps = [f"{question} details about {gap_description}" for gap_description in gaps if gaps and sufficiency < 0.75]
        logger(f"Generated {len(new_queries_from_gaps)} new queries from gaps: {new_queries_from_gaps}")
        current_additional_queries = state.get("additional_queries", [])
        updated_additional_queries = list(set(current_additional_queries + new_queries_from_gaps)) if new_queries_from_gaps else current_additional_queries
        return {
            **state,
            "context_quality": sufficiency,
            "context_gaps": gaps,
            "additional_queries": updated_additional_queries
        }
    except Exception as e:
        logger(f"Error during context assessment: {e}. Setting quality to 0.5.")
        return {**state, "context_quality": 0.5, "context_gaps": ["Error during assessment."], "additional_queries": state.get("additional_queries", [])}

def synthesize_answer(state: AgentState) -> AgentState:
    logger = state.get("logger", lambda msg: print(f"DEBUG: {msg}"))
    question = state["question"]
    context = state.get("context", "")
    if not context or context.startswith("[Could not") or context.startswith("[No relevant"):
        logger("Context is missing or indicates failure. Generating response indicating no information found.")
        answer = f"I could not find sufficiently relevant information in the specified sources ({state.get('sources', 'N/A')}) to answer your question: '{question}'. "
        if state.get('search_results') and not state.get('links_to_fetch') and not state.get('used_sources'):
            answer += "I found some initial search results, but none were deemed relevant enough after validation. "
        elif state.get('links_to_fetch') and not state.get('used_sources'):
            answer += "I found some potential links but could not extract content from them. "
        answer += "You might want to try rephrasing your search terms or checking sources directly."
        return {**state, "answer": answer}
    logger("Generating standard answer using OpenAI...")
    prompt = f"""
You are an expert assistant knowledgeable about NVIDIA products and technologies, based *only* on the provided documentation context.
Use *only* the following documentation context (e.g., from {state.get('sources', 'specified sources')}) to answer the user's question accurately and concisely.
Cite the source URL after the sentence or paragraph that uses information from it, like this: (Source: <url>).
If the answer is not clearly stated in the context, explicitly say that the information is not available in the provided documents and suggest checking the official documentation ({', '.join(state.get('sources', ['docs.nvidia.com'])) if state.get('sources') else 'docs.nvidia.com'}). Do not make assumptions or provide information not present in the context.

Context:
{context}

Question: {question}
Answer:
"""
    try:
        response = main_openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.1,
        )
        answer = response.choices[0].message.content.strip()
        logger("Standard answer generated.")
    except Exception as e:
        logger(f"Error during answer synthesis: {e}")
        answer = f"Sorry, I encountered an error while generating the answer. Please try again. (Error: {e})"
    return {**state, "answer": answer}

def analyze_findings(state: AgentState) -> AgentState:
    logger = state.get("logger", lambda msg: print(f"DEBUG: {msg}"))
    logger("Analyzing research findings for troubleshooting...")
    question = state["original_question"]
    context_to_analyze = state.get("weighted_context", state.get("context", ""))
    history = state.get("history", [])
    user_feedback = state.get("user_feedback", "")
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    if user_feedback:
        history_str += f"\n\nUser Clarification Provided:\n{user_feedback}"
    if not context_to_analyze or context_to_analyze.startswith("[Could not") or context_to_analyze.startswith("[No relevant"):
         logger("Context is missing or indicates failure during fetch. Cannot perform analysis.")
         analysis_text = "Analysis Failed: Could not retrieve context from search results."
         return {
             **state,
             "analysis": analysis_text,
             "needs_clarification": False,
             "clarification_question": ""
         }
    prompt = f"""
You are an expert troubleshooting analyst. Using ONLY the inputs below, output a JSON object that follows this exact schema:

{{
  "root_causes": list[str],              # Possible root causes ranked by likelihood
  "evidence": list[str],                 # Snippets or reasoning supporting each cause
  "fix_steps": list[str],               # Concrete steps or commands to attempt
  "commands_to_run": list[str],         # Shell commands the user should execute to gather more evidence
  "missing_info": list[str],            # Specific pieces of information still required
  "additional_queries": list[str],      # 1–3 new search queries that would help close the gaps
  "confidence": float                   # Overall confidence 0–1 that causes/steps are correct
}}

Constraints:
* Use double quotes for JSON keys.
* Do NOT output anything outside the JSON.

--- Inputs ---
# Problem:
{question}

# Conversation history (latest last):
{history_str}

# User feedback:
{user_feedback or "N/A"}

# Retrieved context (docs, forum posts, etc.):
{context_to_analyze}
--- End Inputs ---
"""
    try:
        structured_llm = ChatOpenAI(model="gpt-4o", temperature=0.1, openai_api_key=os.getenv("OPENAI_API_KEY")).with_structured_output(RCA, method="json_mode")
        rca_obj: RCA = structured_llm.invoke(prompt)
        parsed: Dict[str, Any] = rca_obj.dict()
        analysis_text = json.dumps(parsed, indent=2)
        root_causes = parsed.get("root_causes", [])
        fix_steps = parsed.get("fix_steps", [])
        evidence = parsed.get("evidence", [])
        confidence = float(parsed.get("confidence", 0.0))
        commands_to_run = parsed.get("commands_to_run", [])
        missing_info = parsed.get("missing_info", [])
        additional_queries = parsed.get("additional_queries", [])
        needs_clarification = bool(missing_info)
        clarification_question = ""
        if needs_clarification and missing_info:
            numbered = [f"{idx+1}. {item}" for idx, item in enumerate(missing_info)]
            clarification_question = "To help me confirm the diagnosis, please provide:\n" + "\n".join(numbered)
        logger(f"Parsed analysis JSON. Clarification needed: {needs_clarification} | confidence={confidence}")
    except Exception as e:
        logger(f"Error during analysis: {e}")
        analysis_text = f"Analysis Failed: Error during processing ({e})"
        needs_clarification = False
        clarification_question = ""
        root_causes, evidence, fix_steps, commands_to_run, additional_queries, missing_info = [], [], [], [], [], []
        confidence = 0.0
    return {
        **state,
        "analysis": analysis_text,
        "needs_clarification": needs_clarification,
        "clarification_question": clarification_question,
        "root_causes": root_causes,
        "evidence": evidence,
        "fix_steps": fix_steps,
        "confidence": confidence,
        "commands_to_run": commands_to_run,
        "additional_queries": additional_queries,
        "last_user_feedback": user_feedback,
        "performed_gap_search": state.get("performed_gap_search", False),
        "user_feedback": "",
        "missing_info": missing_info,
        "last_missing_info": state.get("missing_info", []),
        "last_confidence": state.get("confidence", 0.0)
    }

def prepare_clarification_request(state: AgentState) -> AgentState:
    logger = state.get("logger", lambda msg: print(f"DEBUG: {msg}"))
    question_to_ask = state.get('clarification_question', "Could you provide more details?")
    history_list = state.get("clarification_history", [])
    if question_to_ask not in history_list:
        history_list.append(question_to_ask)
    state["clarification_history"] = history_list
    logger(f"Preparing to ask user (batch): {question_to_ask}")
    return state

def synthesize_troubleshooting_report(state: AgentState) -> AgentState:
    logger = state.get("logger", lambda msg: print(f"DEBUG: {msg}"))
    logger("Synthesizing troubleshooting report...")
    original_question = state["original_question"]
    analysis = state["analysis"]
    used_sources = state.get("used_sources", [])
    if "Analysis Failed" in analysis:
         logger("Analysis step failed. Synthesizing based on failure.")
         report = f"Could not complete the troubleshooting process for: '{original_question}'.\n"
         report += f"Reason: {analysis}\n"
         if state.get('context', '').startswith("[Could not"):
              report += f"Underlying Issue: Failed to retrieve or process information from sources ({state.get('sources', 'N/A')}).\n"
         report += "Please try rephrasing your query or checking the sources directly."
         return {**state, "troubleshooting_report": report, "clarification_question": None}
    prompt = f"""
You are a technical support specialist creating a troubleshooting report based *only* on the preceding analysis of retrieved documentation.

**Objective:** Consolidate the analysis findings into a clear, actionable report for the user.

**Source Analysis Summary:**
{analysis}

**Instructions:**
1.  Start with a clear statement of the user's original problem.
2.  Summarize the **Potential Causes** identified in the analysis. If none were found, state that.
3.  Summarize the **Potential Solutions/Troubleshooting Steps** identified in the analysis. If none were found, state that.
4.  Mention the **Information Gaps** identified, explaining why this information is needed for a more definitive answer.
5.  If a clarification question was asked but not answered (e.g., if this is the first pass and clarification was needed), reiterate the question from the analysis.
6.  Conclude with recommendations, which might include: executing the suggested steps, providing the missing information, or consulting official NVIDIA support channels if the documentation was insufficient.
7.  Reference the specific source URLs used ({used_sources}) when mentioning information derived from them, if possible based on the analysis structure (otherwise list them at the end).

**Do not add information not present in the 'Source Analysis Summary' above.**

**Troubleshooting Report:**
"""
    try:
        response = main_openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.2,
        )
        report = response.choices[0].message.content.strip()
        if used_sources and not all(src in report for src in used_sources):
             report += "\n\n**Sources Consulted:**\n" + "\n".join([f"- {src}" for src in used_sources])
        logger("Troubleshooting report generated.")
    except Exception as e:
        logger(f"Error during report synthesis: {e}")
        report = f"Sorry, I encountered an error while generating the final report. Please review the analysis provided earlier.\nAnalysis:\n{analysis}\n(Error: {e})"
    return {**state, "troubleshooting_report": report, "clarification_question": None} 