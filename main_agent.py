# This is the content of main_agent.py, which was previously nvidia_langgraph_agent.py
# All previous code from nvidia_langgraph_agent.py (after all modifications) goes here.
import os
from dotenv import load_dotenv
load_dotenv()
import openai
from typing import Callable, Optional, List, Dict, Any 
# import json # No longer directly used here, graph_nodes.py and search_utils.py handle their json needs
# from pathlib import Path # No longer directly used here, graph_nodes.py and search_utils.py handle their Path needs

from agent_state import AgentState
from graph_builder import workflow, troubleshooting_workflow

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Helper Function for Meta-Clarification Detection ---
def detect_meta_clarification(user_input: str, agent_question: str, logger=None) -> bool:
    logger = logger or (lambda msg: print(f"DEBUG: {msg}"))
    logger(f"Checking if user needs clarification on agent question: '{agent_question}'")
    prompt = f"""The assistant asked the user the following question seeking clarification:
 Agent Question: "{agent_question}"
 
 The user responded:
 User Response: "{user_input}"
 
 Did the user's response ask for an explanation, definition, or clarification of the *agent's question* itself, rather than providing an answer to it? Answer with only YES or NO.
 Answer:"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        decision = response.choices[0].message.content.strip().upper()
        logger(f"Meta-clarification detection result: {decision}")
        return "YES" in decision
    except Exception as e:
        logger(f"Error during meta-clarification detection: {e}")
        return False

# --- Helper Function to Explain Agent's Question ---
def explain_clarification_request(user_request: str, agent_question: str, logger=None) -> str:
    logger = logger or (lambda msg: print(f"DEBUG: {msg}"))
    logger(f"Generating explanation for agent question: '{agent_question}'")
    prompt = f"""An assistant asked a user the following technical question to troubleshoot an issue:
    Agent Question: "{agent_question}"

    The user didn't understand the question and asked for clarification:
    User Request: "{user_request}"

    Explain the *agent's original question* in simpler terms. Provide context or examples to help the user understand what information is needed. Focus on explaining the technical terms or concepts in the agent's question.

    Explanation:"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2,
        )
        explanation = response.choices[0].message.content.strip()
        logger(f"Generated explanation: {explanation}")
        return explanation
    except Exception as e:
        logger(f"Error generating explanation: {e}")
        return "Sorry, I encountered an error trying to explain that question."

# --- Main Function ---
def run_agent(
    question: str,
    mode: str = "answer",
    logger=None,
    search_engine: str = "duckduckgo",
    sources: Optional[List[str]] = None,
    history: Optional[List[dict]] = None,
    resume_from_state: Optional[AgentState] = None
) -> dict:
    if logger is None:
        def print_logger(msg):
            print(f"LOG: {msg}")
        logger = print_logger
    if mode == "troubleshoot" and sources is None:
        sources = ["docs.nvidia.com", "nvidia.custhelp.com", "forums.developer.nvidia.com", "developer.nvidia.com/blog"]
    elif sources is None:
        sources = ["docs.nvidia.com"]
    if history is None:
        history = []
    config = {"recursion_limit": 25}

    if mode == "troubleshoot":
        logger(f"--- Entered run_agent (Troubleshoot Mode) ---")
        current_state_for_invoke: AgentState
        if resume_from_state is None:
            logger(f"Starting new troubleshoot for: {question}")
            current_interaction = {"role": "user", "content": question}
            full_history = history + [current_interaction]
            initial_state: AgentState = {
                "question": question,
                "original_question": question,
                "problem_statement": question,
                "logger": logger,
                "search_engine": search_engine,
                "sources": sources,
                "history": full_history,
                "user_feedback": "",
                "needs_clarification": False,
                "clarification_rounds": 0,
                "performed_gap_search": False,
                "clarification_history": [],
                "search_attempts": 0,
                "context_quality": 0.0,
                "context_gaps": [],
                "confidence": 0.0,
                "additional_queries": [],
                "plan_action": "search",
                "max_queries": 3,
                "skip_search": False,
                "last_missing_info": [],
                "last_confidence": 0.0,
                "max_total_searches": 3,
                "max_total_clarifications": 3,
            }
            current_state_for_invoke = initial_state
        else:
            logger("Resuming troubleshoot workflow...")
            resuming_state: AgentState = resume_from_state
            resuming_state["logger"] = logger
            user_clarification_text = question
            agent_question_to_clarify = resuming_state.get("clarification_question", "")
            if agent_question_to_clarify and user_clarification_text:
                needs_meta_clarification = detect_meta_clarification(
                    user_clarification_text,
                    agent_question_to_clarify,
                    logger
                )
                if needs_meta_clarification:
                    explanation = explain_clarification_request(
                        user_clarification_text,
                        agent_question_to_clarify,
                        logger
                    )
                    return {
                        "status": "explanation_provided",
                        "explanation": explanation,
                        "state": resuming_state
                    }
            resuming_state["user_feedback"] = user_clarification_text
            resuming_state["history"] = resuming_state.get("history", []) + [{
                "role": "user",
                "content": f"(In response to: '{agent_question_to_clarify[:100]}...') {user_clarification_text}"
            }]
            resuming_state["needs_clarification"] = False
            resuming_state["clarification_question"] = ""
            resuming_state["clarification_rounds"] = resuming_state.get("clarification_rounds", 0) + 1
            resuming_state["performed_gap_search"] = False
            logger(f"Resuming troubleshoot with feedback. Round: {resuming_state['clarification_rounds']}")
            logger(f"User feedback to incorporate: '{resuming_state['user_feedback'][:100]}...'")

            # --- Update problem_statement with distilled facts from the latest user feedback ---
            try:
                summarization_prompt = (
                    "Extract the key factual information the user just provided about the issue "
                    "in one short clause (<=15 words). Only include concrete details, no opinions."
                    f"\nUser feedback:\n{user_clarification_text}\n"
                )
                summary_resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": summarization_prompt}],
                    max_tokens=40,
                    temperature=0.0,
                )
                distilled_fact = summary_resp.choices[0].message.content.strip()
                existing_statement = resuming_state.get("problem_statement", resuming_state.get("original_question", ""))
                if distilled_fact and distilled_fact not in existing_statement:
                    resuming_state["problem_statement"] = existing_statement + "; " + distilled_fact
            except Exception as e:
                logger(f"Failed to distill user feedback into problem statement: {e}")

            current_state_for_invoke = resuming_state
        
        result = troubleshooting_workflow.invoke(current_state_for_invoke, config=config)

        final_state_from_graph = result
        needs_clarification_check = False
        if final_state_from_graph.get("plan_action") == "clarify" and \
           final_state_from_graph.get("clarification_question") and \
           final_state_from_graph.get("needs_clarification"): 
            needs_clarification_check = True
            logger(f"--- Interruption Check (Cond 1): Plan was 'clarify' and question ('{final_state_from_graph.get('clarification_question')[:50]}...') is set. ---")
        elif isinstance(final_state_from_graph, dict) and \
             final_state_from_graph.get("clarification_question") and \
             final_state_from_graph.get("needs_clarification") and \
             not final_state_from_graph.get("troubleshooting_report"): 
            needs_clarification_check = True
            logger(f"--- Interruption Check (Cond 2): Final state has question ('{final_state_from_graph.get('clarification_question')[:50]}...') and needs_clarification=True. ---")

        if needs_clarification_check:
            logger("--- INTERRUPTED: Needs clarification from user (based on final graph state analysis) ---")
            return {"status": "needs_clarification", "state": final_state_from_graph}
        else:
            logger("--- Troubleshooting Workflow Complete (or did not explicitly signal clarification) ---")
            report = final_state_from_graph.get("troubleshooting_report", "Error: No report generated.")
            final_sources = final_state_from_graph.get("used_sources", [])
            return {
                "status": "complete",
                "output": report,
                "sources": final_sources,
                "context": final_state_from_graph.get("context", ""),
                "state": final_state_from_graph
            }

    elif mode == "answer":
        logger(f"--- Running Standard Answer Workflow for: {question} ---")
        current_interaction = {"role": "user", "content": question}
        full_history = history + [current_interaction]
        initial_state: AgentState = {
            "question": question,
            "original_question": question,
            "problem_statement": question,
            "logger": logger,
            "search_engine": search_engine,
            "sources": sources,
            "history": full_history,
            "plan_action": "search", 
            "max_queries": 3,    
            "context_quality": 0.0, 
        }
        result = workflow.invoke(initial_state, config=config)

        logger("--- Standard Answer Workflow Complete ---")
        answer = result.get("answer", "Error: No answer generated.")
        answer_sources = result.get("used_sources", [])
        return {
            "status": "complete",
            "output": answer,
            "sources": answer_sources,
            "context": result.get("context", ""),
            "state": result
        }
    else:
        logger(f"Error: Unknown mode '{mode}' requested.")
        return {"status": "error", "output": f"Unknown mode: {mode}", "sources": []}


# --- Example Usage (for testing) ---
if __name__ == "__main__":
    # Example 1: Troubleshooting query
    test_question_troubleshoot = "My NVIDIA DGX node with 8 H100 GPUs has a problem where GPU temps reach 95C under load. What could cause this and how can I fix it?"
    print(f"\n--- TESTING TROUBLESHOOTING MODE ---")
    print(f"Query: {test_question_troubleshoot}")
    result_troubleshoot = run_agent(
        question=test_question_troubleshoot,
        mode="troubleshoot",
        search_engine="duckduckgo", 
    )
    print("\n--- FINAL TROUBLESHOOTING REPORT ---")
    print(result_troubleshoot["output"])

    print("\n" + "="*50 + "\n")

    # Example 2: Standard answer query
    test_question_answer = "What is NVIDIA NVLink?"
    print(f"\n--- TESTING STANDARD ANSWER MODE ---")
    print(f"Query: {test_question_answer}")
    result_answer = run_agent(
        question=test_question_answer,
        mode="answer",
        search_engine="duckduckgo"
    )
    print("\n--- STANDARD ANSWER ---")
    print(result_answer["output"]) 