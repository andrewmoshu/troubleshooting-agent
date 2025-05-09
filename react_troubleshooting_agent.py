import os
from dotenv import load_dotenv
from langchain_core.tools import Tool, tool # Updated import for Tool
from langchain_google_community import GoogleSearchAPIWrapper # For Google Search
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI # Or any other ChatModel provider
from langchain_core.pydantic_v1 import BaseModel, Field
from datetime import datetime
from typing import Dict, List, Optional, Callable, Tuple # Added Callable, Tuple

# Imports needed for content fetching and summarization
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import re
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai # Need direct openai client for one fallback path in summarize

# Load environment variables (e.g., OPENAI_API_KEY, GOOGLE_API_KEY, GOOGLE_CSE_ID)
load_dotenv()

# Ensure OPENAI_API_KEY is available for summarization
if not os.getenv("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY not set. Summarization will fail.")

# Direct OpenAI client needed for summarize_text fallback
main_openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Utility Functions (Copied/Adapted from search_utils.py) ---

# Placeholder logger if needed by copied functions
def _default_logger(msg: str):
    print(f"UTIL_LOG: {msg}")

# --- extract_main_text --- (Ensure imports: requests, BeautifulSoup, fitz, re)
def extract_main_text(url, max_chars=120000, logger=None):
    logger = logger or _default_logger
    text = ""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        resp = requests.get(url, timeout=15, headers=headers)
        resp.raise_for_status()
        content_type = resp.headers.get('Content-Type', '')

        if 'application/pdf' in content_type.lower():
            logger(f"Processing PDF content from {url}")
            try:
                pdf_bytes = resp.content
                with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                    all_text = [page.get_text() for page in doc]
                    text = "\n".join(all_text)
                logger(f"Successfully extracted ~{len(text)} chars from PDF {url}")
            except Exception as pdf_err:
                logger(f"Failed to parse PDF {url}: {pdf_err}")
                return f"[Could not parse PDF content from {url}: {pdf_err}]"
        elif 'html' in content_type.lower():
            logger(f"Processing HTML content from {url}")
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "button", "input"]):
                tag.decompose()
            main_content = soup.find("main") or soup.find("article") or soup.find("div", role="main") or soup.find("div", class_=re.compile(r'(body|content|main)', re.I))
            if main_content:
                text = main_content.get_text(separator="\n", strip=True)
            else:
                body = soup.find("body")
                text = body.get_text(separator="\n", strip=True) if body else soup.get_text(separator="\n", strip=True)
            logger(f"Successfully extracted ~{len(text)} chars from HTML {url}")
        else:
             logger(f"Skipping unsupported content type at {url} (Content-Type: {content_type})")
             return f"[Skipped unsupported content type ({content_type}) at {url}]"

        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = text.strip()
        if not text:
             logger(f"Warning: No text extracted from {url} after cleaning.")
             return f"[No text content extracted from {url}]"
        # Return full text up to max_chars limit before summarization
        return text[:max_chars]
    except requests.exceptions.RequestException as e:
        logger(f"HTTP Error fetching {url}: {e}")
        return f"[Could not fetch content from {url}: Network Error {e}]"
    except Exception as e:
        logger(f"Error processing {url}: {e}")
        return f"[Could not process content from {url}: {e}]"

# --- summarize_text --- (Ensure imports: ChatOpenAI, openai, os)
def summarize_text(text: str, logger: Callable[[str], None], max_tokens: int = 400, focus: str = "") -> str:
    snippet = text[:10000] # Use a large snippet for context

    if focus:
        prompt = f"""Condense the following technical text, focusing *only* on aspects relevant to the question: "{focus}".
Extract key facts, steps, causes, or solutions related to the question.

Text (partial):
{snippet}

---
Relevant concise bullet points (6-12 max):"""
    else:
        prompt = f"""Condense the following technical text into a short, information-dense bullet list (6-12 max).
Capture root causes, symptoms, error codes, commands or fixes.

Text (partial):
{snippet}

---
Concise bullet points:"""
    
    try:
        # Use ChatOpenAI for consistency if available and configured
        llm = ChatOpenAI(model="gpt-4.1", temperature=0.2, openai_api_key=os.getenv("OPENAI_API_KEY"))
        response = llm.invoke(prompt)
        summary = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        if not isinstance(summary, str) or not summary:
            raise ValueError("ChatOpenAI did not return a valid summary string.")
        logger(f"Summarization produced {len(summary)} chars.")
        return summary
    except Exception as e:
        logger(f"Summarization error with ChatOpenAI: {e}. Trying direct API call.")
        # Fallback using the direct client (ensure main_openai_client is initialized)
        try:
            response_direct = main_openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            summary = response_direct.choices[0].message.content.strip()
            logger(f"Summarization (fallback direct API) produced {len(summary)} chars.")
            return summary
        except Exception as e_fallback:
            logger(f"Summarization error (fallback direct API): {e_fallback}. Using truncated text.")
            return text[:3000] # Truncate original text as final fallback

# --- chunked_summarize --- (Ensure imports: ThreadPoolExecutor, as_completed)
def chunked_summarize(text: str, logger: Callable[[str], None], focus: str = "") -> str:
    CHUNK_SIZE = 30000
    OVERLAP = 5000
    if len(text) <= CHUNK_SIZE:
        # Pass focus to single chunk summarization
        return summarize_text(text, logger, 400, focus)

    chunk_list: List[str] = [text[start : start + CHUNK_SIZE] for start in range(0, len(text), CHUNK_SIZE - OVERLAP)]
    logger(f"Dispatching {len(chunk_list)} chunk summarization jobs in parallel…")

    partial_summaries: List[Tuple[int, str]] = []
    with ThreadPoolExecutor(max_workers=min(6, len(chunk_list))) as pool:
        # Pass focus to each worker
        future_map = {pool.submit(summarize_text, chunk, logger, 400, focus): idx for idx, chunk in enumerate(chunk_list)}
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                summary_res = future.result()
                partial_summaries.append((idx, summary_res))
            except Exception as e:
                logger(f"Summarization failed for chunk {idx+1}: {e}")
                partial_summaries.append((idx, "[Summarization Error]"))

    partial_summaries.sort(key=lambda t: t[0])
    combined = "\n".join([s for _, s in partial_summaries if s != "[Summarization Error]"])

    if len(combined) > CHUNK_SIZE: 
        logger("Running reduce summarization on combined chunks…")
        # Pass focus to the final reduce step
        combined = summarize_text(combined, logger, 400, focus)
    elif not combined: # Handle case where all chunks failed
        logger("All chunk summarizations failed. Returning truncated original text.")
        return text[:3000]
        
    return combined

# --- Summary Cache Handling ---
SUMMARY_CACHE_FILE = Path("react_summary_cache.json")
_summary_cache_data: Dict[str, str] = {}

try:
    if SUMMARY_CACHE_FILE.exists():
        _summary_cache_data = json.loads(SUMMARY_CACHE_FILE.read_text())
except Exception as e:
    print(f"Warning: Failed to load summary cache: {e}")
    _summary_cache_data = {}

def _save_summary_cache():
    try:
        SUMMARY_CACHE_FILE.write_text(json.dumps(_summary_cache_data))
    except Exception as e:
        print(f"Warning: Failed to save summary cache: {e}")

# --- 1. Define Tools ---

# Ensure GOOGLE_API_KEY and GOOGLE_CSE_ID are set in your environment for this to work.
# You can get them from: 
# GOOGLE_API_KEY: Google Cloud credential console (https://console.cloud.google.com/apis/credentials)
# GOOGLE_CSE_ID: Programmable Search Engine (https://programmablesearchengine.google.com/controlpanel/create)

if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GOOGLE_CSE_ID"):
    print("WARNING: GOOGLE_API_KEY or GOOGLE_CSE_ID environment variables not set. Google Search will not work.")
    # Fallback to a dummy tool or raise an error if Google Search is critical
    google_search_tool = Tool(
        name="web_search_disabled",
        description="Web search is currently disabled due to missing API keys.",
        func=lambda q: "Search is disabled. Please configure GOOGLE_API_KEY and GOOGLE_CSE_ID."
    )
else:
    google_search_api = GoogleSearchAPIWrapper(k=5) # Get top 5 results

    def google_search_with_metadata_and_content(query: str) -> List[Dict[str, str]]:
        """
        Performs a Google Search, then fetches and summarizes the content of the top results relevant to the query.
        Returns a list of dictionaries, each containing: title, link, snippet, and fetched_content_summary.
        """
        print(f"--- Tool: google_search_with_metadata_and_content, Query: {query} ---")
        processed_results: List[Dict[str, str]] = []
        MAX_RESULTS_TO_FETCH = 3 
        SUMMARY_TRIGGER_LENGTH = 8000

        try:
            # 1. Get SERP results
            serp_results = google_search_api.results(query, num_results=5)
            if not serp_results:
                # Return a list with a single item indicating no results, to maintain type consistency
                return [{"title": "No Results", "link": "", "snippet": "No search results found via Google.", "fetched_content_summary": ""}]

            # Process top N results for content fetching
            links_to_process = [(r.get('link'), r.get('title', 'N/A'), r.get('snippet', 'N/A')) for r in serp_results[:MAX_RESULTS_TO_FETCH] if r.get('link')]
            
            for i, (link, title, snippet) in enumerate(links_to_process):
                print(f"Processing result {i+1}/{len(links_to_process)}: {link}")
                
                content_summary = "[Content not fetched or processed]"
                if link in _summary_cache_data:
                    print(f"Using cached summary for {link}.")
                    content_summary = _summary_cache_data[link]
                else:
                    extracted_text = extract_main_text(link, logger=_default_logger)
                    if extracted_text and not extracted_text.startswith("["):
                        if len(extracted_text) > SUMMARY_TRIGGER_LENGTH:
                            print(f"Text from {link} is large ({len(extracted_text)} chars). Summarizing with focus: '{query}'...")
                            content_summary = chunked_summarize(extracted_text, _default_logger, focus=query)
                        else:
                            content_summary = extracted_text
                        
                        _summary_cache_data[link] = content_summary
                        _save_summary_cache()
                    else:
                        content_summary = extracted_text 
                
                processed_results.append({
                    "title": title,
                    "link": link,
                    "snippet": snippet,
                    "fetched_content_summary": content_summary
                })
            
            # Add remaining SERP results (without fetched content)
            for r in serp_results[MAX_RESULTS_TO_FETCH:]:
                processed_results.append({
                    "title": r.get('title', 'N/A'),
                    "link": r.get('link', 'N/A'),
                    "snippet": r.get('snippet', 'N/A'),
                    "fetched_content_summary": "[Content not fetched for this result]"
                })

            return processed_results

        except Exception as e:
            print(f"Error during enhanced Google search: {str(e)}")
            import traceback; traceback.print_exc()
            # Return a list with an error entry
            return [{"title": "Error", "link": "", "snippet": f"Error during enhanced Google search: {str(e)}", "fetched_content_summary": ""}]

    google_search_tool = Tool(
        name="web_search",
        description="Performs a Google Search for the query, then fetches and summarizes the content of the top results. Use this to get detailed, up-to-date information and potential solutions.",
        func=google_search_with_metadata_and_content, # Use the new function
    )

# --- Tool: Ask User for Clarification ---
class AskUserInput(BaseModel):
    question_to_user: str = Field(description="The specific question you need to ask the user to get clarification or more details.")

@tool("ask_user_for_clarification", args_schema=AskUserInput)
def ask_user_tool(question_to_user: str) -> str:
    """Use this tool when you need to ask the user a follow-up question to clarify their problem, gather more specific details, or request information that cannot be found via web search. The input should be the exact question you want to ask the user."""
    print(f"--- Tool: ask_user_tool --- Triggered with question: {question_to_user}")
    # In a real application, this would pause and wait for user's textual input.
    # For this CLI, the question will be part of the agent's response, and the user will reply in the next turn.
    return f"ACTION_ASK_USER: {question_to_user}" # The agent will see this and should incorporate the question in its response.

# --- Knowledge Log Tools ---
# In a real application, this would be a proper database or at least saved/loaded with the checkpointer.
# For this CLI example, it's a global in-memory list, reset each time the script runs.
knowledge_log_store: Dict[str, List[str]] = {} # thread_id -> list of entries

class UpdateLogInput(BaseModel):
    entry_type: str = Field(description="Type of entry, e.g., 'user_clarification', 'search_finding', 'hypothesis_ruled_out', 'actionable_step_identified', 'current_summary'.")
    summary: str = Field(description="A concise summary of the information or finding.")
    source: str = Field(description="Source of the information (e.g., 'user_input', 'web_search: <url>', 'agent_deduction').", default="agent_deduction")

@tool("update_knowledge_log", args_schema=UpdateLogInput) # Ensure args_schema is correctly linked
def update_knowledge_log_tool(entry_type: str, summary: str, source: str = "agent_deduction") -> str:
    """
    Records a significant finding, user clarification, ruled-out hypothesis, or current understanding summary
    into a structured knowledge log for the current session. This helps maintain a clear understanding of the problem.
    The entry should be a concise summary. Call this after processing new information.
    Example: update_knowledge_log(entry_type='search_finding', summary='Found NVIDIA DGX H100 manual, mentions thermal limits.', source='web_search: docs.nvidia.com/dgx/...')
    """
    global knowledge_log_store
    # This global `thread_id_for_tools` will be set in the `if __name__ == "__main__"` block.
    # This is a workaround for CLI demo purposes.
    global thread_id_for_tools
    if not thread_id_for_tools:
        return "Error: Session ID (thread_id_for_tools) not set for knowledge log."
    current_session_id = thread_id_for_tools

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] Type: {entry_type} | Source: {source} | Summary: {summary}"
    
    if current_session_id not in knowledge_log_store:
        knowledge_log_store[current_session_id] = []
    knowledge_log_store[current_session_id].append(log_entry)
    
    print(f"--- Tool: update_knowledge_log (Session: {current_session_id}) --- Entry added: {log_entry} ---")
    return f"Knowledge log updated with: '{summary}'. Total entries for this session: {len(knowledge_log_store[current_session_id])}."

class QueryLogInput(BaseModel):
    topic_filter: str = Field(description="Optional keyword/topic to filter the log by (e.g., 'error codes', 'cooling'). Leave empty for the full log summary.", default="")

@tool("query_knowledge_log", args_schema=QueryLogInput) # Ensure args_schema is correctly linked
def query_knowledge_log_tool(topic_filter: str = "") -> str:
    """
    Retrieves and summarizes the current knowledge log for this session. Useful for reviewing known facts,
    identifying information gaps, or before formulating a final answer. Optionally filter by a topic.
    Example: query_knowledge_log(topic_filter='thermal limits')
    """
    global knowledge_log_store
    global thread_id_for_tools
    if not thread_id_for_tools:
        return "Error: Session ID (thread_id_for_tools) not set for knowledge log."
    current_session_id = thread_id_for_tools

    if current_session_id not in knowledge_log_store or not knowledge_log_store[current_session_id]:
        return "Knowledge log for this session is currently empty."

    session_log = knowledge_log_store[current_session_id]
    print(f"--- Tool: query_knowledge_log (Session: {current_session_id}) --- Topic: {topic_filter if topic_filter else 'ALL'} ---")

    if topic_filter:
        relevant_entries = [entry for entry in session_log if topic_filter.lower() in entry.lower()]
        if not relevant_entries:
            return f"No entries found in the knowledge log for topic: '{topic_filter}' in this session."
        # For brevity, just return the relevant entries. LLM can summarize if needed.
        # In a more complex system, an LLM could summarize these filtered entries.
        return "Relevant Knowledge Log Entries for this session:\n" + "\n".join(relevant_entries)
    
    # If no filter, provide a summary of the log instead of the whole raw log if it's too long
    if len(session_log) > 5 and len("\n".join(session_log)) > 1000: # Heuristic for "too long"
        # This part could also use an LLM for a better summary of the log
        summary_of_log = f"Knowledge log for this session contains {len(session_log)} entries. Key types found: "
        type_counts = {}
        for entry in session_log:
            try:
                entry_type = entry.split("Type: ", 1)[1].split(" |", 1)[0]
                type_counts[entry_type] = type_counts.get(entry_type, 0) + 1
            except:
                pass # ignore parsing errors for summary
        summary_of_log += ", ".join([f"{k} ({v})" for k, v in type_counts.items()])
        summary_of_log += ". Use a specific topic_filter to get details."
        return summary_of_log

    return "Current Knowledge Log for this session:\n" + "\n".join(session_log)

all_tools = [google_search_tool, ask_user_tool, update_knowledge_log_tool, query_knowledge_log_tool]

# --- 2. Define LLM ---
# Ensure your OPENAI_API_KEY (or other provider's key) is set in your environment
# You can also use other models like Anthropic's Claude
# For example: model = ChatAnthropic(model="claude-3-haiku-20240307")
# model_name = "anthropic/claude-3-haiku-20240307" # Requires ANTHROPIC_API_KEY
model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4.1")
llm = ChatOpenAI(model=model_name, temperature=0.4)


# --- 3. Define Prompt ---
TROUBLESHOOTING_PROMPT = """You are an expert troubleshooting assistant called Daisy. Your goal is to help users diagnose and solve technical problems.

Follow these steps:
1.  **Understand the Problem**: Carefully analyze the user's description of the issue. If crucial details are missing, use the `ask_user_for_clarification` tool to ask specific questions. Record the initial problem and user clarifications using `update_knowledge_log`.
2.  **Formulate Hypotheses**: Based on the problem and current knowledge log (use `query_knowledge_log` if needed), think about potential causes.
3.  **Gather Information (Use Tools)**:
    *   If you need to find general information, solutions, or common causes for an issue, use the `web_search` tool. This tool uses Google Search *and fetches/summarizes content* from the top results.
    *   Provide a concise and targeted search query for the `web_search` tool. After receiving results (which include summaries), use `update_knowledge_log` to record key findings from the *summarized content*, making sure to include the source link.
    *   If, after reviewing the fetched content, you still need more specific details *from the user*, use the `ask_user_for_clarification` tool.
4.  **Analyze Tool Output**: Review the *fetched content summaries* returned by the `web_search` tool. Extract key findings and cite the source link for each relevant piece of information (e.g., "The summary from <url> suggests checking airflow.").
    *   Does the content confirm or deny a hypothesis?
    *   Does it provide a solution?
    *   Does it suggest further diagnostic steps?
    *   Update your understanding by calling `update_knowledge_log`.
5.  **Iterate**:
    *   If you have a likely solution *based on fetched content*, propose it to the user clearly, including actionable steps and *citing the source link(s)*.
    *   If you need more information that can be found online, use `web_search` again.
    *   If you need more information *from the user*, use the `ask_user_for_clarification` tool.
    *   If a search provided irrelevant content summaries, record this with `update_knowledge_log` (`entry_type='search_content_irrelevant'`) and try a different search query or approach.
    *   Periodically, or before asking the user, use `query_knowledge_log` to review what you already know.
6.  **Conclude**: Once you have a high-confidence solution or have exhausted reasonable troubleshooting steps, use `query_knowledge_log` to get a final summary of all findings. Then, synthesize this information (including details from fetched content) into a comprehensive final report for the user, ensuring relevant source links are included. If you couldn't solve it, explain what you tried and suggest where they might seek further help.

- Always think step-by-step before deciding on an action.
- When you use a tool, state the tool and the input you are giving it.
- When using `ask_user_for_clarification`, your output to the user should naturally incorporate the question you are asking via the tool.
- After receiving the tool's observation (which now includes content summaries for web_search), explain how it informs your next step.
- When presenting information derived from fetched web content, always cite the source link. Format citations clearly, for example: `(Source: <link_url>)`.
- Ensure proposed solutions are broken down into clear, actionable steps for the user to follow.
"""

# --- 4. Enable Memory ---
checkpointer = InMemorySaver()

# --- 5. Instantiate Agent ---
# The `create_react_agent` function compiles a new graph that orchestrates the LLM and tools.
troubleshooting_agent_executor = create_react_agent(
    model=llm,
    tools=all_tools,
    prompt=TROUBLESHOOTING_PROMPT, # You can customize this prompt
    checkpointer=checkpointer
)

# --- 6. Example Usage ---
if __name__ == "__main__":
    print("Starting a troubleshooting session with Google Search...")

    # Global variable to hold the current session_id for tools (CLI Demo Workaround)
    thread_id_for_tools = ""

    print("Ensure GOOGLE_API_KEY and GOOGLE_CSE_ID are set in your .env file or environment.")
    print("Type 'exit' to end the session.\n")

    thread_id = "troubleshoot_session_g_001" # Changed thread_id to avoid old state
    config = {"configurable": {"thread_id": thread_id}}

    thread_id_for_tools = thread_id # Set the global for the tools for this session
    initial_problem = input("What technical problem are you facing today?\nUser: ")

    if initial_problem.lower() != 'exit':
        current_input = initial_problem

        while True:
            inputs = {"messages": [{"role": "user", "content": current_input}]}
            print("\nAgent is thinking...")

            # It's useful to keep track of the last printed assistant message to avoid re-printing duplicates
            # from multiple stream events that might not have new visible content for the user.
            last_printed_agent_message = ""

            # stream_mode="debug" gives a richer stream of events.
            # We will look for 'checkpoint' events to get the latest agent state and messages.
            for chunk in troubleshooting_agent_executor.stream(inputs, config=config, stream_mode="debug"):
                print(f"\n--- Agent Event Chunk ({chunk.get('type', 'N/A')} - Step {chunk.get('step', 'N/A')}) ---")
                # For more verbose logging, uncomment the next line
                # import pprint; pprint.pprint(chunk, indent=2, depth=3) # Limit depth to avoid huge prints

                # Check if this chunk is a checkpoint and contains the state we need
                if chunk.get("type") == "checkpoint" and chunk.get("payload") and "values" in chunk["payload"]:
                    agent_state_values = chunk["payload"]["values"]
                    if "messages" in agent_state_values and agent_state_values["messages"]:
                        last_message_obj = agent_state_values["messages"][-1]
                        # Ensure it's an AIMessage (or assistant) and has content
                        if hasattr(last_message_obj, 'type') and last_message_obj.type == "ai" and hasattr(last_message_obj, 'content'):
                            current_agent_content = last_message_obj.content
                            if current_agent_content and current_agent_content != last_printed_agent_message:
                                print(f"\nAgent: {current_agent_content}")
                                last_printed_agent_message = current_agent_content
                        elif hasattr(last_message_obj, 'type') and last_message_obj.type == "tool" and hasattr(last_message_obj, 'content'):
                            print(f"DEBUG TOOL RESULT: Tool: {last_message_obj.name}, Content: {str(last_message_obj.content)[:200]}...")
                elif chunk.get("type") == "task_result": # Usually contains tool results or agent action results
                    if chunk.get("payload") and chunk["payload"].get("name") == "tools" and chunk["payload"].get("result"):
                        tool_results = chunk["payload"]["result"]
                        if isinstance(tool_results, list) and tool_results and isinstance(tool_results[0], tuple) and tool_results[0][0] == "messages":
                            for tool_message in tool_results[0][1]:
                                if hasattr(tool_message, 'type') and tool_message.type == "tool" and hasattr(tool_message, 'content'):
                                    print(f"DEBUG TOOL OUTPUT: Tool: {tool_message.name}, Output: {str(tool_message.content)[:200]}...")

                # The AIMessage that *decides* to call a tool will appear in a checkpoint *before* the tool result.
                # We can also log when the agent is about to call a tool by inspecting AIMessages with tool_calls.
                if chunk.get("type") == "checkpoint" and chunk.get("payload") and "values" in chunk["payload"]:
                    agent_state_values = chunk["payload"]["values"]
                    if "messages" in agent_state_values and agent_state_values["messages"]:
                        last_message_obj = agent_state_values["messages"][-1]
                        if hasattr(last_message_obj, 'tool_calls') and last_message_obj.tool_calls:
                            for tool_call in last_message_obj.tool_calls:
                                print(f"DEBUG TOOL INTENTION: Agent plans to call tool: {tool_call['name']} with args: {tool_call['args']}")

            user_follow_up = input("\nUser: ")
            if user_follow_up.lower() == 'exit':
                print("Exiting session.")
                thread_id_for_tools = "" # Clear it on exit
                break
            current_input = user_follow_up
    else:
        print("No problem provided. Exiting.")

    # To inspect the final state of the conversation for this thread_id:
    # final_thread_state = checkpointer.get(config)
    # print("\n--- Final Conversation State ---")
    # if final_thread_state:
    #     for message in final_thread_state.values['messages']:
    #         print(f"{message.role.capitalize()}: {message.content}") 