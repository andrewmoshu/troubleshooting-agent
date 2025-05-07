import os
from dotenv import load_dotenv
load_dotenv()
import openai
from langgraph.graph import StateGraph, END
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, Callable, Optional, List, Tuple
import time

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Define the state schema for LangGraph
class AgentState(TypedDict, total=False):
    question: str
    logger: Optional[Callable[[str], None]]
    queries: List[str]
    links: List[str]
    context: str
    answer: str
    search_engine: str
    sources: List[str]
    used_sources: List[str]
    history: List[dict]

# Google Custom Search API integration
def google_search(query: str, sources: List[str], max_results: int = 3, logger=None) -> List[str]:
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        if logger:
            logger("Google API key or CSE ID not set. Skipping Google search.")
        return []
    logger = logger or (lambda msg: None)
    links = []
    for source in sources:
        logger(f"[Google] Searching for: site:{source} {query}")
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CSE_ID,
            "q": f"site:{source} {query}",
            "num": max_results,
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("items", []):
                link = item.get("link", "")
                if any(domain in link for domain in sources):
                    links.append(link)
        except Exception as e:
            logger(f"Google search failed for {source}: {e}")
    return links

# Step 1: Reflect and rewrite query

def reflect_and_rewrite(state: AgentState) -> AgentState:
    logger = state.get("logger", lambda msg: None)
    question = state["question"]
    history = state.get("history", [])
    # Build conversation history string
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    logger("Reflecting on the query and generating rewrites if needed...")
    prompt = f"""
You are an expert at information retrieval. Here is the conversation so far:
{history_str}

The user's latest message is:
{question}

Rewrite the latest user message in up to 2 alternative ways, considering the conversation context. Return a list of queries: the original and your rewrites (if any).
List of queries:
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.3,
    )
    queries = [q.strip('- ').strip() for q in response.choices[0].message.content.strip().split('\n') if q.strip()]
    logger(f"Generated {len(queries)} queries: {queries}")
    return {**state, "queries": queries}

# Step 2: Multi-query search with engine selection, sources, and rate limit handling

def search_queries(state: AgentState) -> AgentState:
    logger = state.get("logger", lambda msg: None)
    queries = state["queries"]
    search_engine = state.get("search_engine", "duckduckgo")
    sources = state.get("sources", ["docs.nvidia.com"])
    logger(f"Using search engine: {search_engine}")
    logger(f"Searching sources: {sources}")
    links = set()
    if search_engine == "google":
        for q in queries:
            results = google_search(q, sources=sources, max_results=3, logger=logger)
            for link in results:
                if any(domain in link for domain in sources):
                    links.add(link)
    else:
        with DDGS() as ddgs:
            for q in queries:
                for source in sources:
                    logger(f"[DuckDuckGo] Searching: site:{source} {q}")
                    retries = 0
                    while retries < 5:
                        try:
                            for r in ddgs.text(f"site:{source} {q}", max_results=3):
                                if any(domain in r["href"] for domain in sources):
                                    links.add(r["href"])
                            break  # Success, break out of retry loop
                        except Exception as e:
                            msg = str(e)
                            if "429" in msg or "202" in msg:
                                wait_time = 2 ** retries
                                logger(f"Rate limited by DuckDuckGo (status {msg}). Waiting {wait_time} seconds before retrying...")
                                time.sleep(wait_time)
                                retries += 1
                            else:
                                logger(f"DuckDuckGo search failed: {e}")
                                break
                    else:
                        logger(f"Failed to search DuckDuckGo for query 'site:{source} {q}' after several retries.")
    logger(f"Found {len(links)} unique documentation links.")
    return {**state, "links": list(links)}

# Step 3: Fetch and aggregate content

def extract_main_text(url, max_chars=2000):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        main = soup.find("main")
        text = main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)
        return text[:max_chars]
    except Exception as e:
        return f"[Could not fetch content from {url}: {e}]"

def fetch_and_aggregate(state: AgentState) -> AgentState:
    logger = state.get("logger", lambda msg: None)
    links = state["links"]
    logger(f"Fetching and extracting content from {len(links)} links...")
    docs = []
    used_sources = []
    for url in links:
        logger(f"Fetching: {url}")
        text = extract_main_text(url)
        docs.append(f"Source: {url}\n{text}\n")
        used_sources.append(url)
    logger(f"Aggregated content from {len(docs)} sources.")
    return {**state, "context": '\n---\n'.join(docs), "used_sources": used_sources}

# Step 4: Synthesize answer

def synthesize_answer(state: AgentState) -> AgentState:
    logger = state.get("logger", lambda msg: None)
    question = state["question"]
    context = state["context"]
    logger("Generating answer using OpenAI...")
    prompt = f"""
You are an expert on NVIDIA products and technologies. Use the following documentation context (from docs.nvidia.com and/or catalog.ngc.nvidia.com) to answer the user's question as accurately as possible. If the answer is not in the context, say so and suggest visiting https://docs.nvidia.com/ or https://catalog.ngc.nvidia.com/ for more information.

Context:
{context}

Question: {question}
Answer:
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.2,
    )
    answer = response.choices[0].message.content.strip()
    logger("Answer generated.")
    return {**state, "answer": answer}

# Build the LangGraph workflow
graph = StateGraph(AgentState)
graph.add_node("reflect", reflect_and_rewrite)
graph.add_node("search", search_queries)
graph.add_node("aggregate", fetch_and_aggregate)
graph.add_node("synthesize", synthesize_answer)
graph.add_edge("reflect", "search")
graph.add_edge("search", "aggregate")
graph.add_edge("aggregate", "synthesize")
graph.add_edge("synthesize", END)
graph.set_entry_point("reflect")
workflow = graph.compile()

def langgraph_answer_question(question: str, logger=None, search_engine: str = "duckduckgo", sources: Optional[List[str]] = None, history: Optional[List[dict]] = None) -> Tuple[str, List[str]]:
    if sources is None:
        sources = ["docs.nvidia.com"]
    if history is None:
        history = []
    state: AgentState = {"question": question, "logger": logger, "search_engine": search_engine, "sources": sources, "history": history}
    result = workflow.invoke(state)
    return result["answer"], result.get("used_sources", []) 