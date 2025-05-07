import os
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import re
import json
from pathlib import Path
from datetime import datetime
import time
from typing import List, Dict, Any, Callable, Set, Tuple
from duckduckgo_search import DDGS
from langchain_openai import ChatOpenAI

# Pydantic models used by validate_search_results
from pydantic_models import LinkRelevance # Assuming pydantic_models.py is in the same directory

# Environment variables (consider moving to a config loading mechanism if preferred)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # For summarization and validation LLM calls

# OpenAI client for summarization/validation
client = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY) # General client, can be specified per call if needed

# --- Search and Content Processing Functions ---

def google_search(query: str, sources: List[str], max_results: int = 3, logger=None) -> List[str]:
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        if logger:
            logger("Google API key or CSE ID not set. Skipping Google search.")
        return []
    logger = logger or (lambda msg: print(f"DEBUG: {msg}")) # Basic default logger
    links = []
    for source in sources:
        safe_source = str(source).strip()
        if not safe_source:
            logger(f"Skipping empty source in list: {sources}")
            continue
        api_query = f"site:{safe_source} {query}"
        logger(f"[Google] DEBUG: Processing source: '{safe_source}'")
        logger(f"[Google] Constructed API Query: {api_query}")
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CSE_ID,
            "q": api_query,
            "num": max_results,
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items", [])
            logger(f"Google search for '{api_query}' returned {len(items)} items.")
            for item in items:
                link = item.get("link", "")
                if any(source in link for source in sources):
                    links.append(link)
                else:
                    logger(f"Skipping link (domain mismatch): {link}")
        except Exception as e:
            logger(f"Google search failed for site:{source} {query}: {e}")
    unique_links = list(set(links))
    logger(f"Found {len(unique_links)} unique links from Google for sources {sources}.")
    return unique_links

def search_queries(
    queries: List[str], 
    search_engine: str, 
    sources: List[str], 
    logger: Callable[[str], None],
    max_queries_to_run: int,
    max_results_per_query: int = 3,
    perform_site_search: bool = True # Added based on original usage
    ) -> List[Dict[str, str]]:
    
    logger(f"Using search engine: {search_engine}")
    if perform_site_search and sources:
        logger(f"Searching allowed sources: {sources}")
    else:
        logger("Performing broad web search (no specific sources selected).")
    
    all_results_info: List[Dict[str, str]] = [] 
    unique_urls_processed: Set[str] = set() 

    CACHE_FILE = Path("search_cache.json")
    week_key = datetime.utcnow().strftime("%Y-%U")
    
    # Simplified cache attribute handling for a module-level function
    _cache_data: Dict[str, Any] = {}
    if CACHE_FILE.exists():
        try:
            _cache_data = json.loads(CACHE_FILE.read_text())
        except Exception:
            _cache_data = {} # Initialize if loading fails

    def cache_key(q: str, srcs: List[str], engine: str) -> str:
        src_key = ",".join(sorted(srcs)) if srcs else "_any_"
        return f"{week_key}|{engine}|{q}|{src_key}"

    if search_engine == "google":
        logger("Starting Google Search...")
        for q_idx, q in enumerate(queries):
            if q_idx >= max_queries_to_run:
                logger(f"Reached max_queries limit ({max_queries_to_run}), stopping further Google searches.")
                break
            ck = cache_key(q, sources if perform_site_search else [], search_engine)
            if ck in _cache_data:
                logger(f"Cache hit for Google query '{q}'. Using {len(_cache_data[ck])} cached results.")
                for res_info in _cache_data[ck]:
                    if res_info['url'] not in unique_urls_processed:
                        all_results_info.append(res_info)
                        unique_urls_processed.add(res_info['url'])
                continue
            
            current_query_results_info: List[Dict[str, str]] = []
            if perform_site_search and sources:
                for source_site in sources:
                    api_query = f"site:{source_site} {q}"
                    logger(f"[Google] Site searching: {api_query}")
                    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
                        logger("Google API key or CSE ID not set. Skipping site Google search.")
                        continue
                    url = "https://www.googleapis.com/customsearch/v1"
                    params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID, "q": api_query, "num": max_results_per_query}
                    try:
                        resp = requests.get(url, params=params, timeout=10)
                        resp.raise_for_status()
                        data = resp.json()
                        items = data.get("items", [])
                        logger(f"Google site search for '{api_query}' returned {len(items)} items.")
                        for item in items:
                            link = item.get("link", "")
                            title = item.get("title", "")
                            snippet = item.get("snippet", item.get("htmlSnippet", ""))
                            if link and link not in unique_urls_processed:
                                res_item = {"url": link, "title": title, "snippet": snippet}
                                current_query_results_info.append(res_item)
                                all_results_info.append(res_item)
                                unique_urls_processed.add(link)
                    except Exception as e:
                        logger(f"Google site search failed for {api_query}: {e}")
            else: # Broad Google search
                logger(f"[Google] Broad searching for: {q}")
                if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
                    logger("Google API key or CSE ID not set. Skipping broad Google search.")
                    continue
                url = "https://www.googleapis.com/customsearch/v1"
                params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID, "q": q, "num": max_results_per_query * (len(sources) or 1) }
                try:
                    resp = requests.get(url, params=params, timeout=10)
                    resp.raise_for_status()
                    data = resp.json()
                    items = data.get("items", [])
                    logger(f"Google broad search for '{q}' returned {len(items)} items.")
                    for item in items:
                        link = item.get("link", "")
                        title = item.get("title", "")
                        snippet = item.get("snippet", item.get("htmlSnippet", ""))
                        if link and link not in unique_urls_processed:
                            res_item = {"url": link, "title": title, "snippet": snippet}
                            current_query_results_info.append(res_item)
                            all_results_info.append(res_item)
                            unique_urls_processed.add(link)
                except Exception as e:
                    logger(f"Google broad search failed for query '{q}': {e}")
            
            if current_query_results_info:
                 _cache_data[ck] = current_query_results_info
    else: # DuckDuckGo Search
        logger("Starting DuckDuckGo Search...")
        with DDGS() as ddgs:
            for q_idx, q in enumerate(queries):
                if q_idx >= max_queries_to_run:
                    logger(f"Reached max_queries limit ({max_queries_to_run}), stopping further DDG searches.")
                    break
                
                ck = cache_key(q, sources if perform_site_search else [], search_engine)
                if ck in _cache_data:
                    logger(f"Cache hit for DDG query '{q}'. Using {len(_cache_data[ck])} cached results.")
                    for res_info in _cache_data[ck]:
                        if res_info['url'] not in unique_urls_processed:
                            all_results_info.append(res_info)
                            unique_urls_processed.add(res_info['url'])
                    continue

                current_query_results_info_ddg: List[Dict[str, str]] = []
                search_targets = sources if perform_site_search and sources else [None]
                
                for source_site in search_targets:
                    search_term = f"site:{source_site} {q}" if source_site else q
                    log_prefix = "[DuckDuckGo] Site searching:" if source_site else "[DuckDuckGo] Broad searching:"
                    logger(f"{log_prefix} {search_term}")

                    retries = 0
                    max_retries = 3
                    while retries < max_retries:
                        try:
                            results_gen = ddgs.text(search_term, max_results=max_results_per_query)
                            ddg_raw_results = list(results_gen)
                            logger(f"DDG returned {len(ddg_raw_results)} results for '{search_term}'")
                            
                            for r in ddg_raw_results:
                                href = r.get("href")
                                title = r.get("title", "")
                                snippet = r.get("body", "")
                                
                                if href and href not in unique_urls_processed:
                                    passes_ddg_site_check = True
                                    if source_site and not any(domain_part in href for domain_part in source_site.split(".")):
                                        pass # DDG site: is usually reliable

                                    if passes_ddg_site_check:
                                        res_item = {"url": href, "title": title, "snippet": snippet}
                                        current_query_results_info_ddg.append(res_item)
                                        all_results_info.append(res_item)
                                        unique_urls_processed.add(href)
                            break 
                        except Exception as e:
                            msg = str(e)
                            if "429" in msg or "202" in msg or "timeout" in msg.lower() or "read timed out" in msg.lower():
                                wait_time = 2 ** retries
                                logger(f"Rate limited or timeout by DuckDuckGo for '{search_term}' (attempt {retries+1}/{max_retries}, error: {msg}). Waiting {wait_time}s...")
                                time.sleep(wait_time)
                                retries += 1
                            else:
                                logger(f"DuckDuckGo search failed unexpectedly for '{search_term}': {e}")
                                break 
                    else: 
                        logger(f"Failed to search DuckDuckGo for '{search_term}' after {max_retries} retries.")
                
                if current_query_results_info_ddg:
                    existing_cached_for_q = _cache_data.get(ck, [])
                    existing_urls_in_cache_for_q = {item['url'] for item in existing_cached_for_q}
                    new_items_to_add_to_cache = [item for item in current_query_results_info_ddg if item['url'] not in existing_urls_in_cache_for_q]
                    _cache_data[ck] = existing_cached_for_q + new_items_to_add_to_cache

    logger(f"Found {len(all_results_info)} total search results (URLs with title/snippet) across all queries.")

    try:
        CACHE_FILE.write_text(json.dumps(_cache_data))
    except Exception:
        logger("Warning: Failed to persist search cache to disk.")

    return all_results_info

def validate_search_results(
    search_results_info: List[Dict[str, str]], 
    original_question: str, 
    logger: Callable[[str], None],
    history: List[dict], # Added history
    latest_user_input_for_validation: str, # Added
    max_results_to_validate: int = 10
    ) -> List[str]:

    if not search_results_info:
        logger("No search results to validate.")
        return []

    history_summary_for_validation = ""
    if history:
        recent_turns = history[-4:] 
        summary_parts = []
        for turn in recent_turns:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            if role == "assistant":
                prefixes_to_strip = [
                    "🌼 Daisy needs more information:", "**🌼 Daisy needs more information:**",
                    "🌼 Okay, let me explain:", "**🌼 Okay, let me explain:**",
                    "<span style='color:#76B900;font-size:1.2em'><b>🌼 Daisy's Report:</b></span>",
                    "<span style='color:#76B900;font-size:1.2em'><b>🌼 Daisy's Answer:</b></span>",
                ]
                for prefix in prefixes_to_strip:
                    if content.startswith(prefix):
                        content = content[len(prefix):].strip()
            summary_parts.append(f"{role.capitalize()}: {content[:250]}{'...' if len(content) > 250 else ''}")
        history_summary_for_validation = "\n".join(summary_parts) # Corrected join
        if len(history) > 4:
            history_summary_for_validation = "... (earlier history) ...\n" + history_summary_for_validation
    
    validation_focus_statement = original_question
    if latest_user_input_for_validation: # This was based on is_resuming_after_clarification which is removed. Pass directly.
        validation_focus_statement = f"User's original question was: '{original_question}'. Most recently, the user provided this specific information/clarification: '{latest_user_input_for_validation}'. Judge relevance against THIS LATEST INFORMATION primarily, using the original question for broader context only."
        logger(f"Validating {len(search_results_info)} search results. Primary focus for validation: '{latest_user_input_for_validation[:200]}...'. Original Q: '{original_question}'")
    else:
        logger(f"Validating {len(search_results_info)} search results. Focus for validation: Original Q: '{original_question}'")

    llm = ChatOpenAI(model="gpt-4o", temperature=0.1, openai_api_key=OPENAI_API_KEY)
    structured_llm = llm.with_structured_output(LinkRelevance, method="json_mode")

    validated_urls_to_fetch: List[str] = []
    MIN_RELEVANCE_CONFIDENCE = 0.5 

    for res_info in search_results_info[:max_results_to_validate]:
        url = res_info.get("url")
        title = res_info.get("title", "N/A")
        snippet = res_info.get("snippet", "N/A")

        if not url:
            continue

        truncated_snippet = snippet[:500] if snippet else "N/A"

        prompt_for_validation = f"""You are a relevance assessment expert for a troubleshooting AI agent.

Context for Relevance Assessment:
{validation_focus_statement}

Recent Conversation Snippet (if available, for broader context of interaction flow):
---
{history_summary_for_validation if history_summary_for_validation else "N/A"}
---

Current Search Result to Evaluate:
URL: {url}
Title: {title}
Snippet: {truncated_snippet}

Based on the "Context for Relevance Assessment" (especially the LATEST user information if provided), is this search result LIKELY to contain information that DIRECTLY HELPS ADDRESS THE CURRENT FOCUS of the troubleshooting?

Output a JSON object with the exact schema:
{{
  "url": "{url}",
  "is_relevant": boolean,
  "reasoning": "string",
  "confidence": float
}}
"""
        try:
            if len(prompt_for_validation) > 20000: 
                 logger(f"Warning: Validation prompt for {url} is very long ({len(prompt_for_validation)} chars). May need to shorten history summary further.")
            
            validation_obj: LinkRelevance = structured_llm.invoke(prompt_for_validation) # type: ignore
            logger(f"Validation for {validation_obj.url} - Relevant: {validation_obj.is_relevant}, Conf: {validation_obj.confidence:.2f}, Reasoning: {validation_obj.reasoning}")
            
            if validation_obj.is_relevant and validation_obj.confidence >= MIN_RELEVANCE_CONFIDENCE:
                validated_urls_to_fetch.append(validation_obj.url)
            elif validation_obj.is_relevant: 
                 logger(f"Link {validation_obj.url} assessed relevant but confidence ({validation_obj.confidence:.2f}) is below threshold ({MIN_RELEVANCE_CONFIDENCE}). Discarding.")
        except Exception as e:
            logger(f"Error validating link {url}: {e}. Skipping this link.")
    
    logger(f"Validation complete. {len(validated_urls_to_fetch)} of {min(len(search_results_info), max_results_to_validate)} validated links will be fetched.")
    
    if not validated_urls_to_fetch and search_results_info:
        logger("Warning: No search results passed validation. This might indicate queries were off-topic for current conversation focus, or validation is too strict/failed.")

    return validated_urls_to_fetch

def extract_main_text(url, max_chars=3000, logger=None):
    logger = logger or (lambda msg: print(f"DEBUG: {msg}"))
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
        return text[:max_chars]
    except requests.exceptions.RequestException as e:
        logger(f"HTTP Error fetching {url}: {e}")
        return f"[Could not fetch content from {url}: Network Error {e}]"
    except Exception as e:
        logger(f"Error processing {url}: {e}")
        return f"[Could not process content from {url}: {e}]"

def summarize_text(text: str, logger: Callable[[str], None], max_tokens: int = 400, focus: str = "") -> str:
    snippet = text[:10000]

    # Build the summarization prompt. If a focus/question is provided, instruct the LLM to emphasise
    # information relevant to that focus, otherwise perform a generic technical condense.
    if focus:
        prompt = f"""You are condensing technical NVIDIA (or general) documentation.

Text (partial if very long):
{snippet}

---
Create 6-12 concise bullets that are MOST RELEVANT to the following question/context:
"{focus}"

When selecting which facts to keep, prioritise details that could directly help answer or troubleshoot the focus above.
If the text contains unrelated sections, omit them from the bullets.
"""
    else:
        prompt = f"""You are condensing technical NVIDIA (or general) documentation into a short, information-dense bullet list.

Text to condense (partial if very long):
{snippet}

---
Create 6-12 concise bullets capturing root causes, symptoms, error codes, commands or fixes present in the text."""
    try:
        # Using the module-level client, or create one specifically for ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o", temperature=0.2, openai_api_key=OPENAI_API_KEY)
        response = llm.invoke(prompt) # Direct invoke if not using structured output for this
        summary = response.content.strip() if hasattr(response, 'content') else str(response).strip()


        # Fallback if response is not as expected (e.g. if it's not AIMessage)
        if not isinstance(summary, str) or not summary: # Check if summary is non-empty string
            # Attempt to get from create method if invoke structure differs
            openai_client_direct = openai.OpenAI(api_key=OPENAI_API_KEY)
            response_direct = openai_client_direct.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            summary = response_direct.choices[0].message.content.strip()

        logger(f"Summarization produced {len(summary)} chars.")
        return summary
    except Exception as e:
        logger(f"Summarization error: {e}. Using truncated text instead.")
        # If error, check if client.chat.completions.create was intended.
        # The original code used client.chat.completions.create, let's stick to that for consistency
        try:
            openai_client_direct = openai.OpenAI(api_key=OPENAI_API_KEY)
            response_direct = openai_client_direct.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            summary = response_direct.choices[0].message.content.strip()
            logger(f"Summarization (fallback path) produced {len(summary)} chars.")
            return summary
        except Exception as e_fallback:
            logger(f"Summarization error (fallback path): {e_fallback}. Using truncated text.")
            return text[:3000]


def chunked_summarize(text: str, logger: Callable[[str], None], focus: str = "") -> str:
    CHUNK_SIZE = 30000
    OVERLAP = 5000
    if len(text) <= CHUNK_SIZE:
        return summarize_text(text, logger, 400, focus)

    chunk_list: List[str] = [text[start : start + CHUNK_SIZE] for start in range(0, len(text), CHUNK_SIZE - OVERLAP)]
    logger(f"Dispatching {len(chunk_list)} chunk summarization jobs in parallel…")

    partial_summaries: List[Tuple[int, str]] = []
    # Using ThreadPoolExecutor requires summarize_text to be picklable if running in true parallelism (it should be)
    from concurrent.futures import ThreadPoolExecutor, as_completed # Local import if not at top
    
    with ThreadPoolExecutor(max_workers=min(6, len(chunk_list))) as pool:
        future_map = {pool.submit(summarize_text, chunk, logger, 400, focus): idx for idx, chunk in enumerate(chunk_list)}
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                summary_res = future.result()
                partial_summaries.append((idx, summary_res))
            except Exception as e:
                logger(f"Summarization failed for chunk {idx+1}: {e}")
                partial_summaries.append((idx, "")) # Append empty string for failed chunks to maintain order

    partial_summaries.sort(key=lambda t: t[0])
    combined = "\n".join([s for _, s in partial_summaries if s])

    if len(combined) > CHUNK_SIZE: # Re-check CHUNK_SIZE, not a different one like 8000
        logger("Running reduce summarization on combined chunks…")
        combined = summarize_text(combined, logger, 400, focus)
    return combined 