import streamlit as st
from dotenv import load_dotenv
load_dotenv()
# from nvidia_langgraph_agent import run_agent # Old import
# from nvidia_langgraph_agent import AgentState # Old import
from main_agent import run_agent # New import from main_agent.py
from agent_state import AgentState # AgentState is now in its own file

st.set_page_config(page_title="Daisy - NVIDIA Q&A Agent", page_icon="🌼", layout="centered")

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Each item: {role, content, sources, logs, mode}
if "needs_clarification" not in st.session_state:
    st.session_state.needs_clarification = False
if "clarification_state" not in st.session_state:
    st.session_state.clarification_state = None

# --- UI Setup ---

# Optional: Add a logo or header image (replace with your own if desired)
st.image("https://upload.wikimedia.org/wikipedia/commons/2/21/Nvidia_logo.svg", width=180)
st.markdown("""
# 🌼 Daisy: Your NVIDIA Documentation Q&A Agent

Use **Answer Mode** for direct questions or **Troubleshoot Mode** for step-by-step problem analysis. Daisy searches NVIDIA sources like <span style='color:#76B900'>docs.nvidia.com</span>, <span style='color:#76B900'>forums</span>, and <span style='color:#76B900'>knowledge base articles</span>.
""", unsafe_allow_html=True)

st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    # Disable mode change if clarification is pending
    agent_mode = st.radio("Select Mode:", ("Answer", "Troubleshoot"), horizontal=True, key="agent_mode",
                          disabled=st.session_state.needs_clarification)
with col2:
    search_engine = st.selectbox("Choose search engine:", ["duckduckgo", "google"], format_func=lambda x: "DuckDuckGo" if x=="duckduckgo" else "Google")
with col3:
    sources = st.multiselect(
        "Select sources to search:",
        ["docs.nvidia.com", "catalog.ngc.nvidia.com", "nvidia.custhelp.com", "forums.developer.nvidia.com", "developer.nvidia.com/blog"],
        default=["docs.nvidia.com", "nvidia.custhelp.com", "forums.developer.nvidia.com"]
    )

# --- Chat History Display ---

# Daisy's welcome message or clarification prompt
if not st.session_state.chat_history and not st.session_state.needs_clarification:
    with st.chat_message("assistant", avatar="🌼"):
        st.markdown("""
        **Hi, I'm Daisy!** 🌼

        Choose a mode above:
        *   **Answer Mode:** For specific questions (e.g., "What is NVLink?").
        *   **Troubleshoot Mode:** For problems (e.g., "My GPU is overheating..."). I'll try to analyze the issue based on documentation.

        _Tip: You can change the search engine and sources used._
        """)

# Render chat history
for msg in st.session_state.chat_history:
    avatar = "🌼" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"], unsafe_allow_html=True)
        # Only show expanders for assistant messages that aren't clarification prompts
        if msg["role"] == "assistant" and not msg.get("is_clarification_prompt", False):
            mode_used = msg.get("mode", "Answer")
            if msg.get("sources"):
                with st.expander("Sources used (click to expand)"):
                    for url in msg["sources"]:
                        st.markdown(f"- [🔗 {url}]({url})")
            if msg.get("logs"):
                with st.expander("Step-by-step logs (click to expand)"):
                    for log in msg["logs"]:
                        st.write(log)

# --- Chat Input and Agent Logic ---

# Display clarification request if needed
if st.session_state.needs_clarification and st.session_state.clarification_state:
    clarification_q = st.session_state.clarification_state.get('clarification_question', "Could you please provide more details?")
    st.info(f"**🌼 Daisy needs more information:** {clarification_q}", icon="❓")

# Use a different placeholder text when waiting for clarification
placeholder = "Please provide the requested clarification..." if st.session_state.needs_clarification else "Ask Daisy a question about NVIDIA..."
user_input = st.chat_input(placeholder)

if user_input:
    # Determine mode string for run_agent
    mode_str = agent_mode.lower()
    output_label = "Report" if mode_str == "troubleshoot" else "Answer"

    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Agent processing message
    with st.chat_message("assistant", avatar="🌼"):
        logs = []
        def logger(msg):
            print(f"AGENT_LOG: {msg}") # Print log to terminal
            logs.append(msg)

        with st.spinner("Daisy is thinking..."):
            agent_result = None
            if st.session_state.needs_clarification and st.session_state.clarification_state:
                # --- Resume with clarification ---
                logger("Resuming agent with user clarification...")
                resuming_state: AgentState = st.session_state.clarification_state
                # Add user's clarification to the state for the next analysis step
                resuming_state["user_feedback"] = user_input
                resuming_state["history"] = resuming_state.get("history", []) + [{
                    "role": "user",
                    "content": f"(Clarification Provided) {user_input}"
                }]

                agent_result = run_agent(
                    question="", # Not needed when resuming
                    mode="troubleshoot", # Must be troubleshoot mode if resuming
                    logger=logger,
                    search_engine=search_engine, # Keep original settings
                    sources=sources,
                    history=resuming_state["history"], # Pass updated history
                    resume_from_state=resuming_state
                )
                # Clear clarification state after resuming
                st.session_state.needs_clarification = False
                st.session_state.clarification_state = None
            else:
                # --- Start new query ---
                agent_result = run_agent(
                    question=user_input,
                    logger=logger,
                    mode=mode_str,
                    search_engine=search_engine,
                    sources=sources,
                    history=st.session_state.chat_history[:-1] # History before this turn
                )

        # --- Process Agent Result ---
        final_output = None
        used_sources = []
        assistant_message = ""
        is_clarification_prompt = False

        if agent_result and agent_result["status"] == "complete":
            final_output = agent_result["output"]
            used_sources = agent_result["sources"]
            assistant_message = f"<span style='color:#76B900;font-size:1.2em'><b>🌼 Daisy's {output_label}:</b></span>\n\n{final_output}"
            st.markdown(assistant_message, unsafe_allow_html=True)
            if used_sources:
                with st.expander("Sources used (click to expand)"):
                    for url in used_sources:
                        st.markdown(f"- [🔗 {url}]({url})")
            # Show aggregated context if user wants it
            if agent_result.get("context"):
                with st.expander("Full aggregated context (click to expand)"):
                    st.markdown(agent_result["context"][:35000] + ("…" if len(agent_result["context"])>35000 else ""))
            with st.expander("Step-by-step logs (click to expand)"):
                for log in logs:
                    st.write(log)

        elif agent_result and agent_result["status"] == "needs_clarification":
            st.session_state.needs_clarification = True
            st.session_state.clarification_state = agent_result["state"]
            clarification_q = st.session_state.clarification_state.get('clarification_question', "Could you please provide more details?")
            assistant_message = f"**🌼 Daisy needs more information:** {clarification_q}"
            is_clarification_prompt = True
            st.info(assistant_message, icon="❓") # Display prompt clearly
             # Also add to history
            st.session_state.chat_history.append({
                 "role": "assistant",
                 "content": assistant_message,
                 "is_clarification_prompt": True, # Flag this message
                 "logs": logs # Log up to the point of clarification
             })
            st.rerun() # Rerun to update UI state immediately

        elif agent_result and agent_result["status"] == "explanation_provided":
            explanation = agent_result["explanation"]
            assistant_message = f"**🌼 Okay, let me explain:**\n\n{explanation}\n\n*(Now, please answer the original clarification request based on this explanation.)*"
            # Display the explanation but keep waiting for the original clarification
            st.markdown(assistant_message, unsafe_allow_html=True)
            # Add explanation to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": assistant_message,
                "is_clarification_prompt": False, # It's an explanation, not a prompt
                "logs": logs
            })
            # Keep the needs_clarification state True and preserve the original clarification state
            st.session_state.needs_clarification = True
            st.session_state.clarification_state = agent_result["state"]
            st.rerun() # Rerun to show explanation and keep input active

        elif agent_result and agent_result["status"] == "error":
            assistant_message = f"😕 Sorry, an error occurred: {agent_result['output']}"
            st.error(assistant_message)
        else:
             assistant_message = "😕 Sorry, something went wrong and I couldn't get a response."
             st.error(assistant_message)

        # Add final response (or error) to history, but skip if it was a clarification prompt (already added)
        if not is_clarification_prompt:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": assistant_message,
                "sources": used_sources,
                "logs": logs,
                "mode": agent_mode # Store mode used for this interaction
            })
            # Ensure the log expander is consistently available for non-clarification messages
            if logs:
                with st.expander("Step-by-step logs (click to expand)", expanded=False):
                    st.write(logs)
            if used_sources:
                with st.expander("Sources used (click to expand)", expanded=False):
                    for url in used_sources:
                        st.markdown(f"- [🔗 {url}]({url})")
            if agent_result.get("context"):
                with st.expander("Full aggregated context (click to expand)", expanded=False):
                    st.markdown(agent_result["context"][:35000] + ("…" if len(agent_result["context"])>35000 else ""))

    # Rerun if clarification was needed to ensure UI reflects the waiting state
    # (This might already be handled by the rerun within the elif block, but belts and suspenders)
    # if st.session_state.needs_clarification:
    #     st.rerun() 