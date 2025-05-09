import streamlit as st
import uuid
import time

# Import the agent executor and any other necessary components 
# from your existing react_troubleshooting_agent.py file.
# We need to ensure that file can be imported as a module.

# To make the global variables in react_troubleshooting_agent.py work per session,
# we might need to modify it slightly or handle it carefully here.
# For now, we'll try to set the global `thread_id_for_tools` from this app.
import react_troubleshooting_agent as daisy_agent 
from parse_search_results import parse_search_results # Import the new parser

st.set_page_config(page_title="🌼 Daisy - Troubleshooting Assistant", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS ---
st.markdown("""
    <style>
        /* Base styling */
        body {
            font-family: 'Roboto', sans-serif; /* A more modern, clean font */
            background-color: #f0f2f6; /* Light grey background */
        }
        .stApp {
            background-color: #f0f2f6;
        }

        /* Titles and Headers */
        h1, h2, h3 {
            color: #333; /* Darker text for headers */
        }
        .st-emotion-cache-10trblm { /* Main title */
            color: #2c3e50; /* Dark blue-grey */
            text-align: center;
            padding-bottom: 20px; /* Existing padding */
            padding-top: 30px; /* Added padding top */
            margin-bottom: 0; /* Remove default h1 margin if any */
        }
         .st-emotion-cache-79elbk { /* Main title emoji*/
            font-size: 2.5rem !important;
        }

        /* Custom caption styling */
        .stCaption {
            text-align: center;
            color: #555; /* Slightly lighter text for caption */
            font-size: 0.95rem; /* Adjust caption font size */
            padding-bottom: 20px; /* Space below caption */
            margin-top: -10px; /* Pull caption closer to title */
        }

        /* Hide Streamlit default header/toolbar */
        header[data-testid="stHeader"] {
            display: none !important;
            visibility: hidden !important;
        }
        /* Fallback for other versions/structures - more aggressive */
        .stApp > header {
            display: none !important;
            visibility: hidden !important;
        }
        /* Hide the hamburger menu specifically if still visible */
        #MainMenu {
            display: none !important;
            visibility: hidden !important;
        }
        /* Hide the top decoration bar if present */
        [data-testid="stDecoration"] {
            display: none !important;
            visibility: hidden !important;
        }

        /* Chat messages */
        .st-emotion-cache-1c7y2kd { /* User message container */
            background-color: #dcf8c6; /* Light green for user */
            border-radius: 15px 15px 0 15px;
            padding: 10px 15px;
            margin-bottom: 10px;
            border: 1px solid #c5e7b3;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            align-self: flex-end; /* Align user messages to the right */
            max-width: 70%;
        }
        .st-emotion-cache-4oy321 { /* Assistant message container */
            background-color: transparent; /* MODIFIED: No background color */
            border-radius: 15px 15px 15px 0;
            padding: 10px 15px; /* Padding kept for structure */
            margin-bottom: 10px;
            border: none; /* MODIFIED: No border */
            box-shadow: none; /* MODIFIED: No shadow */
            align-self: flex-start; /* Align assistant messages to the left */
            width: 80%; /* MODIFIED: Explicitly set width */
            max-width: 80%; /* Keep max-width as a fallback/consistency */
            box-sizing: border-box; /* Added for better width calculation */
        }
        
        /* Make chat messages distinct */
        [data-testid="chat-message-container"] > div:nth-child(1) { /* User message */
            background-color: #e1f5fe; /* Light blue */
            border-radius: 12px 12px 0 12px;
            padding: 12px 18px;
            margin: 5px 0;
            border: 1px solid #b3e5fc;
        }
        [data-testid="chat-message-container"] > div:nth-child(2) { /* Assistant message, assuming only two children in the container structure */
             /* This selector might be too broad or incorrect. Need to inspect actual structure.
                Let's assume the default streamlit chat_message structure for now.
                The provided selectors .st-emotion-cache-1c7y2kd and .st-emotion-cache-4oy321 are more robust if they target the specific message blocks.
             */
        }

        /* Chat input */
        .stChatInputContainer {
            background-color: #f8f9fa;
            border-top: 1px solid #e6e9ec;
            padding: 20px 15px;
            box-shadow: 0 -4px 10px rgba(0,0,0,0.03);
            margin-top: 20px;
        }
        
        textarea[data-testid="stChatInput"],
        .st-emotion-cache-13k62yr {
            border-radius: 24px !important;
            border: 1px solid #e0e0e0 !important;
            padding: 14px 20px !important;
            font-size: 16px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
            transition: all 0.2s ease-in-out !important;
            background-color: white !important;
        }
        
        textarea[data-testid="stChatInput"]:focus,
        .st-emotion-cache-13k62yr:focus {
            border-color: #3498db !important;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2) !important;
            outline: none !important;
        }
        
        .stButton>button { /* Send button */
            background-color: #3498db; /* Blue */
            color: white;
            border-radius: 20px;
            border: none;
            padding: 10px 15px;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #2980b9; /* Darker blue on hover */
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }

        /* Expanders and Containers */
        .stExpander {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            margin-bottom: 15px;
            background-color: #ffffff; /* Added background to expander body */
        }
        .stExpander header {
            background-color: #f9f9f9; /* Lighter background for expander header */
            border-radius: 8px 8px 0 0;
            padding: 10px 15px;
            border-bottom: 1px solid #e0e0e0; /* Separator line for header */
        }
        .st-emotion-cache-vj1c9o { /* Container with border for search results */
             border: 1px solid #d1d5db;
             border-radius: 8px;
             padding: 15px;
             background-color: #ffffff;
             box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        
        /* Sidebar styling */
        .st-emotion-cache-16txtl3 { /* Sidebar main area */
            background-color: #e9ecef; /* Lighter grey for sidebar */
            padding: 15px;
        }
        .st-emotion-cache-16txtl3 h1, 
        .st-emotion-cache-16txtl3 h2, 
        .st-emotion-cache-16txtl3 h3, 
        .st-emotion-cache-16txtl3 .stMarkdown p { /* Target paragraph text in sidebar Markdown */
            color: #343a40 !important; /* Darker text in sidebar, ensure override */
        }
        .st-emotion-cache-16txtl3 .stButton>button {
             background-color: #5dade2;
             color: white;
             width: 100%;
        }
         .st-emotion-cache-16txtl3 .stButton>button:hover {
             background-color: #4a90e2;
        }

        /* Spinner */
        .stSpinner > div > svg {
            fill: #3498db; /* Blue spinner */
        }
        
        /* Text Area for fetched content and logs */
        .stTextArea textarea {
            background-color: #fdfdfd;
            border: 1px solid #e0e0e0;
            font-family: 'Courier New', Courier, monospace; /* Monospace for logs/code-like content */
            font-size: 0.9em;
            color: #333; /* Darker text for readability */
        }
        
        /* Avatars in chat */
        .st-emotion-cache-1c7y2kd span.st-emotion-cache-1tl2luj, /* User avatar */
        .st-emotion-cache-4oy321 span.st-emotion-cache-1tl2luj { /* Assistant avatar */
            font-size: 1.5rem; /* Larger avatar emojis */
            margin-right: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🌼 Daisy - The ReAct Troubleshooting Assistant")
st.caption("Powered by LangGraph and Streamlit. I can use Google Search and ask you questions to help solve problems.")
st.markdown("<hr style='margin-top: 0px; margin-bottom: 25px;'>", unsafe_allow_html=True) # Added HR

# Initialize session state variables
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    # Set the global variable in the imported module for the current session
    daisy_agent.thread_id_for_tools = st.session_state.session_id
    # Initialize knowledge log for this session if it's not already (it's global in the module)
    if st.session_state.session_id not in daisy_agent.knowledge_log_store:
        daisy_agent.knowledge_log_store[st.session_state.session_id] = []

if "messages" not in st.session_state:
    st.session_state.messages = [] # List to store chat messages: {"role": "user"/"assistant", "content": ...}

if "turn_events" not in st.session_state: # For structured display in expander
    st.session_state.turn_events = []

# --- Display chat messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🌼"):
        st.markdown(message["content"])

# --- Agent Interaction --- 
if prompt := st.chat_input("What technical problem can I help you solve today?"):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # Prepare agent input
    agent_inputs = {"messages": [{"role": "user", "content": prompt}]}
    config = {"configurable": {"thread_id": st.session_state.session_id}}
    
    # Ensure the global thread_id in the agent module is set for this session
    daisy_agent.thread_id_for_tools = st.session_state.session_id
    if st.session_state.session_id not in daisy_agent.knowledge_log_store:
        daisy_agent.knowledge_log_store[st.session_state.session_id] = []

    st.session_state.turn_events = [] # Clear events for the new turn

    # Variable to store search results from the current turn
    current_turn_search_results = None

    # Stream agent response
    with st.spinner("🌼 Daisy is working on it..."):
        with st.chat_message("assistant", avatar="🌼"):
            message_placeholder = st.empty() # For streaming the main response
            full_response = ""
            
            thinking_area = st.expander("🧠 Daisy's Thought Process (Live Events)", expanded=False)
            event_log_placeholder = thinking_area.empty()

            last_printed_agent_message_for_ui = "" 

            try:
                for chunk in daisy_agent.troubleshooting_agent_executor.stream(agent_inputs, config=config, stream_mode="debug"):
                    event_type = chunk.get('type', 'N/A')
                    event_step = chunk.get('step', 'N/A')
                    
                    agent_response_content = None

                    if event_type == "checkpoint" and chunk.get("payload") and "values" in chunk["payload"]:
                        agent_state_values = chunk["payload"]["values"]
                        if "messages" in agent_state_values and agent_state_values["messages"]:
                            last_message_obj = agent_state_values["messages"][-1]
                            if hasattr(last_message_obj, 'type') and last_message_obj.type == "ai" and hasattr(last_message_obj, 'content'):
                                current_agent_content = last_message_obj.content
                                if current_agent_content and current_agent_content != last_printed_agent_message_for_ui:
                                    agent_response_content = current_agent_content
                                    last_printed_agent_message_for_ui = current_agent_content
                            
                            if hasattr(last_message_obj, 'tool_calls') and last_message_obj.tool_calls:
                                for tool_call in last_message_obj.tool_calls:
                                    st.session_state.turn_events.append({
                                        "type": "tool_intent",
                                        "name": tool_call['name'],
                                        "args": str(tool_call['args'])
                                    })
                    
                    elif event_type == "task_result" and chunk.get("payload") and chunk["payload"].get("name") == "tools" and chunk["payload"].get("result"):
                        tool_results_payload = chunk["payload"]["result"]
                        if isinstance(tool_results_payload, list) and tool_results_payload and isinstance(tool_results_payload[0], tuple) and tool_results_payload[0][0] == "messages":
                            for tool_message in tool_results_payload[0][1]: 
                                if hasattr(tool_message, 'type') and tool_message.type == "tool" and hasattr(tool_message, 'content'):
                                    tool_output_preview = str(tool_message.content)[:300] + "..."
                                    st.session_state.turn_events.append({
                                        "type": "tool_output",
                                        "name": tool_message.name,
                                        "output_preview": tool_output_preview
                                    })

                                    if tool_message.name == "web_search":
                                        raw_search_results = tool_message.content
                                        current_turn_search_results = parse_search_results(raw_search_results) # Use the parser
                                        # Log if parsing changed the type, for debugging
                                        if type(raw_search_results) != type(current_turn_search_results):
                                            print(f"Search results format changed by parser: {type(raw_search_results)} -> {type(current_turn_search_results)}")
                                        elif isinstance(current_turn_search_results, list) and raw_search_results != current_turn_search_results:
                                            print("Search results content was modified by parser.")
                    else:
                        pass

                    if agent_response_content:
                        full_response = agent_response_content 
                        message_placeholder.markdown(full_response + "▌")
                    
                    with event_log_placeholder.container():
                        if not st.session_state.turn_events:
                            st.caption("Waiting for agent events...")
                        for i, event_item in enumerate(st.session_state.turn_events):
                            if event_item["type"] == "tool_intent":
                                st.markdown(f"🛠️ **Tool Call Planned:** `{event_item['name']}`\n   Arguments: `{event_item['args']}`")
                            elif event_item["type"] == "tool_output":
                                st.markdown(f"⚙️ **Tool Result ({event_item['name']}):**\n   `{event_item['output_preview']}`")
                            if i < len(st.session_state.turn_events) - 1: # Add separator if not the last item
                                st.markdown("---", help=f"Event {i+1}")

                message_placeholder.markdown(full_response) 
                if full_response: 
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                if current_turn_search_results and isinstance(current_turn_search_results, list):
                    with st.container(border=True):
                        st.markdown("##### 🔎 Web Search Results & Summaries") # Made header smaller
                        if not current_turn_search_results or (len(current_turn_search_results) == 1 and current_turn_search_results[0]['title'] in ["No Results", "Error"]):
                            st.info(f"No web results found, or an error occurred: {current_turn_search_results[0]['snippet']}")
                        else:
                            for i, result in enumerate(current_turn_search_results):
                                with st.expander(f"Result {i+1}: {result.get('title', 'N/A')}", expanded=i < 2):
                                    st.markdown(f"**Title:** {result.get('title', 'N/A')}")
                                    st.markdown(f"**Link:** <{result.get('link', 'N/A')}>")
                                    st.markdown(f"**Snippet:** {result.get('snippet', 'N/A')}")
                                    st.text_area(
                                        label="Fetched & Summarized Content:", 
                                        value=result.get('fetched_content_summary', '[No content summary available]'), 
                                        height=200, 
                                        disabled=True, 
                                        key=f"search_content_{st.session_state.session_id}_{len(st.session_state.messages)}_{i}"
                                    )
                elif current_turn_search_results: 
                    with st.container(border=True):
                        st.markdown("<h4 style='color:#f39c12;'>🔎 Web Search Results (Unable to Fully Parse)</h4>", unsafe_allow_html=True)
                        st.info("The search results were received in a format that could not be fully parsed into standard list. Displaying raw data.")
                        # Try to make sense of the data with a fallback display
                        try:
                            if isinstance(current_turn_search_results, str):
                                st.code(current_turn_search_results[:1000] + ("..." if len(current_turn_search_results) > 1000 else ""), language="text")
                            else:
                                st.json(current_turn_search_results, expanded=False)
                        except Exception as format_error:
                            st.error(f"Error displaying raw search results: {str(format_error)}")
                            st.text_area("Raw Output", str(current_turn_search_results), height=150, disabled=True)
            
            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})