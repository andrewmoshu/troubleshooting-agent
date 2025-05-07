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

st.set_page_config(page_title="🌼 Daisy - Troubleshooting Assistant", layout="wide")

st.title("🌼 Daisy - The ReAct Troubleshooting Assistant")
st.caption("Powered by LangGraph and Streamlit. I can use Google Search and ask you questions to help solve problems.")

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

if "full_event_log" not in st.session_state:
    st.session_state.full_event_log = [] # List to store all event chunks for debugging

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
    # This is crucial if the Streamlit app might rerun scripts or have multiple users (though globals are tricky)
    daisy_agent.thread_id_for_tools = st.session_state.session_id
    if st.session_state.session_id not in daisy_agent.knowledge_log_store:
        daisy_agent.knowledge_log_store[st.session_state.session_id] = []

    st.session_state.full_event_log.append(f"\n--- User Query: {prompt} ---")

    # Variable to store search results from the current turn
    current_turn_search_results = None

    # Stream agent response
    with st.chat_message("assistant", avatar="🌼"):
        message_placeholder = st.empty() # For streaming the main response
        full_response = ""
        
        thinking_area = st.expander("🧠 Daisy's Thought Process (Live Events)", expanded=False)
        event_log_placeholder = thinking_area.empty()
        current_event_text = ""

        last_printed_agent_message_for_ui = "" # To avoid duplicate UI updates for same message

        try:
            for chunk in daisy_agent.troubleshooting_agent_executor.stream(agent_inputs, config=config, stream_mode="debug"):
                event_type = chunk.get('type', 'N/A')
                event_step = chunk.get('step', 'N/A')
                chunk_info_header = f"\n--- Agent Event Chunk ({event_type} - Step {event_step}) ---"
                st.session_state.full_event_log.append(chunk_info_header)
                current_event_text += chunk_info_header + "\n"
                
                # More detailed logging for debugging in the expander
                # import pprint
                # formatted_chunk = pprint.pformat(chunk, indent=2, depth=3)
                # st.session_state.full_event_log.append(formatted_chunk)
                # current_event_text += formatted_chunk + "\n"

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
                        
                        # Log tool intentions from AIMessage with tool_calls
                        if hasattr(last_message_obj, 'tool_calls') and last_message_obj.tool_calls:
                            for tool_call in last_message_obj.tool_calls:
                                tool_intent_log = f"DEBUG TOOL INTENTION: Agent plans to call tool: {tool_call['name']} with args: {tool_call['args']}"
                                st.session_state.full_event_log.append(tool_intent_log)
                                current_event_text += tool_intent_log + "\n"
                
                elif event_type == "task_result" and chunk.get("payload") and chunk["payload"].get("name") == "tools" and chunk["payload"].get("result"):
                    tool_results_payload = chunk["payload"]["result"]
                    if isinstance(tool_results_payload, list) and tool_results_payload and isinstance(tool_results_payload[0], tuple) and tool_results_payload[0][0] == "messages":
                        for tool_message in tool_results_payload[0][1]: # tool_message is a ToolMessage
                            if hasattr(tool_message, 'type') and tool_message.type == "tool" and hasattr(tool_message, 'content'):
                                # Log the truncated output to the debug stream
                                tool_output_log = f"DEBUG TOOL OUTPUT: Tool: {tool_message.name}, Output: {str(tool_message.content)[:300]}..."
                                st.session_state.full_event_log.append(tool_output_log)
                                current_event_text += tool_output_log + "\n"

                                # If it's the web_search tool, store the full content
                                if tool_message.name == "web_search":
                                    current_turn_search_results = tool_message.content
                else:
                    # Fallback for other event types or structures if needed for debugging
                    # raw_chunk_log = f"DEBUG RAW CHUNK: {str(chunk)[:300]}..."
                    # st.session_state.full_event_log.append(raw_chunk_log)
                    # current_event_text += raw_chunk_log + "\n"
                    pass # Avoid printing too much raw data to the live expander by default

                if agent_response_content:
                    full_response = agent_response_content # We only display the final response for a turn
                    message_placeholder.markdown(full_response + "▌") # Simulate typing
                
                # Update the content *within* the placeholder, no key needed here as it's the same widget container
                event_log_placeholder.text_area("Live Event Log", current_event_text, height=300)
                # time.sleep(0.05) # Small delay if needed to make streaming more visible

            message_placeholder.markdown(full_response) # Final response without cursor
            if full_response: # Add only if there was a response
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.full_event_log.append(f"\n--- Agent Final Response for Turn: {full_response} ---")

            # Display full search results if they were captured in this turn
            if current_turn_search_results:
                with st.container(border=True):
                    st.markdown("**🔎 Web Search Results Used in This Turn:**")
                    st.text_area("Search Results", current_turn_search_results, height=250, disabled=True, key=f"search_results_{time.time()}")
        
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.session_state.full_event_log.append(f"\n--- ERROR: {error_message} ---")

# Sidebar for full debug log
st.sidebar.title("Full Event Log")
st.sidebar.text_area("Complete Debug Stream", "\n".join(st.session_state.full_event_log), height=600, key="sidebar_log")

# For the knowledge log (it's global in daisy_agent module)
if st.sidebar.button("Show/Refresh Knowledge Log"):
    current_thread_id = st.session_state.session_id
    if current_thread_id in daisy_agent.knowledge_log_store and daisy_agent.knowledge_log_store[current_thread_id]:
        st.sidebar.subheader(f"Knowledge Log for Session: ...{current_thread_id[-12:]}")
        log_content = "\n".join(daisy_agent.knowledge_log_store[current_thread_id])
        st.sidebar.text_area("Log Entries", log_content, height=300, key=f"kl_{current_thread_id}")
    else:
        st.sidebar.info(f"Knowledge log for session ...{current_thread_id[-12:]} is empty.") 