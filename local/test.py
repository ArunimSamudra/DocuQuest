import streamlit as st
import random
import time
import requests

# FastAPI backend URL
FASTAPI_URL = "http://127.0.0.1:8000"

st.title("Chat with Session Management")

# Initialize the session state if not already initialized
if "sessions" not in st.session_state:
    st.session_state.sessions = {}
if "current_session" not in st.session_state:
    st.session_state.current_session = None


# Response generator for the assistant
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

# Sidebar: Session management
with st.sidebar:
    st.header("Manage Sessions")

    # Create new session
    if st.button("Create New Session"):
        session_id = f"Session {len(st.session_state.sessions) + 1}"
        st.session_state.sessions[session_id] = []  # Empty message list for the new session
        st.session_state.current_session = session_id

    # Select session from dropdown
    session_id = st.selectbox(
        "Select Session",
        options=list(st.session_state.sessions.keys()),
        index=0 if st.session_state.sessions else None
    )

    # Set the current session to the selected session
    if session_id:
        st.session_state.current_session = session_id

    # Option to delete the current session
    if st.session_state.current_session and st.button("Delete Current Session"):
        del st.session_state.sessions[st.session_state.current_session]
        st.session_state.current_session = None if not st.session_state.sessions else \
        list(st.session_state.sessions.keys())[0]

# Display the selected session's chat history
if st.session_state.current_session:
    st.subheader(f"Chat - {st.session_state.current_session}")

    # Retrieve the message history for the current session
    chat_history = st.session_state.sessions[st.session_state.current_session]

    # Display chat messages from history on app rerun
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle file upload and summarization
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file is not None:
        # Call FastAPI to summarize the PDF
        response = requests.post(
            f"{FASTAPI_URL}/summarize", files={"file": uploaded_file}
        )
        if response.status_code == 200:
            summary = response.json().get("summary")
            chat_history.append({"role": "assistant", "content": summary})
            with st.chat_message("assistant"):
                st.markdown(summary)
        else:
            st.error("Error summarizing the PDF.")

    # Accept user input and handle messages
    if prompt := st.chat_input("What is up?"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        chat_history.append({"role": "user", "content": prompt})

        # Generate assistant response
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator())
        # Add assistant response to chat history
        chat_history.append({"role": "assistant", "content": response})



