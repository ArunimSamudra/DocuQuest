import streamlit as st
import random
import time
import requests
from config import Config

st.title("DocuQuest")

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

# Generator to stream custom static responses
def text_streamer(text):
    for word in text.split():
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
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            # Call FastAPI to summarize the PDF
            response = requests.post(
                f"{Config.SERVER_URL}/summarize",
                files={"file": uploaded_file.getvalue()},
                data={"file_type": "pdf"}  # Correctly passing the flag for file type
            )
        elif uploaded_file.type == "text/plain":
            # Read the TXT file content
            txt_content = uploaded_file.read().decode("utf-8")
            # Send the content directly to FastAPI for summarization
            response = requests.post(
                f"{Config.SERVER_URL}/summarize",
                data={"text": txt_content, "file_type": "txt"}  # Use json for text and file_type
            )
        else:
            with st.chat_message("assistant"):
                st.write(text_streamer("Please upload compatible version"))
            chat_history.append({"role": "assistant", "content": "Please upload compatible version"})

        if response is not None and response.status_code == 200:
            summary = response.json().get("summary")
            chat_history.append({"role": "assistant", "content": summary})
            with st.chat_message("assistant"):
                st.write(text_streamer(summary))
        else:
            st.error("Error summarizing the PDF.")

    # Accept user input and handle messages
    if prompt := st.chat_input("Ask a question about the document:"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        chat_history.append({"role": "user", "content": prompt})

        # Assuming the summary is available from the uploaded file
        if uploaded_file is not None:
            # Call the ask endpoint to get the answer
            ask_response = requests.post(
                f"{Config.SERVER_URL}/ask", json={"question": prompt}
            )
            if ask_response.status_code == 200:
                answer = ask_response.json().get("answer")
                with st.chat_message("assistant"):
                    st.write_stream(answer)
                chat_history.append({"role": "assistant", "content": answer})
            else:
                st.error("Error answering the question.")
        else:
            with st.chat_message("assistant"):
                st.write(text_streamer("Please upload a file first"))
            chat_history.append({"role": "assistant", "content": "Please upload a file first"})






