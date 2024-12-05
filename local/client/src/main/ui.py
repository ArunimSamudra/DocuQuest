import streamlit as st
import random
import time
import requests

SERVER_URL = "http://127.0.0.1:8080"
st.title("DocuQuest")

# Initialize the session state if not already initialized
if "sessions" not in st.session_state:
    st.session_state.sessions = {}
if "current_session" not in st.session_state:
    st.session_state.current_session = None

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
        session_id = f"session_{len(st.session_state.sessions) + 1}"
        st.session_state.sessions[session_id] = {
            "chat_history": [],
            "file_uploaded": False,
            "count": 0
        }
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
        # Delete the session from the backend
        response = requests.delete(
            f"{SERVER_URL}/session/{st.session_state.current_session}"
        )
        if response.status_code == 200:
            del st.session_state.sessions[st.session_state.current_session]
            st.session_state.current_session = None if not st.session_state.sessions else \
                list(st.session_state.sessions.keys())[0]
            st.session_state.file_uploaded = False
            st.success("Session deleted successfully!")
        else:
            st.error("Error deleting the session on the server.")


# Display the selected session's chat history
if st.session_state.current_session:
    st.subheader(f"Chat - {st.session_state.current_session}")

    # Retrieve the message history for the current session
    chat_history = st.session_state.sessions[st.session_state.current_session]["chat_history"]

    # Display chat messages from history on app rerun
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle file upload and summarization
    print(st.session_state)
    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT file",
        type=["pdf", "txt"],
        disabled=st.session_state.sessions[st.session_state.current_session]['file_uploaded']  # Disable if file is already uploaded
    )
    call_backend = False
    print("length", len(st.session_state.sessions))
    if len(st.session_state.sessions) == 1:
        call_backend = True
    else:
        if st.session_state.sessions[st.session_state.current_session]['count'] != 0:
            call_backend = True
        else:
            st.session_state.sessions[st.session_state.current_session]['count'] = 1

    if call_backend and uploaded_file and not st.session_state.sessions[st.session_state.current_session]['file_uploaded']:
        payload = {"session_id": st.session_state.current_session}
        if uploaded_file.type == "application/pdf":
            # Call FastAPI to summarize the PDF
            payload["file_type"] = "pdf"
            response = requests.post(
                f"{SERVER_URL}/summarize",
                files={"file": uploaded_file.getvalue()},
                data=payload  # Correctly passing the flag for file type
            )
        elif uploaded_file.type == "text/plain":
            # Read the TXT file content
            txt_content = uploaded_file.read().decode("utf-8")
            payload.update({"text": txt_content, "file_type": "txt"})
            # Send the content directly to FastAPI for summarization
            response = requests.post(
                f"{SERVER_URL}/summarize",
                data=payload  # Use json for text and file_type
            )
        else:
            with st.chat_message("assistant"):
                st.write(text_streamer("Please upload a compatible version"))
            chat_history.append({"role": "assistant", "content": "Please upload a compatible version"})

        if response is not None and response.status_code == 200:
            summary = response.json().get("summary")
            chat_history.append({"role": "assistant", "content": summary})
            with st.chat_message("assistant"):
                st.write(text_streamer(summary))
            st.session_state.sessions[st.session_state.current_session]["file_uploaded"] = True
        else:
            st.error("Error summarizing the file.")

    # Accept user input and handle messages
    if prompt := st.chat_input("Ask a question about the document:"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        chat_history.append({"role": "user", "content": prompt})

        # Assuming the summary is available from the uploaded file
        if st.session_state.sessions[st.session_state.current_session]["file_uploaded"]:
            # Call the ask endpoint to get the answer
            payload = {
                "question": prompt,
                "session_id": st.session_state.current_session,
            }
            ask_response = requests.post(f"{SERVER_URL}/ask", json=payload)
            if ask_response.status_code == 200:
                answer = ask_response.json().get("answer")
                with st.chat_message("assistant"):
                    st.write(text_streamer(answer))
                chat_history.append({"role": "assistant", "content": answer})
            else:
                st.error("Error answering the question.")
        else:
            with st.chat_message("assistant"):
                st.write(text_streamer("Please upload a file first"))
            chat_history.append({"role": "assistant", "content": "Please upload a file first"})
