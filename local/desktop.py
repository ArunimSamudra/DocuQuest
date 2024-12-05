import os
import time

import webview
import subprocess

if __name__=="__main__":
    # Paths to your scripts
    FASTAPI_SCRIPT = "main.app.py"  # Path to your FastAPI script
    STREAMLIT_SCRIPT = "client/src/main/ui.py"  # Path to your Streamlit script

    # Define the PYTHONPATH
    pythonpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server/src")

    # Set up the environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath

    # Command to run
    command = [
        "uvicorn",
        "main.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8080"
    ]

    # Run the subprocess
    fastapi_process = subprocess.Popen(command, env=env)

    time.sleep(2)

    streamlit_process = subprocess.Popen(["streamlit", "run", STREAMLIT_SCRIPT, "--server.headless", "true"])

    try:
        # Open the Streamlit app in a desktop window
        webview.create_window("DocuQuest", "http://localhost:8501")
        webview.start()
    finally:
        # Terminate both processes when the desktop app is closed
        fastapi_process.terminate()
        streamlit_process.terminate()