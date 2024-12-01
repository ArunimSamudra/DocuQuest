import time

import webview
import subprocess

# Paths to your scripts
FASTAPI_SCRIPT = "server.src.main.app.py"  # Path to your FastAPI script
STREAMLIT_SCRIPT = "client/src/main/ui.py"  # Path to your Streamlit script

fastapi_process = subprocess.Popen(
    ["PYTHONPATH=server/src", "uvicorn", FASTAPI_SCRIPT.replace(".py", ":app"), "--host", "127.0.0.1", "--port",
     "8080"])

time.sleep(5)

streamlit_process = subprocess.Popen(["streamlit", "run", STREAMLIT_SCRIPT, "--server.headless", "true"])

try:
    # Open the Streamlit app in a desktop window
    webview.create_window("DocuQuest", "http://localhost:8501")
    webview.start()
finally:
    # Terminate both processes when the desktop app is closed
    fastapi_process.terminate()
    streamlit_process.terminate()
