# DocuQuest

This application provides a streamlined solution for utilizing local resources alongside powerful cloud-based language models, designed specifically for macOS systems. By leveraging the MLX library for GPU acceleration, it ensures fast and efficient performance for tasks like summarization, question answering, and more.

## Description

The application bridges the gap between local and cloud-based machine learning, making use of macOS's MLX library for enhanced GPU performance. It is tailored for document processing and analysis tasks, such as:
- **Summarization:** Generate concise summaries of documents.
- **Question Answering:** Ask specific questions about uploaded documents.
- **Hybrid Resource Utilization:** Use local resources for lightweight tasks and offload complex tasks to OpenAI's API.

This hybrid approach balances performance and cost while maintaining high-quality results. The application is designed with a user-friendly desktop interface for seamless interaction.

## Prerequisites
- macOS system
- Python 3.x installed
- OpenAI API Key

## Setup Instructions

### Step 1: Clone the Repository
Clone this repository to your local machine:
```bash
git clone <repository_url>
```

### Step 2: Navigate to the Local Directory
Move into the `local` directory:
```bash
cd local
```

### Step 3: Create and Activate a Virtual Environment
Create a virtual environment:
```bash
python3 -m venv venv
```
Activate the virtual environment:
```bash
# For macOS/Linux
source venv/bin/activate
```

### Step 4: Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### Step 5: Download the LLaMA 3.2 1B Model
Download the LLaMA 3.2 1B model from Hugging Face:

[https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)

You can download the model using one of the following methods:

#### Option 1: Direct Download
Manually download the tensor files from the Hugging Face website and place them in the following directory:
```
local/server/src/main/models/llm
```

#### Option 2: Using the Hugging Face CLI
Install the Hugging Face CLI if not already installed:
```bash
pip install huggingface_hub
```
Log in to Hugging Face:
```bash
huggingface-cli login
```
Download the model:
```bash
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --repo-type model -d local/server/src/main/models/llm
```

### Step 6: Configure the OpenAI API Key
Open the `config.py` file located at `/local/server/src/main/config.py` and add your OpenAI API key:
```python
OPENAI_API_KEY = "your_openai_api_key_here"
```

### Step 7: Run the Application
From the terminal, execute the following command:
```bash
python3 desktop.py
```

## Usage
The application provides a graphical interface where users can upload documents, perform summarization, or ask specific questions. The hybrid architecture ensures efficient processing by using local GPU resources and cloud-based LLMs as needed.

## Notes
- The application uses the MLX library for GPU acceleration, which is optimized for macOS devices.
- Ensure that your OpenAI API key is valid and properly set in the configuration file.
- The LLaMA 3.2 1B model must be downloaded and stored in the correct directory for local processing to work.