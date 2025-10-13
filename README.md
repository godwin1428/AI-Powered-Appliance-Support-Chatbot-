# AI-Powered Appliance Support Chatbot

This project is a smart, interactive chatbot designed to help users troubleshoot issues with their Samsung refrigerators. It leverages modern AI and machine learning models to provide accurate, context-aware support through both text-based conversation and image analysis.

## ✨ Features

* **🤖 Conversational AI:** Engage in natural, human-like conversations to diagnose problems.
* **📚 RAG-Powered Knowledge:** Provides answers based on an official Samsung refrigerator user manual for high accuracy.
* **📸 Image-Based Troubleshooting:** Upload a photo of a refrigerator part or error code, and the AI will identify it and provide guidance.
* **🧠 Intelligent Model Integration:** Uses powerful models like `Mistral-7B` for generation and `google/gemma-3-4b-it:free` for visual question answering.
* **⚡ Fast and Scalable:** Built with a lightweight Flask backend and vectorized storage using Pinecone for quick document retrieval.
* **🌐 Simple Web Interface:** Easy-to-use interface for seamless user interaction.

---

## 🛠️ How It Works

The application is built with a Python Flask backend and a simple HTML/CSS/JS frontend. It has two primary modes of operation:

### 1. Text-Based Queries (RAG)

When a user sends a text message, the system follows a **Retrieval-Augmented Generation (RAG)** pipeline:
1.  **Embedding:** The user's query is converted into a vector embedding using sentence transformers.
2.  **Retrieval:** The system searches the **Pinecone** vector database to find the most relevant text chunks from the Samsung user manual.
3.  **Generation:** The retrieved chunks and the original query are passed to the **Mistral-7B** language model, which generates a coherent and contextually relevant answer.

### 2. Image-Based Queries (VQA)

When a user uploads an image:
1.  **Image Analysis:** The image is processed by a **Visual Question Answering (VQA)** model (`google/gemma-3-4b-it:free`).
2.  **Contextual Prompting:** The model is prompted with a specific question like "What is in this image?" or "What error is shown on this display?".
3.  **Answer Generation:** The model provides a text-based description or answer based on the visual content of the image.

---

## 🚀 Tech Stack

* **Backend:** Python, Flask
* **Frontend:** HTML, CSS, JavaScript
* **AI/ML Models:**
    * **LLM:** `mistralai/Mistral-7B-Instruct-v0.2` (via Hugging Face)
    * **VQA:** `google/gemma-3-4b-it:free` (via OpenRouter)
    * **Embedding:** `sentence-transformers/all-MiniLM-L6-v2`
* **Vector Database:** Pinecone
* **PDF Processing:** LangChain, PyPDFLoader
* **Deployment:** (Your deployment platform, e.g., Heroku, AWS, etc.)

---

## ⚙️ Setup and Installation

Follow these steps to get the chatbot running locally.

### Prerequisites

* Python 3.8+
* Pip
* An account with [Hugging Face](https://huggingface.co/)
* An account with [Pinecone](https://www.pinecone.io/)
* An account with [OpenRouter](https://openrouter.ai/) (for the VQA model, or you can run it locally if you have the resources)

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```
### 2. Create a Virtual Environment


It's recommended to use a virtual environment to manage dependencies.

```Bash

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all the required Python packages from the requirements.txt file.

```Bash

pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a file named .env in the root directory of the project and add your API keys:
```
HUGGINGFACEHUB_API_TOKEN="hf_..."
PINECONE_API_KEY="..."
PINECONE_ENVIRONMENT="..."
OPENROUTER_API_KEY="sk-....."
```

### 5. Ingest the Knowledge Base
Before running the app, you need to process the Samsung user manual and store its embeddings in your Pinecone index.

```Bash

python ingest.py
```

This script will load the PDF, split it into chunks, create embeddings, and upload them to your specified Pinecone index.

### 6. Run the Application
Once the ingestion is complete, you can start the Flask server.

```Bash

python main.py
```

The application will be running at http://127.0.0.1:5001.

### 💬 Usage
Open your web browser and navigate to http://127.0.0.1:5001.

For text queries: Type your question about the Samsung refrigerator into the chat box and press Enter.

For image queries: Click the "Upload Image" button, select an image file of an error code or part, and the chatbot will analyze it.

### 📂 Project Structure
```
.
├── data/
│   └── Samsung-user-manual.pdf   # Knowledge base PDF
├── templates/
│   └── sample.html               # Frontend HTML
├── .env                          # API keys and secrets
├── ingest.py                     # Script to process and embed the PDF
├── main.py                       # Main Flask application and API endpoints
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```





