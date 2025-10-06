AI-Powered Samsung Refrigerator Support ChatbotThis project is a smart, interactive chatbot designed to help users troubleshoot issues with their Samsung refrigerators. It leverages modern AI and machine learning models to provide accurate, context-aware support through both text-based conversation and image analysis.<!-- Replace with an actual GIF of your chatbot in action -->✨ FeaturesConversational Q&A: Ask troubleshooting questions in natural language and get clear, step-by-step solutions.Knowledge-Based Answers: The chatbot uses a Retrieval-Augmented Generation (RAG) pipeline, ensuring all answers are grounded in the official Samsung user manual.Visual Diagnostics (VQA): Upload a photo of your refrigerator, and the AI will analyze the image to identify potential issues and recommend actions.Interactive Web Interface: A clean and user-friendly chat interface for a seamless user experience.🛠️ How It WorksThe application is built with a Python Flask backend and a simple HTML/CSS/JS frontend. It has two primary modes of operation:1. Text-Based Queries (RAG)When a user sends a text message, the system follows a RAG pipeline:Embedding: The user's query is converted into a vector embedding.Retrieval: The system searches a Pinecone vector database to find the most relevant text chunks from the ingested Samsung product manual.Generation: The retrieved text chunks and the original query are passed to a Large Language Model (Mistral via Hugging Face) which generates a helpful, conversational answer based only on the provided context.2. Image-Based Queries (VQA)When a user uploads an image:Analysis: The image is sent to the DeepAI Visual Question Answering API.Interpretation: The API's analysis of the image is returned as a text description.Formatting: The LLM formats this description into a user-friendly JSON object containing the detected issue, possible cause, and a recommended solution.🚀 Tech StackBackend: FlaskFrontend: HTML, CSS, JavaScript (no frameworks)AI Framework: LangChainVector Database: PineconeLLM & Embeddings: Hugging Face (Mistral-8x7B, all-MiniLM-L6-v2)Visual Question Answering: DeepAIDeployment: Python WSGI Server⚙️ Setup and InstallationFollow these steps to get the chatbot running locally.PrerequisitesPython 3.8+A Pinecone accountA Hugging Face accountA DeepAI account1. Clone the Repositorygit clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
2. Create a Virtual EnvironmentIt's recommended to use a virtual environment to manage dependencies.# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install DependenciesInstall all the required Python packages from the requirements.txt file.pip install -r requirements.txt
4. Set Up Environment VariablesCreate a file named .env in the root directory of the project and add your API keys:HUGGINGFACEHUB_API_TOKEN="hf_..."
PINECONE_API_KEY="..."
PINECONE_ENVIRONMENT="..."
DEEPAI_API_KEY="..."
5. Ingest the Knowledge BaseBefore running the app, you need to process the Samsung user manual and store its embeddings in your Pinecone index.Create a data folder in the root directory.Place your PDF manual inside it (e.g., data/Samsung-user-manual.pdf).Run the ingestion script:python ingest.py
This script will load the PDF, split it into chunks, create embeddings, and upload them to your specified Pinecone index.6. Run the ApplicationOnce the ingestion is complete, you can start the Flask server.python main.py
The application will be running at http://127.0.0.1:5001.💬 UsageOpen your web browser and navigate to http://127.0.0.1:5001.Click on the chat toggle button in the bottom-right corner to open the chat widget.Ask a question about your refrigerator or click the attachment icon to upload an image for analysis.📂 Project Structure.
├── data/
│   └── Samsung-user-manual.pdf   # Knowledge base PDF
├── templates/
│   └── sample.html               # Frontend HTML
├── .env                          # API keys and secrets
├── ingest.py                     # Script to process and embed the PDF
├── main.py                       # Main Flask application and API endpoints
├── requirements.txt              # Python dependencies
└── README.md                     # This file
📄 LicenseThis project is licensed under the MIT License. See the LICENSE file for details.
