import os
import json
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from PIL import Image
from huggingface_hub import InferenceClient
import io
import deepai

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- Load Environment Variables ---
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

# --- Global Configuration & Initialization ---
PINECONE_INDEX_NAME = "product-manual"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM for Text Generation (RAG)
# 1. Create the base LLM that connects to the endpoint
base_llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation",
    max_new_tokens=512,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

# 2. Wrap the base LLM in the ChatHuggingFace class
llm = ChatHuggingFace(llm=base_llm)

# VQA Client
inference_client = InferenceClient(token=HUGGINGFACEHUB_API_TOKEN)

# Visual Question Answering (VQA) Model
VQA_API_URL = "https://api-inference.huggingface.co/models/dandelin/vilt-b32-finetuned-vqa"
headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

# --- Initialize Embeddings and Vector Store ---
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={'k': 4}) # Retrieve top 4 results

# --- Prompt Template for RAG ---
# This mirrors the system prompt in your n8n AI Agent node
prompt_template = """
System/Context:
You are a helpful AI assistant for troubleshooting Samsung refrigerators. Use a conversational and easy-to-understand tone. Your answers must be based *only* on the retrieved information below.

Retrieved Information:
{context}

User Query:
{question}

Final Instruction:
Based on the "Retrieved Information" and the "User Query," provide a detailed, step-by-step troubleshooting guide. If the information suggests the issue is normal, explain that clearly. Do not use any information that is not explicitly supported by the retrieved text.
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# --- Define the RAG Chain using LangChain Expression Language (LCEL) ---
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

# --- Initialize Flask App ---
app = Flask(__name__)
@app.route("/")
def index():
    """Serves the chatbot HTML page."""
    return render_template("sample.html")

def analyze_image_with_deepai(image_bytes):
    """Analyzes an image using the DeepAI VQA model."""
    try:
        # The DeepAI library needs a file path, so we write the bytes to a temporary file
        with open("temp_image.jpg", "wb") as f:
            f.write(image_bytes)

        # CORRECTED: Call the DeepAI VQA API with the correct function name
        resp = deepai.call_standard_api("visual-question-answering",
            image=open("temp_image.jpg", "rb"),
            question="What is the primary issue visible in this refrigerator image?"
        )
        os.remove("temp_image.jpg") # Clean up the temporary file

        # Extract the answer
        detected_issue = resp.get('output', 'Could not determine the issue.')
        print(f"--- DEEPAI VQA RESPONSE ---\n{detected_issue}\n-------------------------")

        # Use the main LLM to format the output into the desired JSON structure
        formatting_prompt = f'''
        An image of a refrigerator was analyzed. The detected issue is: "{detected_issue}"

        Based on this, generate a JSON object with the following keys:
        - "detected_issue": A short description of the issue.
        - "possible_cause": A likely reason for the issue.
        - "recommended_action": A step-by-step solution for the user.

        If the detected issue is unclear, state that more details are needed.
        Return only the raw JSON object.
        '''

        json_response_str = llm.invoke(formatting_prompt)
        print(f"--- RAW LLM RESPONSE ---\n{json_response_str}\n------------------------")

        json_response_dict = json.loads(json_response_str)
        return json_response_dict

    except Exception as e:
        if os.path.exists("temp_image.jpg"):
            os.remove("temp_image.jpg")
        print(f"--- ERROR IN DEEPAI IMAGE ANALYSIS ---\nType: {type(e)}\nError: {e}\n-----------------------------")
        return {"error": f"Failed to analyze image with DeepAI: {str(e)}"}
    
@app.route("/chat", methods=["POST"])
def chat_handler():
    """
    Main endpoint to handle requests from the HTML front-end.
    """
    if 'file' in request.files and request.files['file'].filename != '':
        # --- Handle Image/File Query ---
        file = request.files['file']
        img_bytes = file.read()
        response_data = analyze_image_with_deepai(img_bytes)
        return jsonify(response_data)

    else:
        # --- Handle Text Query ---
        chat_input = request.form.get("chatInput")
        if not chat_input:
            return jsonify({"error": "chatInput is required"}), 400

        # Invoke the RAG chain
        response_text = rag_chain.invoke(chat_input)
        return jsonify({"output": response_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)