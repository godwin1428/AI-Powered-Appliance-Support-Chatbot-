import os
import json
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from PIL import Image
from huggingface_hub import InferenceClient
import io
import base64 # NEW: Import for encoding the image

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
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") # NEW: Load OpenRouter API Key

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

# NEW: Replaced the deepai function with one for OpenRouter
def analyze_image_with_openrouter(image_bytes):
	"""
	Analyzes an image using a free multimodal model via the OpenRouter API,
	then enhances the result using RAG from product manuals for deeper troubleshooting.
	"""
	if not OPENROUTER_API_KEY:
		print("--- ERROR: OPENROUTER_API_KEY not found in environment variables. ---")
		return {"error":"Server is missing the OpenRouter API key configuration."}

	try:
		# Step 1: Detect image MIME type
		image=Image.open(io.BytesIO(image_bytes))
		image_format=image.format.lower() if image.format else 'jpeg'
		mime_type=f"image/{image_format}"
		print(f"--- Detected image MIME type: {mime_type} ---")

		# Step 2: Encode image
		base64_image=base64.b64encode(image_bytes).decode('utf-8')

		# Step 3: Use free vision model for base image analysis
		vision_model="google/gemma-3-4b-it:free"
		payload={
			"model":vision_model,
			"messages":[
				{
					"role":"user",
					"content":[
						{
							"type":"text",
							"text":"""
								You are a refrigerator troubleshooting assistant. Analyze the following image of a refrigerator.
								Based *only* on the visual information, generate a JSON object with the following keys:
								- "detected_issue": A short, clear description of the problem (e.g., "Excessive frost buildup on the back wall").
								- "possible_cause": A likely reason for this issue (e.g., "Poor air circulation or faulty defrost system.").
								- "recommended_action": A concise, step-by-step solution for the user (e.g., "1. Check if vents are blocked. 2. Manually defrost the unit. 3. If the problem persists, contact a technician.").

								If the image is unclear or shows no obvious issue, state that in the JSON.
								Return ONLY the raw JSON object and nothing else.
							"""
						},
						{
							"type":"image_url",
							"image_url":{"url":f"data:{mime_type};base64,{base64_image}"}
						}
					]
				}
			]
		}

		headers={
			"Authorization":f"Bearer {OPENROUTER_API_KEY}",
			"Content-Type":"application/json",
			"HTTP-Referer":"http://localhost:5001",
			"X-Title":"Samsung Refrigerator Troubleshooter"
		}

		response=requests.post("https://openrouter.ai/api/v1/chat/completions",headers=headers,json=payload,timeout=60)
		response.raise_for_status()

		model_output=response.json()['choices'][0]['message']['content']
		if model_output.strip().startswith("```json"):
			model_output=model_output.replace("```json","").replace("```","").strip()

		print(f"--- VISION MODEL RAW OUTPUT ---\n{model_output}\n-----------------------------")
		image_analysis=json.loads(model_output)

		# Step 4: Use RAG to enrich the response
		query_text=f"The refrigerator shows: {image_analysis.get('detected_issue','unknown issue')}. Explain the cause and steps to fix it."
		retrieved_context="\n".join([doc.page_content for doc in retriever.get_relevant_documents(query_text)])
		
		# Step 5: Combine retrieved manual data with vision findings
		final_prompt=f"""
		System Context:
		You are a Samsung refrigerator troubleshooting assistant.
		Image Analysis Result:
		{json.dumps(image_analysis,indent=2)}
		Retrieved Manual Information:
		{retrieved_context}
		
		Instruction:
		Using the retrieved manual data, expand the recommended_action section into a clear step-by-step troubleshooting guide.
		If possible, mention which components might need checking or servicing.
		"""

		rag_response=llm.invoke(final_prompt)
		return {"image_analysis":image_analysis,"rag_output":rag_response.content}

	except requests.exceptions.HTTPError as e:
		error_text=e.response.text if e.response else str(e)
		print(f"--- HTTP ERROR ---\n{e}\nResponse Body: {error_text}\n-----------------------------")
		return {"error":f"API request failed: {error_text}"}
	except json.JSONDecodeError:
		print("--- ERROR: Invalid JSON from Vision Model ---")
		return {"error":"Model returned invalid JSON. Please retry."}
	except Exception as e:
		print(f"--- UNEXPECTED ERROR ---\nType: {type(e)}\nError: {e}\n-----------------------------")
		return {"error":f"An unexpected error occurred: {str(e)}"}


@app.route("/chat", methods=["POST"])
def chat_handler():
    """
    Main endpoint to handle requests from the HTML front-end.
    """
    if 'file' in request.files and request.files['file'].filename != '':
        # --- Handle Image/File Query ---
        file = request.files['file']
        img_bytes = file.read()
        # MODIFIED: Call the new OpenRouter function
        response_data = analyze_image_with_openrouter(img_bytes)
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
