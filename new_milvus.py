# Import necessary libraries
from fastapi import FastAPI, HTTPException  # FastAPI framework
from fastapi.responses import JSONResponse  # JSON response handling
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Text chunking
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI  # AI models
from langchain.chains.question_answering import load_qa_chain  # QA chain loading
from langchain.prompts import PromptTemplate  # Prompt creation
from dotenv import load_dotenv  # Load environment variables
from web_scrap import scrape_wikipedia  # Custom scraping function
from pymilvus import MilvusClient  # Milvus database client
from langchain_milvus import Milvus  # LangChain Milvus integration

# Load environment variables
load_dotenv()

# Initialize FastAPI and Milvus client
app = FastAPI()
client = MilvusClient(uri="tcp://localhost:19530")

# Drop existing collection if it exists
if client.has_collection("rag_model_example"):
    client.drop_collection("rag_model_example")

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to create vector store in Milvus
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    Milvus.from_texts(text_chunks, embedding=embeddings, collection_name="rag_model_example")

# Function to create a conversational chain for QA
def get_conversational_chain():
    prompt_template = """..."""  # Define your prompt here
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Endpoint for scraping content
@app.post("/scrape")
async def scrape(url: str):
    scraped_content = scrape_wikipedia(url)  # Scrape the given URL
    raw_text = " ".join(scraped_content)  # Combine scraped content
    text_chunks = get_text_chunks(raw_text)  # Split text into chunks
    get_vector_store(text_chunks)  # Store vectors in Milvus
    return JSONResponse(content={"message": "Scraping and processing done."})

# Endpoint for answering questions
@app.get("/ask")
async def ask_question(user_question: str):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = Milvus(embedding_function=embeddings, collection_name="rag_model_example")
    docs = new_db.similarity_search(user_question)  # Search for relevant documents

    chain = get_conversational_chain()  # Get QA chain
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    return JSONResponse(content={"reply": response["output_text"]})  # Return answer

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn  # ASGI server
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Start server
