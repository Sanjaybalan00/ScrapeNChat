from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from web_scrap import scrape_wikipedia
from pymilvus import MilvusClient
from langchain_milvus import Milvus

load_dotenv()

app = FastAPI()
client = MilvusClient(uri="tcp://localhost:19530")
if client.has_collection("rag_model_example"):
    client.drop_collection("rag_model_example")

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Milvus.from_texts(text_chunks, embedding=embeddings, collection_name="rag_model_example")
    # Milvus handles saving and retrieving the index

def get_conversational_chain():
    # This is the key prompt that instructs the model how to respond
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in 
    the provided context, say, "answer is not available in the context." Don't provide a wrong answer.\n\n
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

@app.post("/scrape")
async def scrape(url: str):
    scraped_content = scrape_wikipedia(url)
    raw_text = " ".join(scraped_content)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    return JSONResponse(content={"message": "Scraping and processing done."})

@app.get("/ask")
async def ask_question(user_question: str):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = Milvus(embedding_function=embeddings, collection_name="rag_model_example")
    docs = new_db.similarity_search(user_question)

    # Get the conversational chain which uses the defined prompt
    chain = get_conversational_chain()
    
    # Generate the response using the question and relevant documents
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    # Return the model's response based on the context and the prompt's instructions
    return JSONResponse(content={"reply": response["output_text"]})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
