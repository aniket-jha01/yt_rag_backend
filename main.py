import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from newsapi import NewsApiClient

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Check for API keys
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set.")
if not NEWS_API_KEY:
    raise RuntimeError("NEWS_API_KEY environment variable not set.")

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:8001",
    "https://yt-rag-frontend.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory vector store (will be saved to disk)
vector_store = None
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Pydantic Models for Request Body Validation ---

class TopicPayload(BaseModel):
    topic: str

class QuestionPayload(BaseModel):
    question: str

# --- Utility Functions ---

def fetch_articles(topic: str):
    """Fetches news articles from News API based on a topic."""
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        all_articles = newsapi.get_everything(q=topic, language='en', sort_by='relevancy')
        
        if not all_articles or not all_articles['articles']:
            return None, "No articles found for the given topic."

        full_text = ""
        for article in all_articles['articles']:
            if article['title'] and article['content']:
                full_text += f"Title: {article['title']}\nContent: {article['content']}\n\n"
        
        doc = Document(page_content=full_text, metadata={'source': 'NewsAPI', 'topic': topic})
        
        print(f"✅ Successfully fetched articles for topic: {topic}")
        return [doc], None
    
    except Exception as e:
        error_message = f"An unexpected error occurred while fetching articles: {str(e)}"
        print(error_message)
        return None, error_message


# --- API Endpoints ---

@app.post("/analyze_topic")
async def analyze_topic(payload: TopicPayload):
    """Endpoint to process news articles for a given topic and store its vector index."""
    print(f"📩 Received request to analyze topic: {payload.topic}")
    global vector_store

    topic = payload.topic
    
    documents, error = fetch_articles(topic)
    if error:
        print(f"❌ Article fetching failed: {error}")
        raise HTTPException(status_code=500, detail=f"Error fetching articles: {error}")

    # Text Chunking
    print("🔪 Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    print(f"📄 Created {len(docs)} chunks.")

    # Limit to prevent timeouts
    docs_to_process = docs[:5]
    if not docs_to_process:
        raise HTTPException(status_code=500, detail="No documents to process.")

    try:
        print(f"🧠 Creating embeddings and vector store for {len(docs_to_process)} chunks...")
        vector_store = FAISS.from_documents(docs_to_process, embeddings)
        
        vector_store.save_local("faiss_index") 
        print("✅ Vector store created and saved successfully.")
    except Exception as e:
        print(f"❌ Error during vector store creation: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating vector store: {str(e)}")

    print("🎉 Topic processed successfully.")
    return {"success": True, "message": "Articles processed successfully. Ready to answer questions!"}


@app.post("/ask")
async def ask_question(payload: QuestionPayload):
    """Endpoint to answer questions based on the stored vector index."""
    print(f"📩 Received question: {payload.question}")
    global vector_store
    
    question = payload.question

    if not vector_store:
        print("⚠️ Vector store not in memory. Attempting to load from disk.")
        if os.path.exists("faiss_index"):
            try:
                vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                print("✅ Vector store loaded from disk.")
            except Exception as e:
                print("❌ Failed to load vector store from disk:", e)
                raise HTTPException(status_code=500, detail="Could not load vector store. Please process a topic first.")
        else:
            print("⚠️ Vector store not in memory and file does not exist.")
            raise HTTPException(status_code=400, detail="No topic has been processed yet. Please analyze a topic first.")

    try:
        print("🤖 Setting up RetrievalQA chain...")
        retriever = vector_store.as_retriever()
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
        prompt_template = """
        You are a helpful assistant that answers questions based on a collection of news articles.
        The user has provided a question, and you will use the provided context to answer it.
        If the question cannot be answered from the articles, politely say so.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        result = await chain.ainvoke({"query": question})
        answer = result['result']
        
        print("✅ Question answered successfully.")
        return {"success": True, "answer": answer}
        
    except Exception as e:
        print(f"❌ An error occurred while answering the question: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
