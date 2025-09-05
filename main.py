import os
import re
import json
import yt_dlp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check for API key
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set.")

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

# --- Pydantic Models for Request Body Validation ---

class URLPayload(BaseModel):
    youtube_url: str

class QuestionPayload(BaseModel):
    question: str

# --- Utility Functions ---

def get_video_id(url: str) -> str | None:
    """Extracts YouTube video ID from a URL."""
    regex = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=)?([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    return match.group(1) if match else None

def get_transcript_documents(video_id: str):
    """
    Fetches YouTube transcript using yt-dlp and formats it for LangChain.
    This method is more robust than youtube-transcript-api.
    """
    try:
        ydl_opts = {
            'writesubtitles': True,
            'subtitleslangs': ['en'],
            'skip_download': True,
            'outtmpl': '%(id)s.%(ext)s',
            'writeinfojson': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_id, download=False)
            
            # Check for available subtitles
            if 'subtitles' not in info or 'en' not in info['subtitles']:
                return None, "No English transcript found for this video."

            # Re-fetch with a different option to get the transcript data as a string
            ydl_opts_get_subs = {
                'writesubtitles': True,
                'subtitleslangs': ['en'],
                'skip_download': True,
                'outtmpl': '-',
                'quiet': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts_get_subs) as ydl_get_subs:
                extracted_info = ydl_get_subs.extract_info(video_id, download=False)
                
                transcript_text = ""
                # yt-dlp returns the info dict with the 'subtitles' field populated
                if 'subtitles' in extracted_info and 'en' in extracted_info['subtitles']:
                    # Assuming the first format is the one we want
                    sub_url = extracted_info['subtitles']['en'][0]['url']
                    # Download the subtitle content
                    subtitle_content = ydl.urlopen(sub_url).read().decode('utf-8')
                    transcript_text = subtitle_content
                
                if not transcript_text:
                    return None, "Failed to extract transcript text."
            
            # Create a single LangChain Document object from the transcript string
            doc = Document(page_content=transcript_text, metadata={'source': 'YouTube', 'video_id': video_id})
            
            print(f"Successfully fetched transcript for video ID: {video_id}")
            return [doc], None
            
    except Exception as e:
        error_message = f"An unexpected error occurred while fetching the transcript: {str(e)}"
        print(error_message)
        return None, error_message

# --- API Endpoints ---

@app.post("/analyze")
async def analyze_video(payload: URLPayload):
    """Endpoint to process a YouTube URL and store its vector index."""
    print(f"Received request to analyze URL: {payload.youtube_url}")
    global vector_store

    youtube_url = payload.youtube_url
    video_id = get_video_id(youtube_url)
    if not video_id:
        print("Invalid YouTube URL provided.")
        raise HTTPException(status_code=400, detail="Invalid YouTube URL.")
    
    print(f"Extracted video ID: {video_id}")

    documents, error = get_transcript_documents(video_id)
    if error:
        print(f"Transcript processing failed: {error}")
        raise HTTPException(status_code=500, detail=f"Error fetching transcript: {error}")

    # Text Chunking
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"Created {len(docs)} chunks.")

    # Embedding and Vector Store
    try:
        print("Creating embeddings and vector store...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local("faiss_index") 
        print("Vector store created and saved successfully.")
    except Exception as e:
        print(f"Error during vector store creation: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating vector store: {str(e)}")

    print("Video processed successfully.")
    return {"success": True, "message": "Video processed successfully. Ready to answer questions!"}

@app.post("/ask")
async def ask_question(payload: QuestionPayload):
    """Endpoint to answer questions based on the stored vector index."""
    print(f"Received question: {payload.question}")
    global vector_store
    
    question = payload.question

    if not vector_store:
        print("Vector store not in memory. Attempting to load from disk.")
        if os.path.exists("faiss_index"):
            try:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
                vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                print("Vector store loaded from disk.")
            except Exception as e:
                print("Failed to load vector store from disk:", e)
                raise HTTPException(status_code=500, detail="Could not load vector store. Please process a video first.")
        else:
            print("Vector store not in memory and file does not exist.")
            raise HTTPException(status_code=400, detail="No video has been processed yet. Please process a video first.")

    try:
        print("Setting up RetrievalQA chain...")
        retriever = vector_store.as_retriever()
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
        prompt_template = """
        You are a helpful assistant that answers questions about a YouTube video based on its transcript.
        The user has provided a question, and you will use the provided context to answer it.
        The context includes timestamps, so use them to provide precise timing for your answer.
        If the question cannot be answered from the transcript, politely say so.

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
        
        result = chain.invoke({"query": question})
        answer = result['result']
        source_docs = result['source_documents']
        
        final_answer = answer
        
        print("Question answered successfully.")
        return {"success": True, "answer": final_answer}
        
    except Exception as e:
        print(f"An error occurred while answering the question: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")