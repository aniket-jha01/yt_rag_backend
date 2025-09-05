import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

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
    """Fetches YouTube transcript and formats it for LangChain."""
    print(f"Attempting to fetch transcript for video ID: {video_id}")
    try:
        # Check for English transcript explicitly
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        english_transcript = None
        for t in transcript_list:
            if t.language_code == 'en':
                english_transcript = t
                break
        
        if not english_transcript:
            print(f"Error: Captions not found or not in English for video ID: {video_id}")
            return None, "Captions not found or not in English."
            
        # Use LangChain's loader for cleaner document generation
        loader = YoutubeLoader(video_id=video_id, add_video_info=True, language="en")
        documents = loader.load()
        
        # Add timestamps to the documents for better retrieval
        timestamps = YouTubeTranscriptApi.get_transcript(video_id)
        
        if not documents:
            return None, "Loader returned no documents."
            
        for doc in documents:
            doc_content_with_timestamps = ""
            for item in timestamps:
                start_time_minutes = int(item['start'] // 60)
                start_time_seconds = int(item['start'] % 60)
                doc_content_with_timestamps += f"{start_time_minutes:02d}:{start_time_seconds:02d} - {item['text']}\n"
            doc.page_content = doc_content_with_timestamps
        
        print(f"Successfully fetched and formatted transcript for video ID: {video_id}")
        return documents, None
        
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        error_message = f"Transcript is disabled or not found for this video: {str(e)}"
        print(error_message)
        return None, error_message
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
        # On Render, the file system is ephemeral, but saving to disk is good practice for local development or for more persistent cloud solutions
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
        # Load vector store only if it doesn't exist in memory
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
        
        result = await chain.ainvoke({"query": question})
        answer = result['result']
        source_docs = result['source_documents']
        
        # Extract and format timestamps from source documents
        timestamps_found = []
        for doc in source_docs:
            lines = doc.page_content.strip().split('\n')
            for line in lines:
                if re.match(r"^\d{2}:\d{2} -", line):
                    timestamps_found.append(line.split(' - ')[0])
        
        unique_timestamps = sorted(list(set(timestamps_found)))
        formatted_timestamps = ", ".join(unique_timestamps)
        
        final_answer = f"{answer}\n\nRelevant timestamps: {formatted_timestamps}" if formatted_timestamps else answer
        
        print("Question answered successfully.")
        return {"success": True, "answer": final_answer}
        
    except Exception as e:
        print(f"An error occurred while answering the question: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")