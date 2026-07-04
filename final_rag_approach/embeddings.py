import os
import logging
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Cohere API key
cohere_api_key = os.getenv("COHERE_API_KEY")

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_vector_store(transcribed_text: str) -> Chroma:
    """Create a vector store with Cohere embeddings."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents([Document(page_content=transcribed_text)])
    embeddings = CohereEmbeddings(model='embed-english-v3.0')
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore
