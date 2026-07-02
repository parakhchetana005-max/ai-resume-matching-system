import os
import logging
import httpx
from time import sleep
from dotenv import load_dotenv
import assemblyai as aai
from langchain_community.document_loaders import YoutubeLoader
from typing import Union

# Load environment variables
load_dotenv()

# Set up AssemblyAI API key
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transcribe_audio(file_path: str) -> str:
    """Transcribe the audio/video file to text using AssemblyAI."""
    transcriber = aai.Transcriber()
    try:
        transcript = transcriber.transcribe(file_path)
        return transcript.text
    except httpx.RequestError as e:
        logger.error(f"Request error: {e}")
        return "Error during transcription."
    except httpx.TimeoutException:
        logger.warning(f"Timeout during transcription for {file_path}, retrying...")
        sleep(5)
        return transcribe_audio(file_path)

def transcribe_youtube_video(url: str) -> str:
    """Transcribe a YouTube video to text using YoutubeLoader."""
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
    video_docs = loader.load()
    transcribed_text = " ".join([doc.page_content for doc in video_docs])
    return transcribed_text

def transcribe_content(source: Union[str, str]) -> str:
    """Transcribe either an audio/video file or a YouTube video based on the source type."""
    if source.startswith("http") and "youtube.com" in source:
        return transcribe_youtube_video(source)
    else:
        return transcribe_audio(source)
