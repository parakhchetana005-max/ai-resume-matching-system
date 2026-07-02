import os
import logging
from traanscriber import transcribe_content
from embeddings import create_vector_store
from similarity import get_most_relevant_document_and_summarize
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Example source: could be a YouTube URL or file path (YouTube URL shown here)
    source = "https://www.youtube.com/watch?v=5v-wyR5emRw"  # Change this to your URL or file path
    query = """
    Retrieve and summarize parts of the interview where the candidate demonstrates:
    - Clear and effective communication in expressing their thoughts.
    - Attentiveness and responsiveness to the interviewer's questions (active listening).
    - Engagement with the interviewer, showing rapport, enthusiasm, and a balanced conversation.
"""


    # Step 1: Transcribe the audio/video content
    logger.info("Starting transcription process...")
    transcribed_text = transcribe_content(source)
    
    # Step 2: Create embeddings and a vector store from the transcribed text
    logger.info("Creating vector store...")
    vectorstore = create_vector_store(transcribed_text)

    # Step 3: Perform similarity search and summarize the most relevant document
    logger.info("Performing similarity search and summarizing...")
    summarized_result = get_most_relevant_document_and_summarize(query, transcribed_text, vectorstore=vectorstore)

    # Output the summarized result
    logger.info(f"Summarized result: {summarized_result}")

if __name__ == "__main__":
    main()
