import os
import streamlit as st
from traanscriber import transcribe_content  # Import transcription functionality
from embeddings import create_vector_store  # Import embedding creation
from similarity import get_most_relevant_document_and_summarize  # Import similarity search and summarization
import logging

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Set up Streamlit app title and description
    st.title("Candidate Analysis System")
    st.markdown("Upload an audio/video file, or provide a YouTube URL to automatically process and generate the summary.")

    # Sidebar for file upload or URL input
    st.sidebar.header("Options")
    transcribe_option = st.sidebar.radio("Choose your input type:", ("Upload File", "YouTube URL"))

    vectorstore = None  # Placeholder for Chroma vectorstore

    # Ensure the "temp" directory exists for saving files
    if not os.path.exists("temp"):
        os.makedirs("temp")

    # Predefined query to use for similarity search and summarization
    predefined_query = """
    Retrieve and summarize parts of the interview where the candidate demonstrates:
    - Clear and effective communication in expressing their thoughts.
    - Attentiveness and responsiveness to the interviewer's questions (active listening).
    - Engagement with the interviewer, showing rapport, enthusiasm, and a balanced conversation.
    """

    # Section for uploading files or entering YouTube URL
    if transcribe_option == "Upload File":
        # File upload input for audio/video file
        uploaded_file = st.file_uploader("Upload an audio or video file", type=["mp3", "mp4", "wav", "mkv", "avi"])
        
        if uploaded_file is not None:
            # Save the uploaded file to a temporary location
            temp_file_path = os.path.join("temp", uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Perform transcription
            st.write("Transcribing the uploaded file...")
            transcription = transcribe_content(temp_file_path)

            # Create embeddings for the transcribed text and store them in a vector store
            st.write("Creating embeddings...")
            vectorstore = create_vector_store(transcription)
            st.write("Embeddings created.")

            # Clean up the temporary file after transcription
            os.remove(temp_file_path)

    elif transcribe_option == "YouTube URL":
        # YouTube URL input
        youtube_url = st.text_input("Enter YouTube Video URL:")
        
        if youtube_url:
            # Perform transcription
            st.write("Transcribing the YouTube video...")
            transcription = transcribe_content(youtube_url)

            # Create embeddings for the transcribed text and store them in a vector store
            st.write("Creating embeddings...")
            vectorstore = create_vector_store(transcription)
            st.write("Embeddings created.")

    # Once the vectorstore is ready, perform the similarity search and summarization
    if vectorstore:
        st.write("Performing similarity search and summarizing...")
        summarized_result = get_most_relevant_document_and_summarize(predefined_query, transcription, vectorstore)
        
        # Output the summarized result
        st.subheader("Summarized Result:")
        st.text_area("Summary", summarized_result, height=300)

if __name__ == "__main__":
    main()
