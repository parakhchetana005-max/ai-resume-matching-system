import logging
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_cohere import CohereEmbeddings

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def perform_similarity_search(query: str, db: Chroma, top_k: int = 3):
    """Perform a similarity search."""
    docs = db.similarity_search(query, k=top_k)
    return docs

def get_most_relevant_document_and_summarize(query: str, transcribed_text: str, vectorstore: Chroma, top_k: int = 3) -> str:
    """Retrieve and summarize the most relevant document."""
    results = perform_similarity_search(query, vectorstore, top_k)
    if results:
        most_relevant_chunk = results[0].page_content
        return summarize_with_groq(most_relevant_chunk)
    return "No relevant documents found."

def summarize_with_groq(document_chunk: str) -> str:
    """Summarize document chunk using Groq."""
    try:
        from langchain_groq import ChatGroq
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate

        llm = ChatGroq(model="llama3-8b-8192")
        prompt = ChatPromptTemplate.from_template("""
    Generate a comprehensive contextual summary of the candidate's performance based on the interview content. Focus on the following key traits:

    1. **Communication Style**: 
       - Assess the clarity, coherence, and effectiveness of the candidate’s verbal expression.
       - Highlight any notable strengths (e.g., concise answers, clear articulation) or weaknesses (e.g., verbosity, lack of clarity).

    2. **Active Listening**: 
       - Evaluate how well the candidate listens and responds to the interviewer’s questions.
       - Look for signs of attentiveness, such as relevant follow-up responses, acknowledgment of the interviewer’s points, and appropriate pauses.
       - Identify moments where the candidate may have misunderstood or missed key aspects of the question.

    3. **Engagement with the Interviewer**: 
       - Analyze the candidate’s level of interaction with the interviewer, including both verbal and non-verbal cues (e.g., tone, eye contact, body language).
       - Assess their ability to build rapport, show enthusiasm, and engage in a balanced conversation.

    **Context**: {context}
""")

        chain = create_stuff_documents_chain(llm, prompt)
        result = chain.invoke({"context": [Document(page_content=document_chunk)]})
        return result
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        return "Error summarizing the content."
