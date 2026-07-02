import json
from langchain_groq import ChatGroq
import os 
# Set up Groq API key and initialize ChatGroq model
from dotenv import load_dotenv
load_dotenv()  # Make sure this is called before using the API key

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="mixtral-8x7b-32768",  # Model you want to use
    temperature=0.0,  # deterministic response
    max_retries=2,    # Max retries
    api_key=groq_api_key
)

def calculate_match_with_llm(job_desc_json, resume_json):
    """
    This function uses LLM (Groq) to calculate the match score between a job description and a resume.
    """
    # Prepare the prompt to evaluate and compare key features
    prompt = f"""
    You are an AI that evaluates job-fit based on resumes. Please evaluate the following:
    
    JOB DESCRIPTION:
    - Required Skills: {', '.join(job_desc_json.get('skills', []))}
    - Required Experience: {job_desc_json.get('experience_years', '')}
    - Required Education: {', '.join(job_desc_json.get('qualifications', []))}
    - Required Technologies: {', '.join(job_desc_json.get('technologies', []))}

    RESUME:
    - Candidate Skills: {', '.join(resume_json.get('skills', []))}
    - Candidate Experience: {resume_json.get('experience_years', '')}
    - Candidate Education: {', '.join([edu.get('degree', 'N/A') for edu in resume_json.get('education', [])])}
    - Candidate Technologies: {', '.join(resume_json.get('skills', []))}

    Please provide a detailed analysis of the candidate’s match to the job description, covering:
    - Skill match (percentage and relevant skills)
    - Experience match (alignment with job experience requirements)
    - Education match (how well the candidate's qualifications fit)
    - Technological fit (how well the candidate’s tools and technologies align with the job)
    
    Also, provide an overall match score out of 100.
    
    Output should be in the following format:
    {{
        "skills_match_percentage": <float>,
        "experience_match_percentage": <float>,
        "education_match_percentage": <float>,
        "technologies_match_percentage": <float>,
        "overall_match_score": <float>,
        "analysis": "Detailed explanation of the match be precise and consise dont haluicante"
    }}
    """

    # Define the message format according to the Groq API format
    messages = [
        ("system", "You are an AI an expert that evaluates job-fit based on resumes."),
        ("human", prompt)
    ]
    
    # Invoke the model using the invoke method with messages
    response = llm.invoke(messages)

    # Get the response content and parse it
    response_text = response.content.strip()
    
    try:
        match_analysis = json.loads(response_text)
    except json.JSONDecodeError:
        print("Error decoding the LLM response into JSON.")
        match_analysis = None

    return match_analysis
