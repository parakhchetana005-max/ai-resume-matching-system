import json
import openai
import os
from dotenv import load_dotenv

# Load environment variables for OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to call the LLM for matching and score calculation
def calculate_match_with_llm(job_desc_json, resume_json):
    """
    This function uses LLM (GPT-4) to calculate the match score between a job description and a resume.
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
    - Candidate Education: {', '.join([f'{edu["degree"]} from {edu["institution"]}' for edu in resume_json.get('education', [])])}
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
        "analysis": "Detailed explanation of the match"
    }}
    """

    # Call the LLM (OpenAI GPT) to process the prompt and return the analysis
    response = openai.ChatCompletion.create(
        model="gpt-4",  # or another version you prefer
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.5,
    )
    
    # Get the LLM response and parse the JSON-like structure
    response_text = response.choices[0].message["content"].strip()
    try:
        match_analysis = json.loads(response_text)
    except json.JSONDecodeError:
        print("Error decoding the LLM response into JSON.")
        match_analysis = None

    return match_analysis

# Helper function to load the JSON data from files
def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Example usage
job_description_file = '/home/abdulsamad/dhiwise_project/job_requirements_output.json'
resume_file = '/home/abdulsamad/dhiwise_project/resume_details_output.json'

# Load the JSON files
job_description_json = load_json_file(job_description_file)
resume_json = load_json_file(resume_file)

# Generate match analysis using the LLM
match_report = calculate_match_with_llm(job_description_json[0], resume_json[0])  # Assuming both JSON files contain an array of objects

# Save the match report to a JSON file
output_file = '/home/abdulsamad/dhiwise_project/match_analytics_output_with_llm.json'
with open(output_file, 'w') as f:
    json.dump(match_report, f, indent=4)

print(f"Match analytics (using LLM) saved to {output_file}")
