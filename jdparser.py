import os
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import json
import openai  # Assuming you're using OpenAI's GPT or similar LLM

# Load environment variables for LLAMA_CLOUD_API_KEY (for LlamaParse) and OPENAI_API_KEY (for GPT LLM)
load_dotenv()

# Initialize LlamaParse with the desired output format
parser = LlamaParse(result_type="text")  # or "markdown" depending on your preference

# Set up OpenAI API key for LLM
openai.api_key = os.getenv("OPENAI_API_KEY")

# Helper function to interact with LLM for extracting key features
def extract_key_features_with_llm(parsed_data):
    """
    This function sends the parsed data to an LLM (e.g., GPT-3) with a specific prompt to extract key features.
    """
    prompt = f"""
    Extract the following key details from the given job description text:

    - Skills
    - Qualifications
    - Years of experience (if mentioned)
    - Responsibilities

    Please format the extracted information into a structured JSON like the example below:
    Example:
    {{
        "skills": ["skill1", "skill2", "skill3"],
        "qualifications": ["qualification1", "qualification2"],
        "experience_years": "X years",
        "responsibilities": ["responsibility1", "responsibility2"]
    }}

    Job Description: {parsed_data}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",  # or whichever engine you're using
        messages=[{"role": "user", "content": prompt}],
        #prompt=prompt,
        max_tokens=500,
        temperature=0.5,
    )

    # Get the JSON-like response
    response_text = response.choices[0].message["content"].strip()

    try:
        # Parse the response into a JSON object
        extracted_data = json.loads(response_text)
        return extracted_data
    except json.JSONDecodeError:
        print("Error decoding the LLM response into JSON.")
        return None

# Function to process job descriptions, parse, extract with LLM, and save results
def process_and_extract_job_details(files):
    print(f"Processing job descriptions from files: {files}")
    
    # Load and parse documents using LlamaParse
    file_extractor = {".txt": parser, ".json": parser}
    documents = SimpleDirectoryReader(input_files=files, file_extractor=file_extractor).load_data()
    print(f"Loaded documents: {documents}")

    all_job_requirements = []

    for doc in documents:
        # Print the type of doc to understand its structure
        print(f"Type of document: {type(doc)}")
        
        # Access document text directly
        document_text = doc.text if hasattr(doc, 'text') else ""
        print(f"Document content (first 100 chars): {document_text[:100]}...")

        # Use LLM to extract key features from the document text
        job_requirements = extract_key_features_with_llm(document_text)
        if job_requirements:
            all_job_requirements.append(job_requirements)

    return all_job_requirements

# Example usage
job_description_files = [
    '/home/abdulsamad/dhiwise_project/resume.docx',
    '/home/abdulsamad/dhiwise_project/sample_resume.txt',
    #'/path/to/job_description_3.txt'
]

# Process the job descriptions and extract key features
job_requirements = process_and_extract_job_details(job_description_files)

# Save the extracted job requirements to a JSON file
output_file = '/home/abdulsamad/dhiwise_project/job_requirements_output.json'
with open(output_file, 'w') as f:
    json.dump(job_requirements, f, indent=4)

print(f"Extracted job requirements saved to {output_file}")
