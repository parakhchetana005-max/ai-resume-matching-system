import os
import json
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from langchain_groq import ChatGroq  # Importing ChatGroq from langchain_groq

# Load environment variables for API keys (LlamaParse and Groq)
load_dotenv()

# Initialize LlamaParse with the desired output format (text)
parser = LlamaParse(result_type="text")

# Set up Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize ChatGroq without `openai_api_key`
llm = ChatGroq(
    model="mixtral-8x7b-32768",  # Your model name (could be different, as per your choice)
    temperature=0.0,
    max_retries=2,
    # No need to pass `openai_api_key` here, it's handled by the environment
)

# Helper function to interact with the LLM (ChatGroq) and extract key features from resumes
def extract_key_features_from_resume(parsed_data):
    """
    This function sends the parsed resume data to the LLM (ChatGroq) with a specific prompt to extract key details.
    """
    print("Sending resume data to LLM for extraction...")

    prompt = f"""
    You are an expert at extracting key information from resumes. Please carefully parse the resume provided and extract the following details:

    1. **Full Name**:
    - Extract the full name of the person.

2. **Contact Information**:
    - Extract the phone number, email, and any other relevant contact details.

3. **Skills**:
    - List all the technical and non-technical skills mentioned in the resume.
    - Include hard skills (e.g., programming languages, tools, software) and soft skills (e.g., communication, teamwork, leadership).

4. **Education**:
    - List all educational qualifications including the degree, institution name, and the year of graduation.

5. **Work Experience**:
    - List all work experiences, including job title, company name, duration (e.g., "2 years"), and key responsibilities.
    - Ensure the responsibilities are provided as short, bullet-pointed phrases.

6. **Certifications**:
    - List any certifications or licenses the candidate holds (if applicable).

7. **Languages**:
    - List any languages spoken and the proficiency level if mentioned.

**Important Instructions**:
- **Do not hallucinate any information**. Only include details that are explicitly mentioned in the resume.
- If a field is not mentioned in the resume, leave it as `null` or an empty list `[]`.
- If any information is unclear or missing, provide a reasonable interpretation based on context, but **do not invent details**.
- The information should be structured in the following JSON format:
    Example:
    {{
        "name": "John Doe",
        "contact": {{
            "phone": "123-456-7890",
            "email": "john.doe@example.com"
        }},
        "skills": ["Python", "Django", "Machine Learning"],
        "education": [
            {{
                "degree": "BSc Computer Science",
                "institution": "XYZ University",
                "year": "2020"
            }}
        ],
        "work_experience": [
            {{
                "job_title": "Software Developer",
                "company": "ABC Corp",
                "duration": "2 years",
                "responsibilities": [
                    "Developed web applications",
                    "Collaborated with cross-functional teams"
                ]
            }}
        "certifications": ["AWS Certified Developer"],
        "languages": ["English", "Spanish"]
    }}
        ],

    Resume: {parsed_data}
    """

    # Send the request to the ChatGroq model for processing
    messages = [{"role": "user", "content": prompt}]
    response = llm.invoke(messages)  # Now using the messages format

    # Print the raw response content for debugging
    print("Raw response from LLM:", response.content)  # Print out the raw response

    # Ensure that the response is not empty and is in the expected format
    if not response or not hasattr(response, 'content'):
        print("Error: Invalid response received.")
        return None

    # Get the response content and parse it
    response_text = response.content.strip()  # Use .content instead of subscripting
    print("Received response from LLM.")

    try:
        # Parse the response into a JSON object
        extracted_data = json.loads(response_text)
        print("Successfully extracted resume details.")
        return extracted_data
    except json.JSONDecodeError:
        print(f"Error decoding the LLM response into JSON. Raw response: {response_text}")
        return None

# Function to process resumes, parse them, and extract key features using LlamaParse and ChatGroq
def process_and_extract_resume_details(files):
    """
    Processes resume files, extracts key details from each using LlamaParse and ChatGroq.
    """
    print(f"Processing resumes from files: {files}")
    
    # Load and parse documents using LlamaParse
    file_extractor = {".txt": parser, ".json": parser, ".pdf": parser}  # Include additional file types as needed
    documents = SimpleDirectoryReader(input_files=files, file_extractor=file_extractor).load_data()
    print(f"Loaded {len(documents)} documents.")

    all_resume_details = []

    # Loop through the loaded documents
    for doc in documents:
        print(f"Processing document: {doc}")
        
        # Access document text directly
        document_text = doc.text if hasattr(doc, 'text') else ""
        print(f"Document content (first 100 chars): {document_text[:100]}...")

        # Use LLM to extract key features from the document text
        resume_details = extract_key_features_from_resume(document_text)
        if resume_details:
            all_resume_details.append(resume_details)

    # Save the extracted resume details to a JSON file
    save_json_to_file(all_resume_details, "extracted_resume_details.json")

    return all_resume_details

# Now let's invoke the function to process the resumes
# Function to save parsed data to a JSON file
def save_json_to_file(data, file_path):
    """
    Saves the given data to a JSON file at the specified file path.
    """
    try:
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Successfully saved parsed data to {file_path}")
    except Exception as e:
        print(f"Error saving data to file {file_path}: {e}")