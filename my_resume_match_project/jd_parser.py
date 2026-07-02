import os
import json
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv
from langchain_groq import ChatGroq  # Importing ChatGroq from langchain_groq

# Load environment variables for LLAMA_CLOUD_API_KEY (for LlamaParse) and API key for ChatGroq
load_dotenv()

# Initialize LlamaParse with the desired output format
parser = LlamaParse(result_type="text")  # or "markdown" depending on your preference

# Initialize ChatGroq with model details and parameters
llm = ChatGroq(
    model="mixtral-8x7b-32768",  # Using the specified model
    temperature=0.0,              # Set temperature to 0 for deterministic output
    max_retries=2,                # Set the number of retries
    # other params can be added as needed
)

# Helper function to interact with LLM (ChatGroq) and extract key features from job descriptions
def extract_key_features_with_llm(parsed_data):
    """
    This function sends the parsed data (job description) to an LLM (ChatGroq) with a specific prompt
    to extract key details like skills, qualifications, experience, and responsibilities.
    """
    print("Sending job description data to LLM for extraction...")

    # Defining the conversation in the required format
    messages = [
        ("system", "You are an expert at analyzing and extracting key details from job descriptions. Your task is to carefully parse the job description provided and extract relevant information in a clear and structured format. Focus on the following details."),
        ("human", f"""
    Please extract and organize the following information from the job description below:

            1. **Skills**:
                - List all technical or non-technical skills mentioned in the job description.
                - Include both hard skills (e.g., programming languages, tools, software) and soft skills (e.g., communication, leadership).

            2. **Qualifications**:
                - List the minimum or preferred qualifications (e.g., degrees, certifications, specific industry knowledge).
                - Include any educational background or certifications required for the role.

            3. **Experience**:
                - If explicitly mentioned, provide the required or preferred years of experience (e.g., "3-5 years of experience").
                - If not explicitly mentioned, you can omit this field or set it to `null`.

            4. **Responsibilities**:
                - List the key responsibilities and tasks for the role, as described in the job description.
                - Use bullet points or short phrases to summarize the core duties.

            **Important Instructions**:
            - **Do not hallucinate any information**. Only include details that are explicitly mentioned in the job description. If any category is not mentioned, leave it as `null` or an empty list `[]`.
            - If any information is vague or unclear, provide a reasonable interpretation based on context, but **do not invent details or make assumptions**.
            - Avoid including generalizations or filler content that is not directly related to the job description.

            **Format** the extracted information in a JSON structure like this example:

            ```json
    Example:
    {{
        "skills": ["skill1", "skill2", "skill3"],
        "qualifications": ["qualification1", "qualification2"],
        "experience_years": "X years",
        "responsibilities": ["responsibility1", "responsibility2"]
    }}

    Job Description: {parsed_data}
    """),
    ]

    # Send the request to the ChatGroq model using invoke() method
    response = llm.invoke(messages)

    # Access the content of the response (correcting the previous access)
    response_text = response.content.strip()  # Using the .content attribute

    print("Received response from LLM.")

    try:
        # Parse the response into a JSON object
        extracted_data = json.loads(response_text)
        return extracted_data
    except json.JSONDecodeError:
        print("Error decoding the LLM response into JSON.")
        return None

# Function to process job descriptions, parse them with LlamaParse, extract details using LLM, and save results
def process_and_extract_job_details(files):
    """
    Processes job description files, extracts key details from each using LlamaParse and ChatGroq.
    """
    print(f"Processing job descriptions from files: {files}")
    
    # Load and parse documents using LlamaParse
    file_extractor = {".txt": parser, ".json": parser, ".pdf": parser}  # Include additional file types as needed
    documents = SimpleDirectoryReader(input_files=files, file_extractor=file_extractor).load_data()
    print(f"Loaded {len(documents)} documents.")

    all_job_requirements = []

    # Loop through the loaded documents
    for doc in documents:
        print(f"Processing document: {doc}")
        
        # Access document text directly
        document_text = doc.text if hasattr(doc, 'text') else ""
        print(f"Document content (first 100 chars): {document_text[:100]}...")

        # Use LLM to extract key features from the document text
        job_requirements = extract_key_features_with_llm(document_text)
        if job_requirements:
            all_job_requirements.append(job_requirements)

    # Save the extracted job description details to a JSON file
    save_json_to_file(all_job_requirements, "extracted_job_details.json")

    return all_job_requirements


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