import streamlit as st
import json
import os
from pathlib import Path
from jd_parser import process_and_extract_job_details
from resume_parser import process_and_extract_resume_details
from matched_score import calculate_match_with_llm

# Function to handle file uploads
def upload_files(file_type):
    """
    This function creates a file uploader widget to allow the user to upload files.
    Depending on the file type (job description or resume), it saves the file in a temporary location.
    """
    uploaded_files = st.file_uploader(f"Upload your {file_type}(s)", type=["pdf", "txt", "json"], accept_multiple_files=True)
    
    if uploaded_files:
        # Define the directory path to save uploaded files
        directory = Path(f"./temp/{file_type}_files/")
        
        # Ensure that the directory exists, if not, create it
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = directory / uploaded_file.name
            file_paths.append(file_path)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
        
        return file_paths
    return []

# Function to display match results
def display_match_report(match_report):
    """
    Display the match report in a human-readable format on Streamlit.
    """
    st.subheader("Match Report")
    if match_report:
        # Display match percentages
        st.markdown(f"**Skills Match**: {match_report.get('skills_match_percentage', 0)}%")
        st.markdown(f"**Experience Match**: {match_report.get('experience_match_percentage', 0)}%")
        st.markdown(f"**Education Match**: {match_report.get('education_match_percentage', 0)}%")
        st.markdown(f"**Technologies Match**: {match_report.get('technologies_match_percentage', 0)}%")
        st.markdown(f"**Overall Match Score**: {match_report.get('overall_match_score', 0)}%")
        
        # Display the detailed analysis of the match
        st.text_area("Detailed Match Analysis", match_report.get('analysis', ''), height=300)

        # Optionally, display the raw JSON of the match report
        st.json(match_report)
    else:
        st.error("Error: Could not calculate the match score.")

# Streamlit app structure
def main():
    st.title("Resume & Job Description Matcher")

    # Step 1: Upload Job Description and Resume Files
    st.header("Step 1: Upload Your Job Description and Resume Files")

    job_desc_files = upload_files("job description")
    resume_files = upload_files("resume")

    # Check if files were uploaded
    if not job_desc_files:
        st.error("Please upload at least one job description file.")
    if not resume_files:
        st.error("Please upload at least one resume file.")

    if job_desc_files and resume_files:
        # Step 2: Process Uploaded Job Descriptions and Resumes
        st.header("Step 2: Processing Files")

        # Process job descriptions and resumes
        job_requirements = process_and_extract_job_details(job_desc_files)
        resume_details = process_and_extract_resume_details(resume_files)

        if job_requirements and resume_details:
            # Step 3: Calculate Match Score
            st.header("Step 3: Calculating Match Score")
            job_desc_json = job_requirements[0]
            resume_json = resume_details[0]

            # Calculate match score between job description and resume
            match_report = calculate_match_with_llm(job_desc_json, resume_json)

            # Step 4: Display Match Report
            display_match_report(match_report)
            
            # Optionally, allow the user to download the match report as a JSON file
            output_file = 'match_report.json'
            with open(output_file, 'w') as f:
                json.dump(match_report, f, indent=4)
            st.download_button(label="Download Match Report", data=json.dumps(match_report), file_name=output_file, mime="application/json")
        else:
            st.error("Error: Could not extract valid job requirements or resume details.")
    else:
        st.warning("Please upload both job description and resume files to proceed.")

if __name__ == "__main__":
    main()
