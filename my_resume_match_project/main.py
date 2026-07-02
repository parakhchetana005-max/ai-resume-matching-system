import json
from pathlib import Path
from jd_parser import process_and_extract_job_details
from resume_parser import process_and_extract_resume_details
from matched_score import calculate_match_with_llm

def main():
    # Define the paths to the job description and resume directories or specific files
    job_desc_files = [
        '/home/abdulsamad/dhiwise_project/sample_resume.txt',  # You can specify full paths to files or relative paths
        #'path/to/job_desc_2.json'
    ]
    
    resume_files = [
        '/home/abdulsamad/dhiwise_project/Resume_Samad (1).pdf',
        #'path/to/resume_2.docx'
    ]

    # Convert file paths to Path objects
    job_desc_files = [Path(file) for file in job_desc_files]
    resume_files = [Path(file) for file in resume_files]

    # Print the files being processed
    print(f"Job Description Files: {job_desc_files}")
    print(f"Resume Files: {resume_files}")

    # Check if there are valid job description and resume files
    if not job_desc_files:
        print("Error: No valid job description files found.")
        return
    if not resume_files:
        print("Error: No valid resume files found.")
        return

    # Process job descriptions and resumes
    print(f"Processing {len(job_desc_files)} job description files...")
    job_requirements = process_and_extract_job_details(job_desc_files)

    print(f"Processing {len(resume_files)} resume files...")
    resume_details = process_and_extract_resume_details(resume_files)

    # Check if job requirements and resume details were extracted successfully
    if job_requirements and resume_details:
        # For simplicity, using the first job requirement and resume detail
        job_desc_json = job_requirements[0]
        resume_json = resume_details[0]

        # Calculate match score between job description and resume
        print("Calculating match score...")
        match_report = calculate_match_with_llm(job_desc_json, resume_json)

        # Save the match report to a JSON file
        output_file = 'match_report.json'  # You can change the output file name here
        with open(output_file, 'w') as f:
            json.dump(match_report, f, indent=4)

        print(f"Match report saved to {output_file}")
    else:
        print("Error: Could not extract valid job requirements or resume details.")

if __name__ == "__main__":
    main()
