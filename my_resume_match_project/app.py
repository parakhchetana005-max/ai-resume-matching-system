import json
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import os
import asyncio
from io import BytesIO

# Import the actual processing functions
from jd_parser import process_and_extract_job_details
from resume_parser import process_and_extract_resume_details
from matched_score import calculate_match_with_llm

app = FastAPI()

# Temporary directory to save uploaded files
UPLOAD_DIR = "./temp_uploaded_files"

# Ensure the directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Function to save uploaded file
async def save_uploaded_file(uploaded_file: UploadFile, destination: Path):
    with open(destination, "wb") as f:
        f.write(await uploaded_file.read())

# API endpoint to receive job description and resume files, process them, and return the match report
@app.post("/match")
async def match_files(
    job_desc_files: List[UploadFile] = File(...),
    resume_files: List[UploadFile] = File(...),
):
    try:
        # Save uploaded job description files asynchronously
        job_desc_file_paths = []
        for job_file in job_desc_files:
            job_desc_path = Path(UPLOAD_DIR) / job_file.filename
            await save_uploaded_file(job_file, job_desc_path)
            job_desc_file_paths.append(job_desc_path)

        # Save uploaded resume files asynchronously
        resume_file_paths = []
        for resume_file in resume_files:
            resume_path = Path(UPLOAD_DIR) / resume_file.filename
            await save_uploaded_file(resume_file, resume_path)
            resume_file_paths.append(resume_path)

        # Offload processing to a separate thread to avoid blocking the event loop
        job_requirements = await asyncio.to_thread(process_and_extract_job_details, job_desc_file_paths)

        # Log the job description processing result
        print("Job requirements extracted:", job_requirements)

        resume_details = await asyncio.to_thread(process_and_extract_resume_details, resume_file_paths)

        # Log the resume processing result
        print("Resume details extracted:", resume_details)

        # Check if job requirements and resume details were extracted successfully
        if not job_requirements or not resume_details:
            raise HTTPException(status_code=400, detail="Error extracting job or resume details.")

        # Calculate match score between job description and resume
        job_desc_json = job_requirements[0]
        resume_json = resume_details[0]

        match_report = calculate_match_with_llm(job_desc_json, resume_json)

        # Log match report for debugging
        print("Match report:", match_report)

        # Return the match report as a JSON response
        return {"match_report": match_report}

    except Exception as e:
        print("Error while processing files:", str(e))
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
