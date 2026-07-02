# bring in our LLAMA_CLOUD_API_KEY
from dotenv import load_dotenv
load_dotenv()

# bring in deps
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

# set up parser
parser = LlamaParse(
    result_type="markdown"  # "markdown" and "text" are available
)

# use SimpleDirectoryReader to parse our file
file_extractor = {".docx": parser}
documents = SimpleDirectoryReader(input_files=['/home/abdulsamad/dhiwise_project/resume.docx'], file_extractor=file_extractor).load_data()
print(documents)