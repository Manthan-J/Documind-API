from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
import uvicorn
import csv
import io
import pypdf
import os
from groq import Groq
from dotenv import load_dotenv
from chunk import split_text_into_chunks
# Load environment variables from the .env file
load_dotenv()

# Initialize the FastAPI application
app = FastAPI(
    title="Chunking API",
    description="An API to split large texts, parse files, and query LLMs."
)
# Initialize the Groq client
groq_client = Groq()

class TextSplitRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The largest input to be split.")
    chunk_size: int = Field(default=50, gt=0, description="Maximum size of a chunk.")
    chunk_overlap: int = Field(default=10, ge=0, description="Overlap between chunks.")

class LLMQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The question or prompt to send to the LLM.")

#endpoints
@app.get("/")
async def welcome():
    # returns a welcome message
    return {"message": "Welcome to the Chunking API!"}

@app.post("/split-text")
# returns the chunks
async def split_text(request: TextSplitRequest):
    # Overlap cannot be strictly greater than or equal to chunk size
    if request.chunk_overlap >= request.chunk_size:
        raise HTTPException(
            status_code=400, 
            detail="chunk_overlap must be strictly less than chunk_size"
        )
    try:
        # Call the isolated function from chunk.py
        chunks = split_text_into_chunks(
            text=request.text, 
            chunk_size=request.chunk_size, 
            chunk_overlap=request.chunk_overlap
        )
        return {
            "total_chunks_created": len(chunks),
            "generated_chunks": chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while splitting text: {str(e)}")


#upload a csv file and returns text
@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .csv file.")
    try:
        contents = await file.read()
        decoded_string = contents.decode('utf-8')
        reader = csv.reader(io.StringIO(decoded_string))
        text_output = "\n".join([", ".join(row) for row in reader])
        
        return {
            "filename": file.filename,
            "text": text_output
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV file: {str(e)}")


# upload pdf and returns text
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .pdf file.")
    
    try:
        contents = await file.read()
        pdf_stream = io.BytesIO(contents)
        pdf_reader = pypdf.PdfReader(pdf_stream)
        extracted_text = ""
        
        for page in pdf_reader.pages:
            extracted_text += page.extract_text() + "\n\n"
            
        return {
            "filename": file.filename,
            "total_pages": len(pdf_reader.pages),
            "text": extracted_text.strip()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF file: {str(e)}")


# LLM Query using Groq
@app.post("/query-llm")
async def query_llm(request: LLMQueryRequest):
    """
    Accepts a text query, sends it to the Groq API, 
    and returns the LLM's response.
    """
    try:
        # Verify the API key exists
        if not os.environ.get("GROQ_API_KEY"):
            raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set in the environment.")

        # Send the query to Groq
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": request.query,
                }
            ],
            model="llama-3.3-70b-versatile", 
        )
        
        # Extract and return the response text
        return {
            "query": request.query,
            "response": chat_completion.choices[0].message.content
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying Groq: {str(e)}")

# execution
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)