from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
import uvicorn
import csv
import io
from chunk import split_text_into_chunks

# Initialize the FastAPI application
app = FastAPI(
    title="Chunking API",
    description="An API to split large texts into chunks using Langchain."
)

class TextSplitRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The largest input to be split.")
    chunk_size: int = Field(default=50, gt=0, description="Maximum size of a chunk.")
    chunk_overlap: int = Field(default=10, ge=0, description="Overlap between chunks.")

#endpoints
@app.get("/")
async def welcome():
    #returns a welcome message
    return {"message": "Welcome to the Chunking API!"}

@app.post("/split-text")
#returns the chunks
async def split_text(request: TextSplitRequest):
    #Overlap cannot be strictly greater than or equal to chunk size
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
        # Return the required format
        return {
            "total_chunks_created": len(chunks),
            "generated_chunks": chunks
        }
    except Exception as e:
        #error handling
        raise HTTPException(status_code=500, detail=f"An error occurred while splitting text: {str(e)}")

#extra feature :upload csv and returns text
@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """
    Accepts a CSV file upload, reads the content, 
    and returns it formatted as a single plain text string.
    """
    # Ensure it's a CSV
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .csv file.")
    try:
        # Read and decode the file asynchronously
        contents = await file.read()
        decoded_string = contents.decode('utf-8')
        # Parse the CSV and format it as raw text
        reader = csv.reader(io.StringIO(decoded_string))
        text_output = "\n".join([", ".join(row) for row in reader])
        
        return {
            "filename": file.filename,
            "text": text_output
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV file: {str(e)}")
#execution
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)