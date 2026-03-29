import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rag import pipeline

app = FastAPI(title="Multimodal RAG API")

# 1. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Static Files
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def index(): 
    if os.path.exists("frontend/index.html"):
        return FileResponse("frontend/index.html")
    return {"message": "API Running. Frontend not found."}

@app.get("/style.css")
def css(): return FileResponse("frontend/style.css")

# 3. Request Models
class QueryReq(BaseModel):
    query: str

# 4. Endpoints
@app.post("/query")
def query_endpoint(req: QueryReq):
    try:
        print(f"DEBUG: Handling query: {req.query}")
        res = pipeline.handle_query(req.query)
        print(f"DEBUG: Query response generated")
        return res
    except Exception as e:
        print(f"ERROR: Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Must be a PDF file.")
    
    content = await file.read()
    print(f"DEBUG: Received PDF '{file.filename}' - Size: {len(content)} bytes")
    if not content:
        raise HTTPException(400, "Received empty file content.")
        
    try:
        res = pipeline.add_document(content, file.filename, "pdf")
        return {"message": f"Successfully indexed {res['chunks']} chunks from {file.filename}."}
    except ValueError as ve:
        print(f"VALIDATION ERROR: {ve}")
        raise HTTPException(400, str(ve))
    except Exception as e:
        print(f"ERROR: PDF upload failed: {e}")
        raise HTTPException(500, f"Processing Error: {str(e)}")

@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    content = await file.read()
    try:
        res = pipeline.add_document(content, file.filename, "image")
        if res:
            return {
                "message": "Image analyzed and added to knowledge base.",
                "caption": res.get("caption")
            }
        raise HTTPException(400, "Could not process image.")
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/reset")
def reset_system():
    """Clears vector store, uploaded files, and all in-memory state."""
    try:
        result = pipeline.reset_all()
        return result
    except Exception as e:
        raise HTTPException(500, f"Reset failed: {str(e)}")
