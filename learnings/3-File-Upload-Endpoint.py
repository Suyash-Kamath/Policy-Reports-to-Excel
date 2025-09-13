from fastapi import FastAPI, File,Form,HTTPException,UploadFile
import uvicorn

app = FastAPI()

@app.post("/upload/")
async def upload_file(file:UploadFile = File(...),company_name:str=Form(...)):
    if not file.filename:
        raise HTTPException(status_code=400,detail="No file uploaded")
    content= await file.read()
    if not content:
        raise HTTPException(status_code=400,detail="Empty file uploaded")
    return {
        "filename":file.filename,
        "content_type":file.content_type,
        "company_name":company_name,
        "file_size":len(content)
    }

if __name__=="__main__":
    uvicorn.run('3-File-Upload-Endpoint:app', host="127.0.0.1", port=8000, reload=True)