import pandas as pd
from io import BytesIO
import unicodedata
from fastapi import FastAPI, File, UploadFile, HTTPException
app = FastAPI()
@app.post("/extract/")
async def extract_text(file: UploadFile = File(...)):
    file_extension = file.filename.split(".")[-1].lower()

    if file_extension == "txt":
        extracted = (await file.read()).decode("utf-8", errors="ignore")
        extracted = unicodedata.normalize("NFKD", extracted).encode("ascii", "ignore").decode("ascii")

    elif file_extension == "csv":
        df = pd.read_csv(BytesIO(await file.read()))
        extracted = df.to_string()

    elif file_extension in ["xlsx", "xls"]:
        file_bytes = await file.read()
        all_sheets = pd.read_excel(BytesIO(file_bytes), sheet_name=None)
        dfs = [df for _, df in all_sheets.items()]
        df = pd.concat(dfs, ignore_index=True)
        extracted = df.to_string()

    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    return {"text": extracted[:500]}  # preview first 500 chars
