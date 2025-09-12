from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import BytesIO
import base64
import json
import unicodedata
import os
import requests
from dotenv import load_dotenv
import logging

# Check if openai package is available
try:
    from openai import OpenAI
except ImportError as e:
    logging.error("OpenAI package not found. Please install it using 'pip install openai'")
    raise ImportError("OpenAI package not found. Please install it using 'pip install openai'")

load_dotenv()
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "http://127.0.0.1:5500", "https://commercial-vehicle-insurance-projec.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable not set")
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Initialize OpenAI client
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    raise ValueError(f"Failed to initialize OpenAI client: {str(e)}")

@app.post("/process-file/")
async def process_file(file: UploadFile = File(...), company_name: str = Form("Unknown Company")):
    try:
        logger.info(f"Processing file: {file.filename}, content_type: {file.content_type}")
        
        if file.size == 0:
            logger.error(f"File {file.filename} is empty")
            raise HTTPException(status_code=400, detail=f"File {file.filename} is empty. Please upload a valid file.")
        
        file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        file_type = file.content_type if file.content_type else file_extension
        logger.info(f"Detected file extension: {file_extension}, content_type: {file_type}")
        
        # Handle image files with OCR
        image_extensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']
        if file_extension in image_extensions or file_type.startswith('image/'):
            try:
                # Read image as bytes and base64 encode
                image_bytes = await file.read()
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                
                # Use OpenAI Vision for OCR
                ocr_prompt = """
                Perform OCR on this image. Extract all the text accurately, preserving table structure if present (e.g., use pipes | or describe rows/columns).
                Focus on insurance policy details like segments, locations, policy types, payouts (PO), doable districts, remarks, etc.
                Output the full extracted text.
                """
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": ocr_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/{file_extension};base64,{image_base64}"}
                                }
                            ]
                        }
                    ],
                    temperature=0.1
                )
                extracted_text = response.choices[0].message.content.strip()
                logger.info(f"OCR extracted text length: {len(extracted_text)}")
                if not extracted_text:
                    logger.warning(f"No text extracted from image {file.filename}")
                    extracted_text = ""
            except Exception as e:
                logger.error(f"OCR processing failed for {file.filename}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to process image {file.filename} with OCR: {str(e)}")
        
        # Handle text, CSV, Excel files
        elif file_extension in ['txt'] or file_type in ['text/plain']:
            extracted_text = (await file.read()).decode('utf-8', errors='ignore')
            extracted_text = unicodedata.normalize('NFKD', extracted_text).encode('ascii', 'ignore').decode('ascii')
        
        elif file_extension == 'csv' or file_type == 'text/csv':
            df = pd.read_csv(BytesIO(await file.read()), encoding='utf-8', errors='ignore')
            extracted_text = df.to_string()
            extracted_text = unicodedata.normalize('NFKD', extracted_text).encode('ascii', 'ignore').decode('ascii')
        
        elif file_extension in ['xlsx', 'xls'] or file_type in [
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.ms-excel'
        ]:
            file_bytes = await file.read()
            all_sheets = pd.read_excel(BytesIO(file_bytes), sheet_name=None)
            dfs = []
            for sheet_name, df_sheet in all_sheets.items():
                df_sheet["Source_Sheet"] = sheet_name
                dfs.append(df_sheet)
            df = pd.concat(dfs, ignore_index=True, join="outer")

            extracted_text = df.to_string(index=False, justify="left", max_cols=None, max_rows=None)
            extracted_text = unicodedata.normalize('NFKD', extracted_text).encode('ascii', 'ignore').decode('ascii')
        
        else:
            logger.error(f"Unsupported file: {file.filename}, extension: {file_extension}, content_type: {file_type}")
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type for {file.filename}. Supported: images (.png, .jpg, etc.), .txt, .csv, .xlsx, .xls."
            )
        
        # ---- Send to OpenAI for structured parsing ----
        parse_prompt = f"""
        Analyze the following text, which contains insurance policy details in tabular or list format
        (e.g., segments, locations, payouts/PO, policy types, doable districts, remarks/applicability).

        ⚠️ IMPORTANT: Do not remove or merge duplicate rows. If two rows look the same, include both separately.

        Extract into a list of records (rows). Each record should represent one entry (e.g., one location or combination).
        Fields for each record:
        - Segment: The segment name (e.g., GCV 7.5 ton-12 ton, PCV 3Wheeler Electric). If not specified, infer or use 'Unknown'.
        - Location: The location/RTO state/district (e.g., West Bengal, Assam). If multiple per row, use the primary one.
        - Policy Type: The policy type (e.g., Comprehensive, Third Party, All).
          • If the value is 'Comp/TP' or 'Comp.+TP' alone, do not include it as a single policy type; instead add "Includes Comp/TP policies" to the Remarks field.
          • If 'AOTP' is present under 'Type Comp/TP', set Policy Type to 'AOTP'. Do not add 'Comp/TP' to Remarks unless additional context requires it.
        - Payout: The payout/PO value should be in percentage. If already in percentage, keep as is.
        - Doable District: The doable district name(s). If mentioned and not "All", prefix with "Applicable for " and list district names (comma-separated if multiple). If "All" or not mentioned, use 'All'.
        - Remarks: Only applicable conditions. Do not include exclusions unless part of applicability. If blank, use ''.
          Also, include any additional fields not mapped to the above in Remarks, prefixed with "Additional: <field>: <value>".

        Ignore irrelevant details like dates (e.g., W.E.F.), transaction type (New/Old), age (All), unless they affect Remarks or districts.
        For tables, extract each row as a separate record, pairing fields correctly.
        For single entries, create one record.

        Output only a valid JSON array of objects with the extracted data. Do not drop duplicates.
        Text: {extracted_text.replace('{', '{{').replace('}', '}}').replace('[', '{{[').replace(']', '}}]')}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise extractor of insurance policy data. Preserve row pairings from tables. Convert payout to percentage format (e.g., 0.5 to 50%, 0.34 to 34%). Output only a valid JSON array with extracted data, no examples or extra text."},
                {"role": "user", "content": parse_prompt.encode('ascii', 'ignore').decode('ascii')}
            ],
            temperature=0.1
        )
        parsed_json = response.choices[0].message.content.strip()
        logger.info(f"Raw parsed JSON response: {parsed_json}")  # Debug log
        if parsed_json.startswith('```json'):
            parsed_json = parsed_json[7:-3].strip()
        
        data = json.loads(parsed_json)
        
        if not isinstance(data, list):
            raise ValueError("Parsed data is not a list of records.")
        
        # Create DataFrame from the list of dicts
        df_data = pd.DataFrame(data)
        
        if df_data.empty:
            raise ValueError("No data extracted from the input.")

        # ---- Create Excel output ----
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_data.to_excel(writer, sheet_name='Policy Data', startrow=2, index=False)
            workbook = writer.book
            worksheet = writer.sheets['Policy Data']
            
            # Headers
            headers = ['Segment', 'Location', 'Policy Type', 'Payout', 'Doable District', 'Remarks']
            for col_num, value in enumerate(headers, 1):
                worksheet.cell(row=3, column=col_num, value=value)
                worksheet.cell(row=3, column=col_num).font = worksheet.cell(row=3, column=col_num).font.copy(bold=True)
            
            # Company name spanning 6 columns (Segment to Remarks)
            company_cell = worksheet.cell(row=1, column=1, value=company_name)
            worksheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=6)
            company_cell.font = company_cell.font.copy(bold=True, size=14)
            company_cell.alignment = company_cell.alignment.copy(horizontal='center')
            
            # Optional: Add empty row or title if needed
            title_cell = worksheet.cell(row=2, column=1, value='Policy Data')
            worksheet.merge_cells(start_row=2, start_column=1, end_row=2, end_column=6)
            title_cell.font = title_cell.font.copy(bold=True, size=12)
            title_cell.alignment = title_cell.alignment.copy(horizontal='center')

        output.seek(0)

        excel_base64 = base64.b64encode(output.read()).decode('utf-8')
        return JSONResponse(content={
            "extracted_text": extracted_text,
            "parsed_data": data,
            "excel_file": excel_base64
        })
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error for file {file.filename}: {str(e)} with raw response: {parsed_json}")
        raise HTTPException(status_code=500, detail=f"Failed to parse data into structured format: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error for file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred for file {file.filename}: {str(e)}")
