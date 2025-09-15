from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import BytesIO
import base64
import json
import os
from dotenv import load_dotenv
import logging

# Check if openai package is available
try:
    from openai import OpenAI
except ImportError:
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
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://127.0.0.1:5500",
        "https://report-to-excel.vercel.app"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load OpenAI API key
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


# 🔹 Helper function: extract text/formulas from files
async def extract_text_from_file(file: UploadFile, is_formula: bool = False) -> str:
    file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
    file_type = file.content_type if file.content_type else file_extension

    # Image-based extraction
    image_extensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']
    if file_extension in image_extensions or file_type.startswith('image/'):
        image_bytes = await file.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        prompt = "Extract all formulas accurately." if is_formula else "Extract all insurance policy text accurately."
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/{file_extension};base64,{image_base64}"}}
                ]
            }],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()

    # PDF extraction
    if file_extension == 'pdf':
        pdf_bytes = await file.read()
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
        prompt = "Extract formulas or rules from this PDF." if is_formula else "Extract insurance policy details from this PDF."
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:application/pdf;base64,{pdf_base64}"}}
                ]
            }],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()

    # Text
    if file_extension == 'txt':
        return (await file.read()).decode('utf-8', errors='ignore')

    # CSV
    if file_extension == 'csv':
        df = pd.read_csv(BytesIO(await file.read()), encoding='utf-8', errors='ignore')
        return df.to_string()

    # Excel
    if file_extension in ['xlsx', 'xls']:
        file_bytes = await file.read()
        all_sheets = pd.read_excel(BytesIO(file_bytes), sheet_name=None)
        dfs = []
        for sheet_name, df_sheet in all_sheets.items():
            df_sheet["Source_Sheet"] = sheet_name
            dfs.append(df_sheet)
        df = pd.concat(dfs, ignore_index=True, join="outer")
        return df.to_string(index=False)

    raise HTTPException(status_code=400, detail=f"Unsupported file type for {file.filename}")


# ---- Helper: classify Payin ----
def classify_payin(payin_str):
    """
    Converts Payin string (e.g., '50%') to float and classifies its range.
    """
    payin_value = float(payin_str.replace('%',''))
    if payin_value <= 20:
        category = "Payin Below 20%"
    elif 21 <= payin_value <= 30:
        category = "Payin 21% to 30%"
    elif 31 <= payin_value <= 50:  # explicitly include 50%
        category = "Payin 31% to 50%"
    else:
        category = "Payin from 51%"
    return payin_value, category


@app.post("/process-file/")
async def process_file(
    file: UploadFile = File(...),
    formula_file: UploadFile = File(...),
    company_name: str = Form("Unknown Company")
):
    try:
        logger.info(f"Processing policy file: {file.filename}, formula file: {formula_file.filename}")

        # ---- Extract text ----
        extracted_text = await extract_text_from_file(file, is_formula=False)
        extracted_formula_text = await extract_text_from_file(formula_file, is_formula=True)

        # ---- Parse policy data ----
        parse_prompt = f"""
        Analyze the following text, which contains insurance policy details.

        Extract into JSON records with fields:
        - Segment
        - Location
        - Policy Type
        - Payin (convert payout values to percentage format, e.g. 0.5 → 50%, 34 → 34%)
        - Doable District
        - Remarks

        Text: {extracted_text}
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract policy data as JSON. Always rename 'Payout' → 'Payin'. Normalize all Payin values to percentage format (e.g., 0.5 → 50%, 50 → 50%)."},
                {"role": "user", "content": parse_prompt}
            ],
            temperature=0.1
        )
        parsed_json = response.choices[0].message.content.strip()
        if parsed_json.startswith('```json'):
            parsed_json = parsed_json[7:-3].strip()
        policy_data = json.loads(parsed_json)

        if not isinstance(policy_data, list):
            raise ValueError("Parsed policy data is not a list")

        # ---- Pre-classify Payin values ----
        for record in policy_data:
            payin_val, payin_cat = classify_payin(record['Payin'])
            record['Payin_Value'] = payin_val
            record['Payin_Category'] = payin_cat

        # ---- Apply formulas (Payin - Formula%) ----
        formula_prompt = f"""
        You are given:
        - Company Name: {company_name}
        - Policy Data (with company's Payin values and categories): {json.dumps(policy_data, indent=2)}
        - Formula Rules: {extracted_formula_text}

        Task:
        For each policy record:
        1. Match the correct formula rule using Segment, Location, Policy Type, Doable District, and Payin_Category.
        2. Extract the formula percentage.
        3. Calculate: Calculated Payout = Payin_Value - Formula%.
        4. Round down Calculated Payout to the nearest whole number (e.g., 19.5 → 19).
        5. Keep original 'Payin' unchanged.
        6. Include 'Calculated Payout' and 'Formula Used' in each record.
        Return only a JSON array with all original fields plus 'Calculated Payout' and 'Formula Used', percentages formatted.
        """
        formula_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a strict financial calculator. Output only valid JSON array. Ensure Calculated Payout is in percentage format."},
                {"role": "user", "content": formula_prompt}
            ],
            temperature=0.0
        )
        calc_json = formula_response.choices[0].message.content.strip()
        if calc_json.startswith('```json'):
            calc_json = calc_json[7:-3].strip()
        calculated_data = json.loads(calc_json)

        # ---- Create Excel ----
        df_calc = pd.DataFrame(calculated_data)
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_calc.to_excel(writer, sheet_name='Policy Data', startrow=2, index=False)
            worksheet = writer.sheets['Policy Data']

            headers = list(df_calc.columns)
            for col_num, value in enumerate(headers, 1):
                worksheet.cell(row=3, column=col_num, value=value)
                worksheet.cell(row=3, column=col_num).font = worksheet.cell(row=3, column=col_num).font.copy(bold=True)

            company_cell = worksheet.cell(row=1, column=1, value=company_name)
            worksheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
            company_cell.font = company_cell.font.copy(bold=True, size=14)
            company_cell.alignment = company_cell.alignment.copy(horizontal='center')

            title_cell = worksheet.cell(row=2, column=1, value='Policy Data with Payin and Calculated Payouts')
            worksheet.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(headers))
            title_cell.font = title_cell.font.copy(bold=True, size=12)
            title_cell.alignment = title_cell.alignment.copy(horizontal='center')

        output.seek(0)
        excel_base64 = base64.b64encode(output.read()).decode('utf-8')

        return JSONResponse(content={
            "extracted_text": extracted_text,
            "parsed_data": policy_data,
            "calculated_data": calculated_data,
            "excel_file": excel_base64
        })

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)} with raw response: {parsed_json}")
        raise HTTPException(status_code=500, detail=f"Failed to parse structured JSON: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
