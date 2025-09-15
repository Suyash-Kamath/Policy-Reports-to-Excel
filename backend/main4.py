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


# 📹 Helper function: extract text/formulas from files
async def extract_text_from_file(file: UploadFile, is_formula: bool = False) -> str:
    file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
    file_type = file.content_type if file.content_type else file_extension

    # Image-based extraction with enhanced OCR
    image_extensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']
    if file_extension in image_extensions or file_type.startswith('image/'):
        image_bytes = await file.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        if is_formula:
            prompt = """Extract all formulas, rules and grid information accurately from this image. 
            Focus on identifying:
            - LOB segments (TW, PVT CAR, CV, BUS, TAXI, MISD)
            - Policy types (TP, COMP, SAOD, etc.)
            - Company names and their specific rules
            - Percentage deductions and formulas
            - Payin categories and their corresponding rates
            Extract everything exactly as shown in the grid/table format."""
        else:
            prompt = """Extract all insurance policy text accurately from this image using OCR.
            Pay special attention to identifying:
            - Segment information (like TW, PVT CAR, CV, BUS, TAXI, MISD)
            - Company names
            - Policy types
            - Payin/Payout percentages
            - Any numerical values
            - Location information
            Extract all text exactly as it appears."""
            
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

    # PDF extraction with enhanced prompts
    if file_extension == 'pdf':
        pdf_bytes = await file.read()
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
        
        if is_formula:
            prompt = """Extract formulas, rules, and grid information from this PDF.
            Look for LOB segments, company-specific rules, percentage deductions, and payin categories."""
        else:
            prompt = """Extract insurance policy details from this PDF.
            Focus on identifying segments, company names, policy types, and payout percentages."""
            
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

        # ---- Extract text with enhanced intelligence ----
        extracted_text = await extract_text_from_file(file, is_formula=False)
        extracted_formula_text = await extract_text_from_file(formula_file, is_formula=True)

        # ---- Parse policy data with enhanced segment identification ----
        parse_prompt = f"""
        Analyze the following text, which contains insurance policy details.
        Use your intelligence to identify and extract the data accurately.

        Company Name: {company_name}

        Extract into JSON records with fields:
        - Segment (identify LOB like TW, PVT CAR, CV, BUS, TAXI, MISD and policy type like TP, COMP, SAOD)
        - Location
        - Policy Type
        - Payin (convert payout values to percentage format, e.g. 0.5 → 50%, 34 → 34%)
        - Doable District
        - Remarks

        Be intelligent in identifying segments even if the format varies.
        Look for patterns like:
        - TW (Two Wheeler) related segments
        - PVT CAR (Private Car) segments  
        - CV (Commercial Vehicle) segments
        - BUS segments
        - TAXI segments
        - MISD (Miscellaneous) segments

        Text: {extracted_text}
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract policy data as JSON. Always rename 'Payout' → 'Payin'. Normalize all Payin values to percentage format. Use intelligence to identify segments accurately."},
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

        # ---- Apply formulas with enhanced intelligence ----
        formula_prompt = f"""
        You are an intelligent insurance formula calculator with access to:
        
        Company Name: {company_name}
        Policy Data: {json.dumps(policy_data, indent=2)}
        Formula Rules/Grid: {extracted_formula_text}

        IMPORTANT GRID STRUCTURE RULES:
        - BUS LOB has two segments: "SCHOOL BUS" and "STAFF BUS" - treat them separately
        - TAXI LOB has one segment: "TAXI" with multiple payin categories
        - PVT CAR has: "PVT CAR COMP + SAOD" and "PVT CAR TP"
        - CV has: "Upto 2.5 GVW" and "All GVW & PCV 3W, GCV 3W"
        - Each segment can have different company rules and payin category rules

        TASK: Use your intelligence to:
        1. Analyze each policy record's Segment and identify the correct LOB (TW, PVT CAR, CV, BUS, TAXI, MISD)
        2. Within the LOB, identify the specific segment (e.g., for BUS: "SCHOOL BUS" vs "STAFF BUS")
        3. Match the company name ({company_name}) with the formula grid:
           - Look for exact matches first (DIGIT, BAJAJ, ICICI, SBI, TATA, RELIANCE)
           - Handle variations (e.g., "Bajaj Allianz" → "BAJAJ", "SBI General" → "SBI")
           - If no specific company match, use "Rest of Companies" or "All Companies"
        4. Match the Payin category correctly:
           - "Payin Below 20%" for payin ≤ 20%
           - "Payin 21% to 30%" for payin 21-30%
           - "Payin 31% to 50%" for payin 31-50%
           - "Payin Above 50%" for payin > 50%
        5. Apply the correct formula based on the grid:
           - "90% of Payin" → multiply Payin_Value by 0.9
           - "88% of Payin" → multiply Payin_Value by 0.88
           - "-2%" → subtract 2 from Payin_Value
           - "-3%" → subtract 3 from Payin_Value
           - "-4%" → subtract 4 from Payin_Value
           - "-5%" → subtract 5 from Payin_Value
           - "Less 2 of Payin" → subtract 2 from Payin_Value
        6. Calculate: Calculated Payout = Apply the formula to Payin_Value
        7. Round down Calculated Payout to the nearest whole number
        8. Format as percentage (e.g., 19 → "19%")

        SPECIAL ATTENTION:
        - For BUS policies, clearly distinguish between SCHOOL BUS and STAFF BUS
        - For TAXI, all companies follow the same payin-category-based rules
        - Pay attention to company-specific rules vs "All Companies" rules

        Return JSON array with all original fields plus:
        - 'Calculated Payout' (as percentage string)
        - 'Formula Used' (the exact rule applied)
        - 'Rule Explanation' (detailed explanation: LOB → Segment → Company → Payin Category → Rule)
        """
        
        formula_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an intelligent financial calculator. Use your intelligence to match policy data with formula grid rules accurately. Output only valid JSON array."},
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