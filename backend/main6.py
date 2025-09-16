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
import re
import math

# Check if required packages are available
try:
    from openai import OpenAI
except ImportError:
    logging.error("OpenAI package not found. Please install it using 'pip install openai'")
    raise ImportError("OpenAI package not found. Please install it using 'pip install openai'")

try:
    import PyPDF2
except ImportError:
    logging.warning("PyPDF2 not found. PDF text extraction will use OpenAI vision only.")
    PyPDF2 = None

try:
    import pdfplumber
except ImportError:
    logging.warning("pdfplumber not found. PDF text extraction will use PyPDF2 or OpenAI vision only.")
    pdfplumber = None

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


def extract_text_from_pdf_file(pdf_bytes: bytes) -> str:
    """Extract text from PDF using multiple methods"""
    extracted_text = ""
    
    if pdfplumber:
        try:
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += page_text + "\n"
            if extracted_text.strip():
                logger.info("PDF text extracted using pdfplumber")
                return extracted_text.strip()
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {str(e)}")
    
    if PyPDF2:
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n"
            if extracted_text.strip():
                logger.info("PDF text extracted using PyPDF2")
                return extracted_text.strip()
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {str(e)}")
    
    logger.warning("All PDF text extraction methods failed")
    return ""


async def extract_text_from_file(file: UploadFile, is_formula: bool = False) -> str:
    file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
    file_type = file.content_type if file.content_type else file_extension

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

    if file_extension == 'pdf':
        pdf_bytes = await file.read()
        
        pdf_text = extract_text_from_pdf_file(pdf_bytes)
        
        if pdf_text and len(pdf_text.strip()) > 50:
            logger.info("Using direct PDF text extraction")
            return pdf_text
        else:
            logger.info("Using OpenAI vision for PDF")
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

    if file_extension == 'txt':
        return (await file.read()).decode('utf-8', errors='ignore')

    if file_extension == 'csv':
        df = pd.read_csv(BytesIO(await file.read()), encoding='utf-8', errors='ignore')
        return df.to_string()

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


def clean_json_response(response_text: str) -> str:
    cleaned = re.sub(r'```json\s*', '', response_text)
    cleaned = re.sub(r'```\s*$', '', cleaned)
    
    json_start = -1
    for i, char in enumerate(cleaned):
        if char in '[{':
            json_start = i
            break
    
    if json_start != -1:
        cleaned = cleaned[json_start:]
    
    json_end = -1
    for i in range(len(cleaned) - 1, -1, -1):
        if cleaned[i] in ']}':
            json_end = i + 1
            break
    
    if json_end != -1:
        cleaned = cleaned[:json_end]
    
    return cleaned.strip()


def ensure_list_format(data) -> list:
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return [data]
    else:
        raise ValueError(f"Expected list or dict, got {type(data)}")


def classify_payin(payin_str):
    try:
        payin_clean = str(payin_str).replace('%', '').replace(' ', '')
        payin_value = float(payin_clean)
        
        if payin_value <= 20:
            category = "Payin Below 20%"
        elif 21 <= payin_value <= 30:
            category = "Payin 21% to 30%"
        elif 31 <= payin_value <= 50:
            category = "Payin 31% to 50%"
        else:
            category = "Payin from 51%"
        return payin_value, category
    except (ValueError, TypeError):
        logger.warning(f"Could not parse payin value: {payin_str}")
        return 0.0, "Payin Below 20%"


def validate_calculated_payout(record):
    """Validate and correct Calculated Payout based on Formula Used"""
    payin_value = record.get('Payin_Value', 0.0)
    formula_used = record.get('Formula Used', '')
    
    try:
        if formula_used.startswith('-') and formula_used.endswith('%'):
            deduction = float(formula_used.replace('%', ''))
            expected_payout = payin_value + deduction  # e.g., -5% means subtract 5
            expected_payout = math.floor(expected_payout)  # Round down
            if str(expected_payout) + '%' != record.get('Calculated Payout'):
                logger.warning(f"Calculation mismatch for record {record['Segment']}: "
                             f"Expected {expected_payout}%, got {record['Calculated Payout']}")
                record['Calculated Payout'] = f"{expected_payout}%"
                record['Remarks'] = (f"{record.get('Remarks', '')} | Calculation corrected: "
                                   f"Applied {formula_used} to Payin_Value {payin_value}").strip()
        elif 'of Payin' in formula_used:
            percentage = float(formula_used.replace('% of Payin', '')) / 100
            expected_payout = payin_value * percentage
            expected_payout = math.floor(expected_payout)
            if str(expected_payout) + '%' != record.get('Calculated Payout'):
                logger.warning(f"Calculation mismatch for record {record['Segment']}: "
                             f"Expected {expected_payout}%, got {record['Calculated Payout']}")
                record['Calculated Payout'] = f"{expected_payout}%"
                record['Remarks'] = (f"{record.get('Remarks', '')} | Calculation corrected: "
                                   f"Applied {formula_used} to Payin_Value {payin_value}").strip()
    except Exception as e:
        logger.error(f"Validation failed for record {record['Segment']}: {str(e)}")
        record['Remarks'] = (f"{record.get('Remarks', '')} | Validation error: {str(e)}").strip()
    
    return record


@app.post("/process-file/")
async def process_file(
    file: UploadFile = File(...),
    formula_file: UploadFile = File(...),
    company_name: str = Form("Unknown Company")
):
    try:
        logger.info(f"Processing policy file: {file.filename}, formula file: {formula_file.filename}")

        extracted_text = await extract_text_from_file(file, is_formula=False)
        extracted_formula_text = await extract_text_from_file(formula_file, is_formula=True)

        logger.info(f"Extracted text length: {len(extracted_text)}")
        logger.info(f"Extracted formula text length: {len(extracted_formula_text)}")
        logger.info(f"Formula grid content: {extracted_formula_text[:500]}...")  # Log formula grid for debugging

        parse_prompt = f"""
        Analyze the following text, which contains insurance policy details.
        Use your intelligence to identify and extract the data accurately.

        Company Name: {company_name}

        IMPORTANT: Always return a valid JSON array (list) of records, even if there's only one record.

        Extract into JSON records with these exact fields:
        - "Segment": identify LOB like TW, PVT CAR, CV, BUS, TAXI, MISD and policy type like TP, COMP, SAOD
        - "Location": location information
        - "Policy Type": policy type information (e.g., TP, COMP, SAOD). If not specified, use "COMP/TP" instead of "N/A"
        - "Payin": convert payout values to percentage format, e.g. 0.5 → 50%, 34 → 34%
        - "Doable District": district information
        - "Remarks": any additional information

        Be intelligent in identifying segments even if the format varies.
        Look for patterns like:
        - TW (Two Wheeler) related segments
        - PVT CAR (Private Car) segments  
        - CV (Commercial Vehicle) segments
        - BUS segments
        - TAXI segments
        - MISD (Miscellaneous) segments

        If you cannot find specific information for any field (except Policy Type), use "N/A" or reasonable defaults. For Policy Type, use "COMP/TP" if not specified.

        Text to analyze:
        {extracted_text}
        
        Return ONLY a valid JSON array, no other text.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract policy data as JSON array. Always rename 'Payout' → 'Payin'. Normalize all Payin values to percentage format. Use intelligence to identify segments accurately. For Policy Type, use 'COMP/TP' if not specified instead of 'N/A'. Return ONLY valid JSON array."},
                {"role": "user", "content": parse_prompt}
            ],
            temperature=0.1
        )
        
        parsed_json = response.choices[0].message.content.strip()
        logger.info(f"Raw OpenAI response: {parsed_json[:500]}...")
        
        cleaned_json = clean_json_response(parsed_json)
        logger.info(f"Cleaned JSON: {cleaned_json[:500]}...")
        
        try:
            policy_data = json.loads(cleaned_json)
            policy_data = ensure_list_format(policy_data)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            logger.error(f"Problematic JSON: {cleaned_json}")
            policy_data = [{
                "Segment": "Unknown",
                "Location": "N/A",
                "Policy Type": "COMP/TP",
                "Payin": "0%",
                "Doable District": "N/A",
                "Remarks": f"Error parsing data: {str(e)}"
            }]

        logger.info(f"Successfully parsed {len(policy_data)} records")

        for record in policy_data:
            try:
                payin_val, payin_cat = classify_payin(record.get('Payin', '0%'))
                record['Payin_Value'] = payin_val
                record['Payin_Category'] = payin_cat
            except Exception as e:
                logger.warning(f"Error classifying payin for record: {record}, error: {str(e)}")
                record['Payin_Value'] = 0.0
                record['Payin_Category'] = "Payin Below 20%"

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
        3. Match the company name EXACTLY with {company_name} first:
           - If {company_name} is not found in the grid, check for variations (e.g., "Bajaj Allianz" → "BAJAJ", "SBI General" → "SBI")
           - If no match, use "Rest of Companies" or "All Companies" rules
           - Log a warning in Remarks if a non-exact match or default rule is used
        4. Match the Payin category correctly:
           - "Payin Below 20%" for payin ≤ 20%
           - "Payin 21% to 30%" for payin 21-30%
           - "Payin 31% to 50%" for payin 31-50%
           - "Payin Above 50%" for payin > 50%
        5. Apply the correct formula based on the grid:
           - "-X%" → subtract X from Payin_Value (e.g., "-5%" → Payin_Value - 5)
           - "X% of Payin" → multiply Payin_Value by X/100 (e.g., "90% of Payin" → Payin_Value * 0.9)
           - "Less X of Payin" → subtract X from Payin_Value (e.g., "Less 2 of Payin" → Payin_Value - 2)
        6. Calculate: Calculated Payout = Apply the formula to Payin_Value
        7. Round down Calculated Payout to the nearest whole number using floor
        8. Format as percentage (e.g., 19 → "19%")
        9. Verify the calculation: If the formula is "-X%", ensure Calculated Payout = floor(Payin_Value - X). If "X% of Payin", ensure Calculated Payout = floor(Payin_Value * X/100).

        SPECIAL ATTENTION:
        - For BUS policies, distinguish between SCHOOL BUS and STAFF BUS
        - For TAXI, all companies follow the same payin-category-based rules
        - Ensure company name matches {company_name} exactly or log a warning in Remarks
        - Do NOT make calculation mistakes. Double-check arithmetic.

        Return ONLY a valid JSON array with all original fields plus:
        - 'Calculated Payout' (as percentage string)
        - 'Formula Used' (the exact rule applied)
        - 'Rule Explanation' (detailed explanation: LOB → Segment → Company → Payin Category → Rule)
        """
        
        formula_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an intelligent financial calculator. Use your intelligence to match policy data with formula grid rules accurately. Output ONLY valid JSON array."},
                {"role": "user", "content": formula_prompt}
            ],
            temperature=0.0
        )
        
        calc_json = formula_response.choices[0].message.content.strip()
        logger.info(f"Raw formula response: {calc_json[:500]}...")
        
        calc_json_cleaned = clean_json_response(calc_json)
        
        try:
            calculated_data = json.loads(calc_json_cleaned)
            calculated_data = ensure_list_format(calculated_data)
        except json.JSONDecodeError as e:
            logger.error(f"Formula JSON decode error: {str(e)}")
            logger.error(f"Problematic formula JSON: {calc_json_cleaned}")
            calculated_data = []
            for record in policy_data:
                record_copy = record.copy()
                record_copy['Calculated Payout'] = f"{int(record.get('Payin_Value', 0))}%"
                record_copy['Formula Used'] = "Default (no formula applied)"
                record_copy['Rule Explanation'] = "Formula parsing failed, using original payin value"
                calculated_data.append(record_copy)

        # Validate and correct calculations
        calculated_data = [validate_calculated_payout(record) for record in calculated_data]

        logger.info(f"Successfully calculated {len(calculated_data)} records")

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

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)