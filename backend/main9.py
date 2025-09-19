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
    
    # Method 1: Try pdfplumber first (most accurate)
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
    
    # Method 2: Try PyPDF2 as fallback
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
    
    # Method 3: Return empty string if all methods fail
    logger.warning("All PDF text extraction methods failed")
    return ""


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

    # Enhanced PDF extraction
    if file_extension == 'pdf':
        pdf_bytes = await file.read()
        
        # First try to extract text directly from PDF
        pdf_text = extract_text_from_pdf_file(pdf_bytes)
        
        if pdf_text and len(pdf_text.strip()) > 50:
            # If we got good text extraction, use it
            logger.info("Using direct PDF text extraction")
            return pdf_text
        else:
            # Fallback to OpenAI vision for PDF
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

    # Text files
    if file_extension == 'txt':
        return (await file.read()).decode('utf-8', errors='ignore')

    # CSV files
    if file_extension == 'csv':
        df = pd.read_csv(BytesIO(await file.read()), encoding='utf-8', errors='ignore')
        return df.to_string()

    # Excel files
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
    """Clean and extract JSON from OpenAI response"""
    # Remove markdown code blocks
    cleaned = re.sub(r'```json\s*', '', response_text)
    cleaned = re.sub(r'```\s*$', '', cleaned)
    
    # Remove any text before the first [ or {
    json_start = -1
    for i, char in enumerate(cleaned):
        if char in '[{':
            json_start = i
            break
    
    if json_start != -1:
        cleaned = cleaned[json_start:]
    
    # Remove any text after the last ] or }
    json_end = -1
    for i in range(len(cleaned) - 1, -1, -1):
        if cleaned[i] in ']}':
            json_end = i + 1
            break
    
    if json_end != -1:
        cleaned = cleaned[:json_end]
    
    return cleaned.strip()


def ensure_list_format(data) -> list:
    """Ensure data is in list format"""
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return [data]  # Convert single object to list
    else:
        raise ValueError(f"Expected list or dict, got {type(data)}")


# ---- Helper: classify Payin ----
def classify_payin(payin_str):
    """
    Converts Payin string (e.g., '50%') to float and classifies its range.
    """
    try:
        # Handle various formats
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

        logger.info(f"Extracted text length: {len(extracted_text)}")
        logger.info(f"Extracted formula text length: {len(extracted_formula_text)}")

        # ---- Parse policy data with enhanced segment identification ----
        parse_prompt = f"""
You are given a motor insurance report. 
The report may come from image, CSV, or Excel.

Your task: Extract records using intelligence. Always return a valid JSON array (list) of objects.

⚠️ Important:
- Output structure should remain exactly like the original file’s format, same-to-same.
- Only the "Segment" should be normalized intelligently.

Extract with these exact fields:
- "Company Name": Extract insurer/company name from the text (e.g., ICICI, Bajaj, SBI, Digit, Reliance, Tata, Zuno). 
   - If multiple possible names, pick the most relevant one.
   - If missing, use "{company_name}" as fallback.
- "Segment": Normalize to one of the allowed values:
   * 1+5
   * TW SAOD + COMP
   * TW TP
   * PVT CAR COMP + SAOD
   * PVT CAR TP
   * Upto 2.5 GVW
   * All GVW & PCV 3W, GCV 3W
   * SCHOOL BUS
   * STAFF BUS
   * TAXI
   * MISD, Tractor
- "Location": If state/city not given, use RTO code (e.g., GUJ-1).
- "Policy Type": Can be AOTP, TP, or COMP. If not mentioned, default to COMP/TP.
   - Note: AOTP = TP, OD = SAOD
- "Doable District": Extract district if available.
- "Remarks": Any other unstructured info not fitting above.
- "Payin": If payrates/numbers are given, treat them as Payin. Normalize to percentage format (e.g., 0.5 → 50%).

Text to analyze:
{extracted_text}

Constraints:
- Output must be valid JSON array only.
- Maintain same structure as original file, same field names.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract policy data as JSON array. Always rename 'Payout' → 'Payin'. Normalize all Payin values to percentage format. Use intelligence to identify segments accurately. Return ONLY valid JSON array."},
                {"role": "user", "content": parse_prompt}
            ],
            temperature=0.1
        )
        
        parsed_json = response.choices[0].message.content.strip()
        logger.info(f"Raw OpenAI response: {parsed_json[:500]}...") 
        
        # Clean the JSON response
        cleaned_json = clean_json_response(parsed_json)
        logger.info(f"Cleaned JSON: {cleaned_json[:500]}...")
        
        try:
            policy_data = json.loads(cleaned_json)
            policy_data = ensure_list_format(policy_data)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            logger.error(f"Problematic JSON: {cleaned_json}")
            
            # Fallback: create a basic structure
            policy_data = [{
                "Segment": "Unknown",
                "Location": "N/A",
                "Policy Type": "N/A", 
                "Payin": "0%",
                "Doable District": "N/A",
                "Remarks": f"Error parsing data: {str(e)}"
            }]

        logger.info(f"Successfully parsed {len(policy_data)} records")

        # ---- Pre-classify Payin values ----
        for record in policy_data:
            try:
                payin_val, payin_cat = classify_payin(record.get('Payin', '0%'))
                record['Payin_Value'] = payin_val
                record['Payin_Category'] = payin_cat
            except Exception as e:
                logger.warning(f"Error classifying payin for record: {record}, error: {str(e)}")
                record['Payin_Value'] = 0.0
                record['Payin_Category'] = "Payin Below 20%"

        # ---- Apply formulas with enhanced intelligence ----
        formula_prompt = f"""
You are an intelligent insurance payout calculator. 
You are given:

Company Name (fallback from form): {company_name}
Policy Data (already extracted): {json.dumps(policy_data, indent=2)}
Formula Rules/Grid: {extracted_formula_text}

The formula grid contains LOB (Line of Business):
- TW
- PVT CAR
- CV
- BUS
- TAXI
- MISD

Segments can only be one of these values:
- 1+5
- TW SAOD + COMP
- TW TP
- PVT CAR COMP + SAOD
- PVT CAR TP
- Upto 2.5 GVW
- All GVW & PCV 3W, GCV 3W
- SCHOOL BUS
- STAFF BUS
- TAXI
- MISD, Tractor

TASK:
1. For each record, determine the correct company:
   - Prefer the "Company Name" extracted from text.
   - If missing, use fallback {company_name}.
   - Normalize variations (e.g., "Bajaj Allianz" → BAJAJ, "SBI General" → SBI).
   - ⚠️ If the company name does not match any company listed in the formula grid, then classify it under "All Companies" or "Rest of Companies" as appropriate.

2. Match record with formula grid using:
   - Segment
   - Company Name (or fallback to All Companies/Rest of Companies if not matched)
   - Payin Category (decided from Remarks rules)
   - Formula

3. Interpret Remarks carefully:
   - "Payin Below X%" = Payin ≤ X
   - "Payin X% to Y%" = inclusive range [X, Y]
   - "Payin Above X%" = Payin ≥ (X+1)
   - Remarks like "All Fuel", "Zuno-21", "-" are valid notes, not ranges.

4. Apply the correct formula:
   - "90% of Payin" → multiply Payin_Value × 0.90
   - "88% of Payin" → multiply Payin_Value × 0.88
   - "-2%" → subtract 2
   - "-3%" → subtract 3
   - "-4%" → subtract 4
   - "-5%" → subtract 5
   - "Less 2 of Payin" → Payin_Value - 2
   - General rule: Payout = Payin - Formula(PO)% if such notation used.

5. Return each record with:
   - Original fields (Company Name, Segment, Location, Policy Type, Doable District, Remarks, Payin)
   - "Calculated Payout": percentage string (rounded down, e.g., 19 → "19%")
   - "Formula Used": the applied formula text
   - "Rule Explanation": clear reasoning chain (Segment → Company/All Companies → Payin Category → Formula → Result)

Constraints:
- Only output valid JSON.
- Preserve structure same-to-same as original file.
- Do not miscalculate formulas.
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
        
        # Clean and parse formula response
        calc_json_cleaned = clean_json_response(calc_json)
        
        try:
            calculated_data = json.loads(calc_json_cleaned)
            calculated_data = ensure_list_format(calculated_data)
        except json.JSONDecodeError as e:
            logger.error(f"Formula JSON decode error: {str(e)}")
            logger.error(f"Problematic formula JSON: {calc_json_cleaned}")
            
            # Fallback: use original data with default calculated values
            calculated_data = []
            for record in policy_data:
                record_copy = record.copy()
                record_copy['Calculated Payout'] = f"{int(record.get('Payin_Value', 0))}%"
                record_copy['Formula Used'] = "Default (no formula applied)"
                record_copy['Rule Explanation'] = "Formula parsing failed, using original payin value"
                calculated_data.append(record_copy)

        logger.info(f"Successfully calculated {len(calculated_data)} records")

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

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
