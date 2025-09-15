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

# September 2025 Grid Rules
SEPTEMBER_2025_GRID = {
    "TW": {
        "1+5": {
            "All Companies": "90% of Payin"
        },
        "TW SAOD + COMP": {
            "DIGIT": {
                "Payin Below 20%": "-2%",
                "Payin 21% to 30%": "-3%",
                "Payin 31% to 50%": "-4%",
                "Payin Above 50%": "-5%"
            }
        },
        "TW TP": {
            "Bajaj, Digit, ICICI": {
                "Payin Below 20%": "-2%",
                "Payin Above 20%": "-3%"
            },
            "Rest of Companies": {
                "Payin Below 20%": "-2%",
                "Payin 21% to 30%": "-3%",
                "Payin 31% to 50%": "-4%",
                "Payin Above 50%": "-5%"
            }
        }
    },
    "PVT CAR": {
        "PVT CAR COMP + SAOD": {
            "All Companies": "90% of Payin"
        },
        "PVT CAR TP": {
            "Bajaj, Digit, SBI": {
                "Payin Below 20%": "-2%",
                "Payin Above 20%": "-3%"
            },
            "Rest of Companies": "90% of Payin"
        }
    },
    "CV": {
        "Upto 2.5 GVW": {
            "Reliance, SBI": {
                "Payin Below 20%": "-2%",
                "Payin Above 20%": "-2%"
            }
        },
        "All GVW & PCV 3W, GCV 3W": {
            "Rest of Companies": {
                "Payin Below 20%": "-2%",
                "Payin 21% to 30%": "-3%",
                "Payin 31% to 50%": "-4%",
                "Payin Above 50%": "-5%"
            }
        }
    },
    "BUS": {
        "SCHOOL BUS": {
            "TATA, Reliance, Digit, ICICI": "Less 2 of Payin",
            "Rest of Companies": "88% of Payin"
        },
        "STAFF BUS": {
            "All Companies": "88% of Payin"
        }
    },
    "TAXI": {
        "TAXI": {
            "All Companies": {
                "Payin Below 20%": "-2%",
                "Payin 21% to 30%": "-3%",
                "Payin 31% to 50%": "-4%",
                "Payin Above 50%": "-5%"
            }
        }
    },
    "MISD": {
        "Misd, Tractor": {
            "All Companies": "88% of Payin"
        }
    }
}

# Helper function: extract text/formulas from files
async def extract_text_from_file(file: UploadFile, is_formula: bool = False) -> str:
    file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
    file_type = file.content_type if file.content_type else file_extension

    # Image-based extraction
    image_extensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']
    if file_extension in image_extensions or file_type.startswith('image/'):
        image_bytes = await file.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        prompt = "Extract all formulas accurately." if is_formula else "Extract all insurance policy text accurately. Focus on identifying segments (like TW, PVT CAR, CV, BUS, TAXI, MISD), company names, policy types, and payout/payin percentages."
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
        prompt = "Extract formulas or rules from this PDF." if is_formula else "Extract insurance policy details from this PDF. Focus on identifying segments, company names, policy types, and payout percentages."
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

def classify_payin(payin_str):
    """
    Converts Payin string (e.g., '50%') to float and classifies its range.
    """
    # Handle various formats
    payin_clean = str(payin_str).replace('%', '').replace(' ', '')
    try:
        payin_value = float(payin_clean)
    except:
        payin_value = 0.0
    
    if payin_value <= 20:
        category = "Payin Below 20%"
    elif 21 <= payin_value <= 30:
        category = "Payin 21% to 30%"
    elif 31 <= payin_value <= 50:
        category = "Payin 31% to 50%"
    else:
        category = "Payin Above 50%"
    
    return payin_value, category

def normalize_company_name(company_name):
    """Normalize company name for matching"""
    company_name = company_name.upper().strip()
    # Common mappings
    mappings = {
        "BAJAJ ALLIANZ": "BAJAJ",
        "DIGIT INSURANCE": "DIGIT",
        "ICICI LOMBARD": "ICICI",
        "SBI GENERAL": "SBI",
        "RELIANCE GENERAL": "RELIANCE",
        "TATA AIG": "TATA"
    }
    for full_name, short_name in mappings.items():
        if full_name in company_name:
            return short_name
    return company_name

def find_matching_rule(lob, segment, company_name, payin_category):
    """
    Find the matching rule from September 2025 grid
    """
    try:
        # Normalize inputs
        lob = lob.upper().strip()
        segment = segment.upper().strip()
        company_name = normalize_company_name(company_name)
        
        # Check if LOB exists in grid
        if lob not in SEPTEMBER_2025_GRID:
            return None, f"LOB '{lob}' not found in grid"
        
        lob_rules = SEPTEMBER_2025_GRID[lob]
        
        # Find matching segment
        matching_segment = None
        for seg_key in lob_rules.keys():
            if seg_key.upper() in segment or segment in seg_key.upper():
                matching_segment = seg_key
                break
        
        if not matching_segment:
            return None, f"Segment '{segment}' not found in LOB '{lob}'"
        
        segment_rules = lob_rules[matching_segment]
        
        # Find matching company rule
        company_rule = None
        
        # Check for "All Companies" first
        if "All Companies" in segment_rules:
            company_rule = segment_rules["All Companies"]
        else:
            # Check for specific company matches
            for company_key, rule in segment_rules.items():
                companies_in_key = [c.strip().upper() for c in company_key.split(',')]
                if company_name in companies_in_key:
                    company_rule = rule
                    break
            
            # If no specific match, try "Rest of Companies"
            if not company_rule and "Rest of Companies" in segment_rules:
                company_rule = segment_rules["Rest of Companies"]
        
        if not company_rule:
            return None, f"No rule found for company '{company_name}' in segment '{matching_segment}'"
        
        # Handle different rule types
        if isinstance(company_rule, str):
            return company_rule, f"Applied rule: {company_rule}"
        elif isinstance(company_rule, dict):
            # Rules based on payin category
            if payin_category in company_rule:
                return company_rule[payin_category], f"Applied rule: {company_rule[payin_category]} for {payin_category}"
            else:
                return None, f"No rule found for payin category '{payin_category}'"
        
        return None, "Unknown rule format"
        
    except Exception as e:
        return None, f"Error finding rule: {str(e)}"

def calculate_payout(payin_value, rule):
    """
    Calculate the final payout based on the rule
    """
    try:
        if "90% of Payin" in rule:
            return payin_value * 0.9
        elif "88% of Payin" in rule:
            return payin_value * 0.88
        elif "Less 2 of Payin" in rule:
            return payin_value - 2
        elif rule.startswith("-"):
            # Handle percentage deduction (e.g., "-2%", "-3%")
            deduction = float(rule.replace("-", "").replace("%", ""))
            return payin_value - deduction
        else:
            # Try to parse as direct percentage
            if "%" in rule:
                percentage = float(rule.replace("%", ""))
                return payin_value * (percentage / 100)
            return payin_value
    except:
        return payin_value

@app.post("/process-file/")
async def process_file(
    file: UploadFile = File(...),
    company_name: str = Form("Unknown Company")
):
    try:
        logger.info(f"Processing policy file: {file.filename}, company: {company_name}")

        # Extract text from uploaded file
        extracted_text = await extract_text_from_file(file, is_formula=False)

        # Use OpenAI to intelligently parse the policy data
        parse_prompt = f"""
        Analyze the following insurance policy text and extract structured data.

        Company Name from form: {company_name}

        The text may contain various formats of insurance policy information. Please extract:
        - LOB (Line of Business): TW, PVT CAR, CV, BUS, TAXI, MISD, etc.
        - Segment: Like "1+5", "TW SAOD + COMP", "TW TP", "PVT CAR COMP + SAOD", "SCHOOL BUS", etc.
        - Company Name: Extract or use the provided company name
        - Policy Type: TP, COMP, SAOD, etc.
        - Payin: Convert any payout values to percentage format (e.g., 0.5 → 50%, 34 → 34%)
        - Location: If mentioned
        - Remarks: Any additional information

        Be intelligent in identifying these fields even if the structure varies.

        Text to analyze:
        {extracted_text}

        Return as JSON array with these fields:
        - LOB
        - Segment  
        - Company
        - Policy_Type
        - Payin
        - Location
        - Remarks
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert insurance data extractor. Extract policy data as JSON array. Be intelligent in identifying LOB, Segment, Company, and other fields even from unstructured text."},
                {"role": "user", "content": parse_prompt}
            ],
            temperature=0.1
        )

        parsed_json = response.choices[0].message.content.strip()
        if parsed_json.startswith('```json'):
            parsed_json = parsed_json[7:-3].strip()
        
        policy_data = json.loads(parsed_json)

        if not isinstance(policy_data, list):
            policy_data = [policy_data]

        # Process each record
        processed_data = []
        for record in policy_data:
            try:
                # Get payin value and category
                payin_str = record.get('Payin', '0%')
                payin_value, payin_category = classify_payin(payin_str)
                
                # Get other fields
                lob = record.get('LOB', '')
                segment = record.get('Segment', '')
                company = record.get('Company', company_name)
                
                # Find matching rule from September 2025 grid
                rule, rule_explanation = find_matching_rule(lob, segment, company, payin_category)
                
                # Calculate payout
                if rule:
                    calculated_payout = calculate_payout(payin_value, rule)
                    calculated_payout = int(calculated_payout)  # Round down to nearest whole number
                else:
                    calculated_payout = payin_value
                    rule_explanation = "No matching rule found"
                
                # Add calculated fields to record
                record['Payin_Value'] = payin_value
                record['Payin_Category'] = payin_category
                record['Formula_Used'] = rule if rule else "No rule applied"
                record['Rule_Explanation'] = rule_explanation
                record['Calculated_Payout'] = f"{calculated_payout}%"
                
                processed_data.append(record)
                
            except Exception as e:
                logger.error(f"Error processing record: {str(e)}")
                record['Formula_Used'] = "Error in calculation"
                record['Rule_Explanation'] = str(e)
                record['Calculated_Payout'] = record.get('Payin', '0%')
                processed_data.append(record)

        # Create Excel output
        df_calc = pd.DataFrame(processed_data)
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_calc.to_excel(writer, sheet_name='Policy Data', startrow=2, index=False)
            worksheet = writer.sheets['Policy Data']

            # Format headers
            headers = list(df_calc.columns)
            for col_num, value in enumerate(headers, 1):
                cell = worksheet.cell(row=3, column=col_num, value=value)
                cell.font = cell.font.copy(bold=True)

            # Add company name header
            company_cell = worksheet.cell(row=1, column=1, value=f"Company: {company_name}")
            worksheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
            company_cell.font = company_cell.font.copy(bold=True, size=14)
            company_cell.alignment = company_cell.alignment.copy(horizontal='center')

            # Add title
            title_cell = worksheet.cell(row=2, column=1, value='Policy Data with Calculated Payouts (September 2025 Grid)')
            worksheet.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(headers))
            title_cell.font = title_cell.font.copy(bold=True, size=12)
            title_cell.alignment = title_cell.alignment.copy(horizontal='center')

        output.seek(0)
        excel_base64 = base64.b64encode(output.read()).decode('utf-8')

        return JSONResponse(content={
            "extracted_text": extracted_text,
            "parsed_data": policy_data,
            "processed_data": processed_data,
            "excel_file": excel_base64,
            "company_name": company_name
        })

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to parse structured JSON: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Insurance Policy Processing API with September 2025 Grid"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)