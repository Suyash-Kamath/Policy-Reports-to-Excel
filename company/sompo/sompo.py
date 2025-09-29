import streamlit as st
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
    st.error("OpenAI package not found. Please install it using 'pip install openai'")
    st.stop()

try:
    import PyPDF2
except ImportError:
    st.warning("PyPDF2 not found. PDF text extraction will use OpenAI vision only.")
    PyPDF2 = None

try:
    import pdfplumber
except ImportError:
    st.warning("pdfplumber not found. PDF text extraction will use PyPDF2 or OpenAI vision only.")
    pdfplumber = None

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY environment variable not set. Please set it in your .env file or environment variables.")
    st.stop()

# Initialize OpenAI client
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize OpenAI client: {str(e)}")
    st.stop()

# Embedded Formula Data
FORMULA_DATA = [
    {"LOB": "TW", "SEGMENT": "1+5", "INSURER": "All Companies", "PO": "90% of Payin", "REMARKS": "NIL"},
    {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "INSURER": "All Companies", "PO": "90% of Payin", "REMARKS": "NIL"},
    {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "INSURER": "DIGIT", "PO": "-2%", "REMARKS": "Payin Below 20%"},
    {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "INSURER": "DIGIT", "PO": "-3%", "REMARKS": "Payin 21% to 30%"},
    {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "INSURER": "DIGIT", "PO": "-4%", "REMARKS": "Payin 31% to 50%"},
    {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "INSURER": "DIGIT", "PO": "-5%", "REMARKS": "Payin Above 50%"},
    {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Bajaj, Digit, ICICI", "PO": "-2%", "REMARKS": "Payin Below 20%"},
    {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Bajaj, Digit, ICICI", "PO": "-3%", "REMARKS": "Payin Above 20%"},
    {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Rest of Companies", "PO": "-2%", "REMARKS": "Payin Below 20%"},
    {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Rest of Companies", "PO": "-3%", "REMARKS": "Payin 21% to 30%"},
    {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Rest of Companies", "PO": "-4%", "REMARKS": "Payin 31% to 50%"},
    {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Rest of Companies", "PO": "-5%", "REMARKS": "Payin Above 50%"},
    {"LOB": "PVT CAR", "SEGMENT": "PVT CAR COMP + SAOD", "INSURER": "All Companies", "PO": "90% of Payin", "REMARKS": "All Fuel"},
    {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "Bajaj, Digit, SBI", "PO": "-2%", "REMARKS": "Payin Below 20%"},
    {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "Bajaj, Digit, SBI", "PO": "-3%", "REMARKS": "Payin Above 20%"},
    {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "Rest of Companies", "PO": "90% of Payin", "REMARKS": "Zuno -  21"},
    {"LOB": "CV", "SEGMENT": "Upto 2.5 GVW", "INSURER": "Reliance, SBI, Tata", "PO": "-2%", "REMARKS": "NIL"},
    {"LOB": "CV", "SEGMENT": "All GVW & PCV 3W, GCV 3W", "INSURER": "Rest of Companies", "PO": "-2%", "REMARKS": "Payin Below 20%"},
    {"LOB": "CV", "SEGMENT": "All GVW & PCV 3W, GCV 3W", "INSURER": "Rest of Companies", "PO": "-3%", "REMARKS": "Payin 21% to 30%"},
    {"LOB": "CV", "SEGMENT": "All GVW & PCV 3W, GCV 3W", "INSURER": "Rest of Companies", "PO": "-4%", "REMARKS": "Payin 31% to 50%"},
    {"LOB": "CV", "SEGMENT": "All GVW & PCV 3W, GCV 3W", "INSURER": "Rest of Companies", "PO": "-5%", "REMARKS": "Payin Above 50%"},
    {"LOB": "BUS", "SEGMENT": "SCHOOL BUS", "INSURER": "TATA, Reliance, Digit, ICICI", "PO": "Less 2% of Payin", "REMARKS": "NIL"},
    {"LOB": "BUS", "SEGMENT": "SCHOOL BUS", "INSURER": "Rest of Companies", "PO": "88% of Payin", "REMARKS": "NIL"},
    {"LOB": "BUS", "SEGMENT": "STAFF BUS", "INSURER": "All Companies", "PO": "88% of Payin", "REMARKS": "NIL"},
    {"LOB": "TAXI", "SEGMENT": "TAXI", "INSURER": "All Companies", "PO": "-2%", "REMARKS": "Payin Below 20%"},
    {"LOB": "TAXI", "SEGMENT": "TAXI", "INSURER": "All Companies", "PO": "-3%", "REMARKS": "Payin 21% to 30%"},
    {"LOB": "TAXI", "SEGMENT": "TAXI", "INSURER": "All Companies", "PO": "-4%", "REMARKS": "Payin 31% to 50%"},
    {"LOB": "TAXI", "SEGMENT": "TAXI", "INSURER": "All Companies", "PO": "-5%", "REMARKS": "Payin Above 50%"},
    {"LOB": "MISD", "SEGMENT": "Misd, Tractor", "INSURER": "All Companies", "PO": "88% of Payin", "REMARKS": "NIL"}
]

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

def extract_text_from_file(file_bytes: bytes, filename: str, content_type: str) -> dict:
    """Extract text from uploaded file, returning a dictionary with sheet-wise data for Excel files"""
    file_extension = filename.split('.')[-1].lower() if '.' in filename else ''
    file_type = content_type if content_type else file_extension

    # Image-based extraction with enhanced OCR
    image_extensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']
    if file_extension in image_extensions or file_type.startswith('image/'):
        image_base64 = base64.b64encode(file_bytes).decode('utf-8')
        
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
        token_usage = response.usage
        st.info(f"Tokens used for image extraction: Prompt: {token_usage.prompt_tokens}, Completion: {token_usage.completion_tokens}, Total: {token_usage.total_tokens}")
        return {"Sheet1": response.choices[0].message.content.strip()}

    # Enhanced PDF extraction
    if file_extension == 'pdf':
        # First try to extract text directly from PDF
        pdf_text = extract_text_from_pdf_file(file_bytes)
        
        if pdf_text and len(pdf_text.strip()) > 50:
            # If we got good text extraction, use it
            logger.info("Using direct PDF text extraction")
            return {"Sheet1": pdf_text}
        else:
            # Fallback to OpenAI vision for PDF
            logger.info("Using OpenAI vision for PDF")
            pdf_base64 = base64.b64encode(file_bytes).decode("utf-8")
            
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
            token_usage = response.usage
            st.info(f"Tokens used for image extraction: Prompt: {token_usage.prompt_tokens}, Completion: {token_usage.completion_tokens}, Total: {token_usage.total_tokens}")
            return {"Sheet1": response.choices[0].message.content.strip()}

    # Text files
    if file_extension == 'txt':
        return {"Sheet1": file_bytes.decode('utf-8', errors='ignore')}

    # CSV files
    if file_extension == 'csv':
        df = pd.read_csv(BytesIO(file_bytes), encoding='utf-8', errors='ignore')
        return {"Sheet1": df.to_string()}

    # Excel files
    if file_extension in ['xlsx', 'xls']:
        all_sheets = pd.read_excel(BytesIO(file_bytes), sheet_name=None)
        sheet_data = {}
        for sheet_name, df_sheet in all_sheets.items():
            sheet_data[sheet_name] = df_sheet.to_string(index=False)
        return sheet_data

    raise ValueError(f"Unsupported file type for {filename}")

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
            category = "Payin Above 50%"
        return payin_value, category
    except (ValueError, TypeError):
        logger.warning(f"Could not parse payin value: {payin_str}")
        return 0.0, "Payin Below 20%"

# def apply_formula_directly(policy_data, company_name):
#     """
#     Apply formula rules directly using Python logic instead of OpenAI
#     """
#     calculated_data = []
    
#     for record in policy_data:
#         # Get policy details
#         segment = record.get('Segment', '').upper()
#         payin_value = record.get('Payin_Value', 0)
#         payin_category = record.get('Payin_Category', '')
        
#         # Determine LOB from segment
#         lob = ""
#         segment_upper = segment.upper()
        
#         if any(tw_keyword in segment_upper for tw_keyword in ['TW', '2W', 'TWO WHEELER']):
#             lob = "TW"
#         elif any(car_keyword in segment_upper for car_keyword in ['PVT CAR', 'PRIVATE CAR', 'CAR']):
#             lob = "PVT CAR"
#         elif any(cv_keyword in segment_upper for cv_keyword in ['CV', 'COMMERCIAL', 'LCV', 'GVW', 'TN']):
#             lob = "CV"
#         elif 'BUS' in segment_upper:
#             lob = "BUS"
#         elif 'TAXI' in segment_upper:
#             lob = "TAXI"
#         elif any(misd_keyword in segment_upper for misd_keyword in ['MISD', 'TRACTOR', 'MISC']):
#             lob = "MISD"
#         else:
#             # Try to infer from remarks or other fields
#             remarks_upper = str(record.get('Remarks', '')).upper()
#             if any(cv_keyword in remarks_upper for cv_keyword in ['TATA', 'MARUTI', 'GVW', 'TN']):
#                 lob = "CV"
#             else:
#                 lob = "UNKNOWN"
        
#         # Find matching formula rule
#         matched_rule = None
#         rule_explanation = ""
        
#         # Normalize company name for matching
#         company_normalized = company_name.upper().replace('GENERAL', '').replace('INSURANCE', '').strip()
        
#         for rule in FORMULA_DATA:
#             if rule["LOB"] != lob:
#                 continue
                
#             # Check if this rule applies to this segment
#             rule_segment = rule["SEGMENT"].upper()
            
#             # Segment matching logic
#             segment_match = False
#             if lob == "CV":
#                 # More specific matching for CV segments
#                 if "UPTO 2.5" in rule_segment:
#                     # Check for explicit "upto 2.5", "2.5", "2.5 Tn", "2.5 GVW" mentions
#                     if any(keyword in segment.upper() for keyword in ["UPTO 2.5", "2.5 TN", "2.5 GVW", "2.5TN", "2.5GVW", "UPTO2.5"]):
#                         segment_match = True
#                 elif "ALL GVW" in rule_segment:
#                     # Only match if it's NOT a 2.5 GVW case
#                     if not any(keyword in segment.upper() for keyword in ["UPTO 2.5", "2.5 TN", "2.5 GVW", "2.5TN", "2.5GVW", "UPTO2.5", "2.5"]):
#                         segment_match = True
#             elif lob == "BUS":
#                 if "SCHOOL" in rule_segment and "SCHOOL" in segment:
#                     segment_match = True
#                 elif "STAFF" in rule_segment and "STAFF" in segment:
#                     segment_match = True
#             elif lob == "PVT CAR":
#                 if "COMP" in rule_segment and any(keyword in segment for keyword in ["COMP", "COMPREHENSIVE"]):
#                     segment_match = True
#                 elif "TP" in rule_segment and "TP" in segment and "COMP" not in segment:
#                     segment_match = True
#             elif lob == "TW":
#                 if "1+5" in rule_segment and "1+5" in segment:
#                     segment_match = True
#                 elif "SAOD + COMP" in rule_segment and any(keyword in segment for keyword in ["SAOD", "COMP"]):
#                     segment_match = True
#                 elif "TP" in rule_segment and "TP" in segment:
#                     segment_match = True
#             else:
#                 # For TAXI and MISD, simple matching
#                 segment_match = True
            
#             if not segment_match:
#                 continue
            
#             # Check company matching
#             insurers = [ins.strip().upper() for ins in rule["INSURER"].split(',')]
#             company_match = False
            
#             if "ALL COMPANIES" in insurers:
#                 company_match = True
#             elif "REST OF COMPANIES" in insurers:
#                 # Check if company is NOT in any specific company list for this LOB/SEGMENT
#                 is_in_specific_list = False
#                 for other_rule in FORMULA_DATA:
#                     if (other_rule["LOB"] == rule["LOB"] and 
#                         other_rule["SEGMENT"] == rule["SEGMENT"] and
#                         "REST OF COMPANIES" not in other_rule["INSURER"] and
#                         "ALL COMPANIES" not in other_rule["INSURER"]):
#                         other_insurers = [ins.strip().upper() for ins in other_rule["INSURER"].split(',')]
#                         if any(company_key in company_normalized for company_key in other_insurers):
#                             is_in_specific_list = True
#                             break
#                 if not is_in_specific_list:
#                     company_match = True
#             else:
#                 # Check specific company names
#                 for insurer in insurers:
#                     if insurer in company_normalized or company_normalized in insurer:
#                         company_match = True
#                         break
            
#             if not company_match:
#                 continue
            
#             # Check if remarks require payin category matching
#             remarks = rule.get("REMARKS", "")
            
#             if remarks == "NIL" or "NIL" in remarks.upper():
#                 # No payin category check needed - apply directly
#                 matched_rule = rule
#                 rule_explanation = f"Direct match: LOB={lob}, Segment={rule_segment}, Company={rule['INSURER']}, No payin category check (NIL remarks)"
#                 break
#             elif any(payin_keyword in remarks for payin_keyword in ["Payin Below", "Payin 21%", "Payin 31%", "Payin Above"]):
#                 # Need to match payin category
#                 if payin_category in remarks:
#                     matched_rule = rule
#                     rule_explanation = f"Payin category match: LOB={lob}, Segment={rule_segment}, Company={rule['INSURER']}, Payin={payin_category}"
#                     break
#             else:
#                 # Other remarks - apply directly
#                 matched_rule = rule
#                 rule_explanation = f"Other remarks match: LOB={lob}, Segment={rule_segment}, Company={rule['INSURER']}, Remarks={remarks}"
#                 break
        
#         # Calculate payout based on matched rule
#         if matched_rule:
#             po_formula = matched_rule["PO"]
#             calculated_payout = 0
            
#             if "90% of Payin" in po_formula:
#                 calculated_payout = payin_value * 0.9
#             elif "88% of Payin" in po_formula:
#                 calculated_payout = payin_value * 0.88
#             elif "Less 2% of Payin" in po_formula:
#                 calculated_payout = payin_value - 2
#             elif "-2%" in po_formula:
#                 calculated_payout = payin_value - 2
#             elif "-3%" in po_formula:
#                 calculated_payout = payin_value - 3
#             elif "-4%" in po_formula:
#                 calculated_payout = payin_value - 4
#             elif "-5%" in po_formula:
#                 calculated_payout = payin_value - 5
#             else:
#                 calculated_payout = payin_value  # Default to original payin
            
#             # Ensure non-negative result
#             calculated_payout = max(0, calculated_payout)
            
#             formula_used = po_formula
#         else:
#             # No rule matched - use default
#             calculated_payout = payin_value
#             formula_used = "No matching rule found"
#             rule_explanation = f"No formula rule matched for LOB={lob}, Segment={segment}, Company={company_name}"
        
#         # Create result record
#         result_record = record.copy()
#         result_record['Calculated Payout'] = f"{int(calculated_payout)}%"
#         result_record['Formula Used'] = formula_used
#         result_record['Rule Explanation'] = rule_explanation
        
#         calculated_data.append(result_record)
    
#     return calculated_data

def apply_formula_directly(policy_data, company_name):
    """
    Apply formula rules directly using Python logic instead of OpenAI
    """
    calculated_data = []
    
    for record in policy_data:
        # Get policy details
        segment = record.get('Segment', '').upper()
        payin_value = record.get('Payin_Value', 0)
        payin_category = record.get('Payin_Category', '')
        
        # Check if Payin_Value is 0; set Calculated Payout to 0% and skip formula application
        if payin_value == 0:
            result_record = record.copy()
            result_record['Calculated Payout'] = "0%"
            result_record['Formula Used'] = "Payin is 0"
            result_record['Rule Explanation'] = f"Payin is 0 for Segment={segment}, Company={company_name}; set Calculated Payout to 0%"
            calculated_data.append(result_record)
            continue
        
        # Determine LOB from segment
        lob = ""
        segment_upper = segment.upper()
        
        if any(tw_keyword in segment_upper for tw_keyword in ['TW', '2W', 'TWO WHEELER']):
            lob = "TW"
        elif any(car_keyword in segment_upper for car_keyword in ['PVT CAR', 'PRIVATE CAR', 'CAR']):
            lob = "PVT CAR"
        elif any(cv_keyword in segment_upper for cv_keyword in ['CV', 'COMMERCIAL', 'LCV', 'GVW', 'TN']):
            lob = "CV"
        elif 'BUS' in segment_upper:
            lob = "BUS"
        elif 'TAXI' in segment_upper:
            lob = "TAXI"
        elif any(misd_keyword in segment_upper for misd_keyword in ['MISD', 'TRACTOR', 'MISC']):
            lob = "MISD"
        else:
            # Try to infer from remarks or other fields
            remarks_upper = str(record.get('Remarks', '')).upper()
            if any(cv_keyword in remarks_upper for cv_keyword in ['TATA', 'MARUTI', 'GVW', 'TN']):
                lob = "CV"
            else:
                lob = "UNKNOWN"
        
        # Find matching formula rule
        matched_rule = None
        rule_explanation = ""
        
        # Normalize company name for matching
        company_normalized = company_name.upper().replace('GENERAL', '').replace('INSURANCE', '').strip()
        
        for rule in FORMULA_DATA:
            if rule["LOB"] != lob:
                continue
                
            # Check if this rule applies to this segment
            rule_segment = rule["SEGMENT"].upper()
            
            # Segment matching logic
            segment_match = False
            if lob == "CV":
                # More specific matching for CV segments
                if "UPTO 2.5" in rule_segment:
                    # Check for explicit "upto 2.5", "2.5", "2.5 Tn", "2.5 GVW" mentions
                    if any(keyword in segment.upper() for keyword in ["UPTO 2.5", "2.5 TN", "2.5 GVW", "2.5TN", "2.5GVW", "UPTO2.5"]):
                        segment_match = True
                elif "ALL GVW" in rule_segment:
                    # Only match if it's NOT a 2.5 GVW case
                    if not any(keyword in segment.upper() for keyword in ["UPTO 2.5", "2.5 TN", "2.5 GVW", "2.5TN", "2.5GVW", "UPTO2.5", "2.5"]):
                        segment_match = True
            elif lob == "BUS":
                if "SCHOOL" in rule_segment and "SCHOOL" in segment:
                    segment_match = True
                elif "STAFF" in rule_segment and "STAFF" in segment:
                    segment_match = True
            elif lob == "PVT CAR":
                if "COMP" in rule_segment and any(keyword in segment for keyword in ["COMP", "COMPREHENSIVE"]):
                    segment_match = True
                elif "TP" in rule_segment and "TP" in segment and "COMP" not in segment:
                    segment_match = True
            elif lob == "TW":
                if "1+5" in rule_segment and "1+5" in segment:
                    segment_match = True
                elif "SAOD + COMP" in rule_segment and any(keyword in segment for keyword in ["SAOD", "COMP"]):
                    segment_match = True
                elif "TP" in rule_segment and "TP" in segment:
                    segment_match = True
            else:
                # For TAXI and MISD, simple matching
                segment_match = True
            
            if not segment_match:
                continue
            
            # Check company matching
            insurers = [ins.strip().upper() for ins in rule["INSURER"].split(',')]
            company_match = False
            
            if "ALL COMPANIES" in insurers:
                company_match = True
            elif "REST OF COMPANIES" in insurers:
                # Check if company is NOT in any specific company list for this LOB/SEGMENT
                is_in_specific_list = False
                for other_rule in FORMULA_DATA:
                    if (other_rule["LOB"] == rule["LOB"] and 
                        other_rule["SEGMENT"] == rule["SEGMENT"] and
                        "REST OF COMPANIES" not in other_rule["INSURER"] and
                        "ALL COMPANIES" not in other_rule["INSURER"]):
                        other_insurers = [ins.strip().upper() for ins in other_rule["INSURER"].split(',')]
                        if any(company_key in company_normalized for company_key in other_insurers):
                            is_in_specific_list = True
                            break
                if not is_in_specific_list:
                    company_match = True
            else:
                # Check specific company names
                for insurer in insurers:
                    if insurer in company_normalized or company_normalized in insurer:
                        company_match = True
                        break
            
            if not company_match:
                continue
            
            # Check if remarks require payin category matching
            remarks = rule.get("REMARKS", "")
            
            if remarks == "NIL" or "NIL" in remarks.upper():
                # No payin category check needed - apply directly
                matched_rule = rule
                rule_explanation = f"Direct match: LOB={lob}, Segment={rule_segment}, Company={rule['INSURER']}, No payin category check (NIL remarks)"
                break
            elif any(payin_keyword in remarks for payin_keyword in ["Payin Below", "Payin 21%", "Payin 31%", "Payin Above"]):
                # Need to match payin category
                if payin_category in remarks:
                    matched_rule = rule
                    rule_explanation = f"Payin category match: LOB={lob}, Segment={rule_segment}, Company={rule['INSURER']}, Payin={payin_category}"
                    break
            else:
                # Other remarks - apply directly
                matched_rule = rule
                rule_explanation = f"Other remarks match: LOB={lob}, Segment={rule_segment}, Company={rule['INSURER']}, Remarks={remarks}"
                break
        
        # Calculate payout based on matched rule
        if matched_rule:
            po_formula = matched_rule["PO"]
            calculated_payout = 0
            
            if "90% of Payin" in po_formula:
                calculated_payout = payin_value * 0.9
            elif "88% of Payin" in po_formula:
                calculated_payout = payin_value * 0.88
            elif "Less 2% of Payin" in po_formula:
                calculated_payout = payin_value - 2
            elif "-2%" in po_formula:
                calculated_payout = payin_value - 2
            elif "-3%" in po_formula:
                calculated_payout = payin_value - 3
            elif "-4%" in po_formula:
                calculated_payout = payin_value - 4
            elif "-5%" in po_formula:
                calculated_payout = payin_value - 5
            else:
                calculated_payout = payin_value  # Default to original payin
            
            # Ensure non-negative result
            calculated_payout = max(0, calculated_payout)
            
            formula_used = po_formula
        else:
            # No rule matched - use default
            calculated_payout = payin_value
            formula_used = "No matching rule found"
            rule_explanation = f"No formula rule matched for LOB={lob}, Segment={segment}, Company={company_name}"
        
        # Create result record
        result_record = record.copy()
        result_record['Calculated Payout'] = f"{int(calculated_payout)}%"
        result_record['Formula Used'] = formula_used
        result_record['Rule Explanation'] = rule_explanation
        
        calculated_data.append(result_record)
    
    return calculated_data
# def process_files(policy_file_bytes, policy_filename, policy_content_type, company_name):
#     """Main processing function, handling sheets separately and creating a single Excel file with multiple sheets"""
#     try:
#         st.info("🔍 Extracting text from policy file...")
        
#         # Extract text, which returns a dictionary for Excel files
#         extracted_data = extract_text_from_file(policy_file_bytes, policy_filename, policy_content_type)

#         logger.info(f"Extracted data: {list(extracted_data.keys())} sheets")

#         # Initialize results
#         results = {
#             "extracted_text": {},
#             "parsed_data": {},
#             "calculated_data": {},
#             "excel_data": None,  # Single Excel file bytes
#             "df_calc": {}
#         }

#         # Process each sheet separately
#         output = BytesIO()
#         with pd.ExcelWriter(output, engine='openpyxl') as writer:
#             for sheet_name, extracted_text in extracted_data.items():
#                 st.success(f"✅ Text extracted successfully for {sheet_name}! Length: {len(extracted_text)} chars")

#                 # Parse policy data for this sheet
#                 st.info(f"🧠 Parsing policy data for {sheet_name}...")
                
#                 parse_prompt = f"""
#                 Analyze the following text, which contains insurance policy details for sheet {sheet_name} from an Excel grid.
#                 Use your intelligence to identify and extract the data accurately.

#                 Company Name: {company_name}

#                 IMPORTANT: Always return a valid JSON array (list) of records, even if there's only one record or the sheet is empty (use 'N/A' defaults).

#                 Extract into JSON records with these exact fields:
#                 - "Segment": identify LOB like TW, PVT CAR, CV, BUS, TAXI, MISD and policy type like TP, COMP, SAOD. 
#                   For CV/Commercial Vehicle/Tractor, also extract weight/tonnage/GVW information like "0 T-3.5T", "3.5T-7.5T", "New", "Roll over", "SATP". Use sheet name as base (e.g., "CV 0 T-3.5T").
#                   PRESERVE EXACT TONNAGE/WEIGHT INFORMATION as mentioned in the text.
#                 - "Location": location information, e.g., region (East/West/North/South) and states (CG, AS, JH, etc.). Combine as "Region: State" for example: region is East and state is CG , then EAST: CG, if grid-based.
#                 - "Policy Type": policy type information, e.g., "New", "Roll over", "SATP" from rows, or "COMP/TP" if not specified.
#                 - "Payin": convert payout/payrate values to percentage format, e.g. 0.625 → 62.5%, 34 → 34%. Use "NIL" or 0% for empty/NIL.
#                 - "Doable District": district/RTO information from notes, e.g., RTO codes like "UP21", "DL-NCR".
#                 - "Remarks": any additional information INCLUDING vehicle make information, notes, conditions, declined RTOs, special terms.

#                 ENHANCED SEGMENT EXTRACTION FOR CV/TRACTOR:
#                 - Grid rows are GVW ranges or types (e.g., "0 T-3.5T", "New"); create one record per cell/value, mapping to state/region.
#                 - If text mentions "upto 2.5 Tn", "2.5 GVW", extract as "CV upto 2.5 Tn"; for Tractor, use "Tractor New", etc.
#                 - PRESERVE exact GVW/State from headers.

#                 ENHANCED REMARKS FIELD INSTRUCTIONS:
#                 - Extract ALL additional notes/conditions (e.g., "Special Condition (MH)...", "Declined RTOs...").
#                 - Include vehicle make information if present (e.g., "Make – Tata, Maruti" → "Vehicle Makes: Tata, Maruti").
#                 - Look for AGE (e.g., ">5 years"), TRANSACTION (e.g., "New", "Roll over"), VALIDITY (e.g., "Jul-2025").
#                 - RTO groupings (e.g., "UP2-Moradabad-UP21...") as "RTOs: UP21, UP81...".
#                 - Combine into remarks separated by semicolons, e.g., "Terms: If falsification...; Declined RTOs: HR27...; Vehicle Makes: None".
#                 - If sheet empty, use "No data in sheet".

#                 Be intelligent in identifying grids: Rows as categories (GVW/Type), columns as states/regions, values as Payin.
#                 Create multiple records for grid cells (e.g., one per state/payin pair).
#                 Look for patterns like:
#                 - Payin grids by GVW/State.
#                 - Regional groupings (East: CG,AS,...).
#                 - Notes with RTO lists, terms.

#                 If no information, use "N/A". If Policy not defined, use "COMP/TP".
#                 Note: Payrate/PO means Payin; normalize to %.
#                 Note: Mention original sheet segment in remarks if relevant.
#                 note: if payin is declined or No Biz , please don't consider it 
                
#                 Text to analyze:
#                 {extracted_text}
                
#                 Return ONLY a valid JSON array, no other text.
#                 """
                
#                 response = client.chat.completions.create(
#                     model="gpt-4o-mini",
#                     messages=[
#                         {"role": "system", "content": "Extract policy data as JSON array. Always rename 'Payout' → 'Payin'. Normalize all Payin values to percentage format. Use intelligence to identify segments accurately. Extract ALL additional information for remarks including vehicle makes, age information, and transaction type ONLY when explicitly found in text. Do NOT infer or assume transaction type if not clearly stated. Return ONLY valid JSON array."},
#                         {"role": "user", "content": parse_prompt}
#                     ],
#                     temperature=0.1
#                 )
                
#                 parsed_json = response.choices[0].message.content.strip()
#                 logger.info(f"Raw OpenAI response for {sheet_name}: {parsed_json[:500]}...")
                
#                 # Clean the JSON response
#                 cleaned_json = clean_json_response(parsed_json)
#                 logger.info(f"Cleaned JSON for {sheet_name}: {cleaned_json[:500]}...")
                
#                 try:
#                     policy_data = json.loads(cleaned_json)
#                     policy_data = ensure_list_format(policy_data)
#                 except json.JSONDecodeError as e:
#                     logger.error(f"JSON decode error for {sheet_name}: {str(e)}")
#                     logger.error(f"Problematic JSON: {cleaned_json}")
#                     policy_data = [{
#                         "Segment": "Unknown",
#                         "Location": "N/A",
#                         "Policy Type": "N/A", 
#                         "Payin": "0%",
#                         "Doable District": "N/A",
#                         "Remarks": f"Error parsing data for {sheet_name}: {str(e)}"
#                     }]

#                 st.success(f"✅ Successfully parsed {len(policy_data)} policy records for {sheet_name}")

#                 # Pre-classify Payin values
#                 for record in policy_data:
#                     try:
#                         payin_val, payin_cat = classify_payin(record.get('Payin', '0%'))
#                         record['Payin_Value'] = payin_val
#                         record['Payin_Category'] = payin_cat
#                     except Exception as e:
#                         logger.warning(f"Error classifying payin for record in {sheet_name}: {record}, error: {str(e)}")
#                         record['Payin_Value'] = 0.0
#                         record['Payin_Category'] = "Payin Below 20%"

#                 # Apply formulas directly
#                 st.info(f"🧮 Applying formulas and calculating payouts for {sheet_name}...")
#                 calculated_data = apply_formula_directly(policy_data, company_name)

#                 st.success(f"✅ Successfully calculated {len(calculated_data)} records for {sheet_name}")

#                 # Create DataFrame for this sheet
#                 df_calc = pd.DataFrame(calculated_data)
                
#                 # Write to Excel sheet
#                 st.info(f"📊 Writing data to Excel sheet: {sheet_name}...")
#                 df_calc.to_excel(writer, sheet_name=sheet_name, startrow=2, index=False)
#                 worksheet = writer.sheets[sheet_name]
#                 headers = list(df_calc.columns)
#                 for col_num, value in enumerate(headers, 1):
#                     worksheet.cell(row=3, column=col_num, value=value)
#                     worksheet.cell(row=3, column=col_num).font = worksheet.cell(row=3, column=col_num).font.copy(bold=True)
#                 company_cell = worksheet.cell(row=1, column=1, value=company_name)
#                 worksheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
#                 company_cell.font = company_cell.font.copy(bold=True, size=14)
#                 company_cell.alignment = company_cell.alignment.copy(horizontal='center')
#                 title_cell = worksheet.cell(row=2, column=1, value=f'Policy Data with Payin and Calculated Payouts - {sheet_name}')
#                 worksheet.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(headers))
#                 title_cell.font = title_cell.font.copy(bold=True, size=12)
#                 title_cell.alignment = title_cell.alignment.copy(horizontal='center')

#                 # Store results for this sheet
#                 results["extracted_text"][sheet_name] = extracted_text
#                 results["parsed_data"][sheet_name] = policy_data
#                 results["calculated_data"][sheet_name] = calculated_data
#                 results["df_calc"][sheet_name] = df_calc

#         # Finalize Excel file
#         output.seek(0)
#         results["excel_data"] = output.read()

#         return results

#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}", exc_info=True)
#         raise Exception(f"An error occurred: {str(e)}")
def process_files(policy_file_bytes, policy_filename, policy_content_type, company_name):
    """Main processing function, handling sheets separately and creating a single Excel file with multiple sheets"""
    try:
        st.info("🔍 Extracting text from policy file...")
        
        # Extract text, which returns a dictionary for Excel files
        extracted_data = extract_text_from_file(policy_file_bytes, policy_filename, policy_content_type)

        logger.info(f"Extracted data: {list(extracted_data.keys())} sheets")

        # Initialize results
        results = {
            "extracted_text": {},
            "parsed_data": {},
            "calculated_data": {},
            "excel_data": None,  # Single Excel file bytes
            "df_calc": {}
        }

        # Process each sheet separately
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for sheet_name, extracted_text in extracted_data.items():
                st.success(f"✅ Text extracted successfully for {sheet_name}! Length: {len(extracted_text)} chars")

                # Parse policy data for this sheet
                st.info(f"🧠 Parsing policy data for {sheet_name}...")
                
                parse_prompt = f"""
                Analyze the following text, which contains insurance policy details for sheet {sheet_name} from an Excel grid.
                Use your intelligence to identify and extract the data accurately.

                Company Name: {company_name}

                IMPORTANT: Always return a valid JSON array (list) of records, even if there's only one record or the sheet is empty (use 'N/A' defaults).

                Extract into JSON records with these exact fields:
                - "Segment": identify LOB like TW, PVT CAR, CV, BUS, TAXI, MISD and policy type like TP, COMP, SAOD. 
                  For CV/Commercial Vehicle/Tractor, also extract weight/tonnage/GVW information like "0 T-3.5T", "3.5T-7.5T", "New", "Roll over", "SATP". Use sheet name as base (e.g., "CV 0 T-3.5T").
                  PRESERVE EXACT TONNAGE/WEIGHT INFORMATION as mentioned in the text.
                - "Location": location information, e.g., region (East/West/North/South) and states (CG, AS, JH, etc.). Combine as "Region: State" if grid-based.
                - "Policy Type": policy type information, e.g., "New", "Roll over", "SATP" from rows, or "COMP/TP" if not specified.
                - "Payin": convert payout/payrate values to percentage format, e.g. 0.625 → 62.5%, 34 → 34%. Use "NIL" or 0% for empty/NIL.
                - "Doable District": district/RTO information from notes, e.g., RTO codes like "UP21", "DL-NCR".
                - "Remarks": any additional information INCLUDING vehicle make information, notes, conditions, declined RTOs, special terms.

                ENHANCED SEGMENT EXTRACTION FOR CV/TRACTOR:
                - Grid rows are GVW ranges or types (e.g., "0 T-3.5T", "New"); create one record per cell/value, mapping to state/region.
                - If text mentions "upto 2.5 Tn", "2.5 GVW", extract as "CV upto 2.5 Tn"; for Tractor, use "Tractor New", etc.
                - PRESERVE exact GVW/State from headers.

                ENHANCED REMARKS FIELD INSTRUCTIONS:
                - Extract ALL additional notes/conditions (e.g., "Special Condition (MH)...", "Declined RTOs...").
                - Include vehicle make information if present (e.g., "Make – Tata, Maruti" → "Vehicle Makes: Tata, Maruti").
                - Look for AGE (e.g., ">5 years"), TRANSACTION (e.g., "New", "Roll over"), VALIDITY (e.g., "Jul-2025").
                - RTO groupings (e.g., "UP2-Moradabad-UP21...") as "RTOs: UP21, UP81...".
                - Combine into remarks separated by semicolons, e.g., "Terms: If falsification...; Declined RTOs: HR27...; Vehicle Makes: None".
                - If sheet empty, use "No data in sheet".

                Be intelligent in identifying grids: Rows as categories (GVW/Type), columns as states/regions, values as Payin.
                Create multiple records for grid cells (e.g., one per state/payin pair).
                Look for patterns like:
                - Payin grids by GVW/State.
                - Regional groupings (East: CG,AS,...).
                - Notes with RTO lists, terms.

                If no information, use "N/A". If Policy not defined, use "COMP/TP".
                Note: Payrate/PO means Payin; normalize to %.
                Note: Mention original sheet segment in remarks if relevant.
                
                Text to analyze:
                {extracted_text}
                
                Return ONLY a valid JSON array, no other text.
                """
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Extract policy data as JSON array. Always rename 'Payout' → 'Payin'. Normalize all Payin values to percentage format. Use intelligence to identify segments accurately. Extract ALL additional information for remarks including vehicle makes, age information, and transaction type ONLY when explicitly found in text. Do NOT infer or assume transaction type if not clearly stated. Return ONLY valid JSON array."},
                        {"role": "user", "content": parse_prompt}
                    ],
                    temperature=0.1
                )
                token_usage = response.usage
                logger.info(f"Tokens used for image extraction: Prompt: {token_usage.prompt_tokens}, Completion: {token_usage.completion_tokens}, Total: {token_usage.total_tokens}")
                
                parsed_json = response.choices[0].message.content.strip()
                logger.info(f"Raw OpenAI response for {sheet_name}: {parsed_json[:500]}...")
                
                # Clean the JSON response
                cleaned_json = clean_json_response(parsed_json)
                logger.info(f"Cleaned JSON for {sheet_name}: {cleaned_json[:500]}...")
                
                try:
                    policy_data = json.loads(cleaned_json)
                    policy_data = ensure_list_format(policy_data)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error for {sheet_name}: {str(e)}")
                    logger.error(f"Problematic JSON: {cleaned_json}")
                    policy_data = [{
                        "Segment": "Unknown",
                        "Location": "N/A",
                        "Policy Type": "N/A", 
                        "Payin": "0%",
                        "Doable District": "N/A",
                        "Remarks": f"Error parsing data for {sheet_name}: {str(e)}"
                    }]

                st.success(f"✅ Successfully parsed {len(policy_data)} policy records for {sheet_name}")

                # Pre-classify Payin values
                for record in policy_data:
                    try:
                        payin_val, payin_cat = classify_payin(record.get('Payin', '0%'))
                        record['Payin_Value'] = payin_val
                        record['Payin_Category'] = payin_cat
                    except Exception as e:
                        logger.warning(f"Error classifying payin for record in {sheet_name}: {record}, error: {str(e)}")
                        record['Payin_Value'] = 0.0
                        record['Payin_Category'] = "Payin Below 20%"

                # Apply formulas directly
                st.info(f"🧮 Applying formulas and calculating payouts for {sheet_name}...")
                calculated_data = apply_formula_directly(policy_data, company_name)

                st.success(f"✅ Successfully calculated {len(calculated_data)} records for {sheet_name}")

                # Create DataFrame for this sheet
                df_calc = pd.DataFrame(calculated_data)
                
                # Write to Excel sheet
                st.info(f"📊 Writing data to Excel sheet: {sheet_name}...")
                df_calc.to_excel(writer, sheet_name=sheet_name, startrow=2, index=False)
                worksheet = writer.sheets[sheet_name]
                headers = list(df_calc.columns)
                for col_num, value in enumerate(headers, 1):
                    worksheet.cell(row=3, column=col_num, value=value)
                    worksheet.cell(row=3, column=col_num).font = worksheet.cell(row=3, column=col_num).font.copy(bold=True)
                company_cell = worksheet.cell(row=1, column=1, value=company_name)
                worksheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
                company_cell.font = company_cell.font.copy(bold=True, size=14)
                company_cell.alignment = company_cell.alignment.copy(horizontal='center')
                title_cell = worksheet.cell(row=2, column=1, value=f'Policy Data with Payin and Calculated Payouts - {sheet_name}')
                worksheet.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(headers))
                title_cell.font = title_cell.font.copy(bold=True, size=12)
                title_cell.alignment = title_cell.alignment.copy(horizontal='center')

                # Store results for this sheet
                results["extracted_text"][sheet_name] = extracted_text
                results["parsed_data"][sheet_name] = policy_data
                results["calculated_data"][sheet_name] = calculated_data
                results["df_calc"][sheet_name] = df_calc

        # Finalize Excel file
        output.seek(0)
        results["excel_data"] = output.read()

        return results

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise Exception(f"An error occurred: {str(e)}")


# Streamlit App

def main():
    st.set_page_config(
        page_title="Insurance Policy Processing", 
        page_icon="📋", 
        layout="wide"
    )
    
    st.title("🏢 Insurance Policy Processing System")
    st.markdown("---")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📁 File Upload")
        
        # Company name input
        company_name = st.text_input(
            "Company Name", 
            value="Unknown Company",
            help="Enter the insurance company name"
        )
        
        # Policy file upload
        policy_file = st.file_uploader(
            "📄 Upload Policy File",
            type=['pdf', 'png', 'jpg', 'jpeg', 'txt', 'csv', 'xlsx', 'xls'],
            help="Upload your insurance policy document"
        )
        
        # Show formula data info
        st.info("📊 Formula rules are embedded in the system and will be automatically applied.")
        
        # Process button
        process_button = st.button(
            "🚀 Process Policy File", 
            type="primary",
            disabled=not policy_file
        )
    
    # Main content area
    if not policy_file:
        st.info("👆 Please upload a policy file to begin processing.")
        st.markdown("""
        ### 📋 Instructions:
        1. **Company Name**: Enter the insurance company name
        2. **Policy File**: Upload the document containing policy data (PDF, Image, Excel, CSV, etc.)
        3. **Process**: Click the process button to extract data and calculate payouts
        
        ### 🎯 Features:
        - **Multi-format Support**: PDF, Images, Excel, CSV, Text files
        - **AI-Powered Extraction**: Uses GPT-4 for intelligent text extraction
        - **Enhanced Remarks Extraction**: Automatically detects and extracts:
          - Vehicle make information (Tata, Maruti, etc.)
          - Age information (>5 years, etc.)
          - Transaction type (New/Old/Renewal)
        - **Smart Formula Application**: Uses embedded formula rules for accurate calculations
        - **Excel Export**: Download processed data as a single Excel file with multiple sheets
        
        ### 📊 Formula Rules:
        The system uses pre-configured formula rules for different LOBs:
        - **TW**: Two Wheeler segments with company-specific rules
        - **PVT CAR**: Private Car segments (COMP+SAOD, TP)
        - **CV**: Commercial Vehicle segments
        - **BUS**: School Bus and Staff Bus segments
        - **TAXI**: Taxi segments with payin-based rules
        - **MISD**: Miscellaneous segments including tractors
        """)
        return
    
    if process_button:
        try:
            # Read file contents
            policy_file_bytes = policy_file.read()
            
            # Process files
            with st.spinner("Processing policy file... This may take a few moments."):
                results = process_files(
                    policy_file_bytes, policy_file.name, policy_file.type,
                    company_name
                )
            
            st.success("🎉 Processing completed successfully!")
            
            # Display results in tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Final Results", 
                "📝 Extracted Text", 
                "🧾 Parsed Data", 
                "🧮 Calculated Data",
                "📥 Download"
            ])
            
            with tab1:
                st.subheader("📊 Final Processed Data")
                for sheet_name, df_calc in results["df_calc"].items():
                    st.write(f"### {sheet_name}")
                    st.dataframe(df_calc, use_container_width=True)
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(f"Total Records ({sheet_name})", len(results["calculated_data"][sheet_name]))
                    with col2:
                        avg_payin = sum([r.get('Payin_Value', 0) for r in results["calculated_data"][sheet_name]]) / len(results["calculated_data"][sheet_name]) if results["calculated_data"][sheet_name] else 0
                        st.metric(f"Avg Payin ({sheet_name})", f"{avg_payin:.1f}%")
                    with col3:
                        segments = set([r.get('Segment', 'N/A') for r in results["calculated_data"][sheet_name]])
                        st.metric(f"Unique Segments ({sheet_name})", len(segments))
                    with col4:
                        st.metric("Company", company_name)
                    st.write(f"**Formula Rules Applied for {sheet_name}**")
                    formula_summary = {}
                    for record in results["calculated_data"][sheet_name]:
                        formula = record.get('Formula Used', 'Unknown')
                        if formula not in formula_summary:
                            formula_summary[formula] = 0
                        formula_summary[formula] += 1
                    for formula, count in formula_summary.items():
                        st.write(f"• **{formula}**: Applied to {count} record(s)")
            
            with tab2:
                st.subheader("📝 Extracted Text from Policy File")
                for sheet_name, text in results["extracted_text"].items():
                    st.write(f"### {sheet_name}")
                    st.text_area(
                        f"Policy Text - {sheet_name}", 
                        text, 
                        height=400,
                        key=f"policy_text_{sheet_name}"
                    )
                st.subheader("📊 Embedded Formula Rules")
                st.write("The following formula rules are embedded in the system:")
                df_formula = pd.DataFrame(FORMULA_DATA)
                st.dataframe(df_formula, use_container_width=True)
            
            with tab3:
                st.subheader("🧾 Parsed Policy Data")
                for sheet_name, data in results["parsed_data"].items():
                    st.write(f"### {sheet_name}")
                    st.json(data)
            
            with tab4:
                st.subheader("🧮 Calculated Data with Formulas")
                for sheet_name, data in results["calculated_data"].items():
                    st.write(f"### {sheet_name}")
                    st.json(data)
                    st.write(f"**Rule Explanations for {sheet_name}**")
                    for i, record in enumerate(data):
                        with st.expander(f"Record {i+1}: {record.get('Segment', 'Unknown')}"):
                            st.write(f"**Payin**: {record.get('Payin', 'N/A')}")
                            st.write(f"**Calculated Payout**: {record.get('Calculated Payout', 'N/A')}")
                            st.write(f"**Formula Used**: {record.get('Formula Used', 'N/A')}")
                            st.write(f"**Rule Explanation**: {record.get('Rule Explanation', 'N/A')}")
            
            with tab5:
                st.subheader("📥 Download Results")
                # Single Excel file download
                st.download_button(
                    label="📊 Download Consolidated Excel File",
                    data=results["excel_data"],
                    file_name=f"{company_name}_processed_policies.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                # Sheet-specific JSON and CSV downloads
                for sheet_name, data in results["calculated_data"].items():
                    st.write(f"### {sheet_name}")
                    json_data = json.dumps(data, indent=2)
                    st.download_button(
                        label=f"📄 Download JSON Data ({sheet_name})",
                        data=json_data,
                        file_name=f"{company_name}_processed_data_{sheet_name}.json",
                        mime="application/json"
                    )
                    csv_data = results["df_calc"][sheet_name].to_csv(index=False)
                    st.download_button(
                        label=f"📋 Download CSV File ({sheet_name})",
                        data=csv_data,
                        file_name=f"{company_name}_processed_policies_{sheet_name}.csv",
                        mime="text/csv"
                    )
                st.info("💡 The Excel file contains all processed data, with one sheet per input sheet, formatted with company header and calculated payouts.")

        except Exception as e:
            st.error(f"❌ Error processing files: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()
