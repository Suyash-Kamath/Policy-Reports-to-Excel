import streamlit as st
from io import BytesIO
import base64
import json
import os
from dotenv import load_dotenv
import logging
import re
import pandas as pd

# Check if required packages are available
try:
    from openai import OpenAI
except ImportError:
    st.error("OpenAI package not found. Please install it using 'pip install openai'")
    st.stop()

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

# Updated Formula Data from the provided image/table
FORMULA_DATA = [
    {"LOB": "TW", "SEGMENT": "1+5", "INSURER": "All Companies", "PO": "90% of Payin", "REMARKS": "NIL"},
    {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "INSURER": "All Companies", "PO": "90% of Payin", "REMARKS": "NIL"},
    {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "INSURER": "DIGIT", "PO": "-2%", "REMARKS": "Payin Below 20%"},
    {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "INSURER": "DIGIT", "PO": "-3%", "REMARKS": "Payin 21% to 30%"},
    {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "INSURER": "DIGIT", "PO": "-4%", "REMARKS": "Payin 31% to 50%"},
    {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "INSURER": "DIGIT", "PO": "-5%", "REMARKS": "Payin Above 50%"},
    {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Bajaj, Digit, ICICI", "PO": "-3%", "REMARKS": "Payin Above 20%"},
    {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Rest of Companies", "PO": "-2%", "REMARKS": "Payin Below 20%"},
    {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Rest of Companies", "PO": "-3%", "REMARKS": "Payin 21% to 30%"},
    {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Rest of Companies", "PO": "-4%", "REMARKS": "Payin 31% to 50%"},
    {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Rest of Companies", "PO": "-5%", "REMARKS": "Payin Above 50%"},
    {"LOB": "PVT CAR", "SEGMENT": "PVT CAR COMP + SAOD", "INSURER": "All Companies", "PO": "90% of Payin", "REMARKS": "All Fuel"},
    {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "Bajaj, Digit, SBI", "PO": "-2%", "REMARKS": "Payin Below 20%"},
    {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "Bajaj, Digit, SBI", "PO": "-3%", "REMARKS": "Payin Above 20%"},
    {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "Rest of Companies", "PO": "90% of Payin", "REMARKS": "Zuno - 21"},
    {"LOB": "CV", "SEGMENT": "Upto 2.5 GVW", "INSURER": "Reliance, SBI", "PO": "-2%", "REMARKS": "NIL"},
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


def extract_text_from_file(file_bytes: bytes, filename: str, content_type: str) -> str:
    """Extract text from uploaded image file using OCR with enhanced prompting"""
    file_extension = filename.split('.')[-1].lower() if '.' in filename else ''
    file_type = content_type if content_type else file_extension

    # Image-based extraction with enhanced OCR
    image_extensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']
    if file_extension in image_extensions or file_type.startswith('image/'):
        try:
            image_base64 = base64.b64encode(file_bytes).decode('utf-8')
            
            prompt = """Extract ALL text from this insurance policy image with extreme accuracy.

CRITICAL INSTRUCTIONS:
1. Read EVERY piece of text visible in the image, including:
   - Headers, titles, and section names
   - All table data (columns and rows)
   - Segment/LOB information (TW, PVT CAR, CV, BUS, TAXI, MISD)
   - Company names
   - Policy types (TP, COMP, SAOD, etc.)
   - Payin/Payout percentages or decimals
   - Weight/tonnage (e.g., "upto 2.5 Tn", "2.5 GVW")
   - Vehicle makes (Tata, Maruti, etc.)
   - Age information (>5 years, etc.)
   - Transaction types (New, Old, Renewal)
   - Location/district information
   - Validity dates
   - ALL numerical values
   - Any remarks, notes, or conditions

2. Preserve the EXACT format and structure of tables if present
3. If there's a table, clearly indicate column headers and separate rows
4. For numbers that look like decimals (0.625, 0.34), preserve them exactly
5. For percentages (34%, 62.5%), preserve them exactly
6. Extract text in a structured, organized manner

Return the complete text extraction - do not summarize or skip anything."""
                
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/{file_extension};base64,{image_base64}"}}
                    ]
                }],
                temperature=0.0,
                max_tokens=4000
            )
            
            extracted_text = response.choices[0].message.content.strip()
            
            if not extracted_text or len(extracted_text) < 10:
                logger.error("OCR returned very short or empty text")
                return ""
            
            return extracted_text
            
        except Exception as e:
            logger.error(f"Error in OCR extraction: {str(e)}")
            raise ValueError(f"Failed to extract text from image: {str(e)}")

    raise ValueError(f"Unsupported file type for {filename}. Only images are supported.")


def clean_json_response(response_text: str) -> str:
    """Clean and extract valid JSON array from OpenAI response"""
    # Remove markdown code blocks
    cleaned = re.sub(r'```json\s*|\s*```', '', response_text).strip()
    
    # Find the start and end of the JSON array
    start_idx = cleaned.find('[')
    end_idx = cleaned.rfind(']') + 1 if cleaned.rfind(']') != -1 else len(cleaned)
    
    if start_idx != -1 and end_idx > start_idx:
        cleaned = cleaned[start_idx:end_idx]
    else:
        logger.warning("No valid JSON array found in response, returning empty array")
        return "[]"
    
    # Ensure the cleaned string is valid JSON by adding brackets if missing
    if not cleaned.startswith('['):
        cleaned = '[' + cleaned
    if not cleaned.endswith(']'):
        cleaned += ']'
    
    return cleaned


def ensure_list_format(data) -> list:
    """Ensure data is in list format"""
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return [data]
    else:
        raise ValueError(f"Expected list or dict, got {type(data)}")


def classify_payin(payin_str):
    """Converts Payin string to float and classifies its range"""
    try:
        payin_clean = str(payin_str).replace('%', '').replace(' ', '').strip()
        
        if not payin_clean or payin_clean.upper() == 'N/A':
            return 0.0, "Payin Below 20%"
        
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
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not parse payin value: {payin_str}, error: {e}")
        return 0.0, "Payin Below 20%"


def apply_formula_directly(policy_data, company_name):
    """Apply formula rules directly using Python logic with default STAFF BUS for unspecified BUS"""
    if not policy_data:
        logger.warning("No policy data to process")
        return []
    
    calculated_data = []
    
    for record in policy_data:
        try:
            segment = str(record.get('Segment', '')).upper()
            payin_value = record.get('Payin_Value', 0)
            payin_category = record.get('Payin_Category', '')
            
            lob = ""
            segment_upper = segment.upper()
            
            # Enhanced LOB detection with more keywords
            if any(tw_keyword in segment_upper for tw_keyword in ['TW', '2W', 'TWO WHEELER', 'TWO-WHEELER']):
                lob = "TW"
            elif any(car_keyword in segment_upper for car_keyword in ['PVT CAR', 'PRIVATE CAR', 'CAR', 'PCI']):
                lob = "PVT CAR"
            elif any(cv_keyword in segment_upper for cv_keyword in ['CV', 'COMMERCIAL', 'LCV', 'GVW', 'TN', 'UPTO', 'ALL GVW', 'PCV', 'GCV']):
                lob = "CV"
            elif 'BUS' in segment_upper:
                lob = "BUS"
            elif 'TAXI' in segment_upper:
                lob = "TAXI"
            elif any(misd_keyword in segment_upper for misd_keyword in ['MISD', 'TRACTOR', 'MISC', 'AMBULANCE', 'POLICE VAN', 'GARBAGE VAN']):
                lob = "MISD"
            else:
                remarks_upper = str(record.get('Remarks', '')).upper()
                if any(cv_keyword in remarks_upper for cv_keyword in ['TATA', 'MARUTI', 'GVW', 'TN']):
                    lob = "CV"
                else:
                    lob = "UNKNOWN"
            
            matched_segment = segment_upper
            if lob == "BUS":
                if "SCHOOL" not in segment_upper and "STAFF" not in segment_upper:
                    matched_segment = "STAFF BUS"
                elif "SCHOOL" in segment_upper:
                    matched_segment = "SCHOOL BUS"
                elif "STAFF" in segment_upper:
                    matched_segment = "STAFF BUS"
            
            matched_rule = None
            rule_explanation = ""
            company_normalized = company_name.upper().replace('GENERAL', '').replace('INSURANCE', '').strip()
            
            for rule in FORMULA_DATA:
                if rule["LOB"] != lob:
                    continue
                    
                rule_segment = rule["SEGMENT"].upper()
                segment_match = False
                
                if lob == "CV":
                    if "UPTO 2.5" in rule_segment:
                        if any(keyword in segment_upper for keyword in ["UPTO 2.5", "2.5 TN", "2.5 GVW", "2.5TN", "2.5GVW", "UPTO2.5"]):
                            segment_match = True
                    elif "ALL GVW" in rule_segment:
                        segment_match = True  # Always match for All GVW; fallback if Upto company doesn't match
                elif lob == "BUS":
                    if matched_segment == rule_segment:
                        segment_match = True
                elif lob == "PVT CAR":
                    if "COMP" in rule_segment and any(keyword in segment for keyword in ["COMP", "COMPREHENSIVE", "PACKAGE", "1ST PARTY", "1+1"]):
                        segment_match = True
                    elif "TP" in rule_segment and "TP" in segment and "COMP" not in segment:
                        segment_match = True
                elif lob == "TW":
                    if "1+5" in rule_segment and any(keyword in segment for keyword in ["1+5", "NEW", "FRESH"]):
                        segment_match = True
                    elif "SAOD + COMP" in rule_segment and any(keyword in segment for keyword in ["SAOD", "COMP", "PACKAGE", "1ST PARTY", "1+1"]):
                        segment_match = True
                    elif "TP" in rule_segment and "TP" in segment:
                        segment_match = True
                else:
                    segment_match = True
                
                if not segment_match:
                    continue
                
                insurers = [ins.strip().upper() for ins in rule["INSURER"].split(',')]
                company_match = False
                
                if "ALL COMPANIES" in insurers:
                    company_match = True
                elif "REST OF COMPANIES" in insurers:
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
                    for insurer in insurers:
                        if insurer in company_normalized or company_normalized in insurer:
                            company_match = True
                            break
                
                if not company_match:
                    continue
                
                remarks = rule.get("REMARKS", "")
                
                if remarks == "NIL" or "NIL" in remarks.upper():
                    matched_rule = rule
                    rule_explanation = f"Direct match: LOB={lob}, Segment={rule_segment}, Company={rule['INSURER']}"
                    break
                elif any(payin_keyword in remarks for payin_keyword in ["Payin Below", "Payin 21%", "Payin 31%", "Payin Above"]):
                    if payin_category in remarks:
                        matched_rule = rule
                        rule_explanation = f"Payin category match: LOB={lob}, Segment={rule_segment}, Payin={payin_category}"
                        break
                else:
                    matched_rule = rule
                    rule_explanation = f"Other remarks match: LOB={lob}, Segment={rule_segment}, Remarks={remarks}"
                    break
            
            if matched_rule:
                po_formula = matched_rule["PO"]
                calculated_payout = payin_value
                
                if "90% of Payin" in po_formula:
                    calculated_payout *= 0.9
                elif "88% of Payin" in po_formula:
                    calculated_payout *= 0.88
                elif "Less 2% of Payin" in po_formula:
                    calculated_payout -= 2
                elif "-2%" in po_formula:
                    calculated_payout -= 2
                elif "-3%" in po_formula:
                    calculated_payout -= 3
                elif "-4%" in po_formula:
                    calculated_payout -= 4
                elif "-5%" in po_formula:
                    calculated_payout -= 5
                
                calculated_payout = max(0, calculated_payout)
                formula_used = po_formula
            else:
                calculated_payout = payin_value
                formula_used = "No matching rule found"
            
            result_record = record.copy()
            result_record['Calculated Payout'] = f"{calculated_payout:.2f}%"
            result_record['Formula Used'] = formula_used
            result_record['Rule Explanation'] = rule_explanation
            
            calculated_data.append(result_record)
            
        except Exception as e:
            logger.error(f"Error processing record: {record}, error: {str(e)}")
            result_record = record.copy()
            result_record['Calculated Payout'] = "Error"
            result_record['Formula Used'] = "Error in calculation"
            result_record['Rule Explanation'] = f"Error: {str(e)}"
            calculated_data.append(result_record)
    
    return calculated_data



def process_files(policy_file_bytes, policy_filename, policy_content_type, company_name):
    """Main processing function with enhanced error handling"""
    try:
        st.info("🔍 Extracting text from policy image...")
        
        extracted_text = extract_text_from_file(policy_file_bytes, policy_filename, policy_content_type)
        logger.info(f"Extracted text length: {len(extracted_text)}, Content: {extracted_text}")

        if not extracted_text.strip():
            logger.error("No text extracted from the image")
            st.error("❌ No text could be extracted. Please ensure the image is clear and contains readable text.")
            return {
                "extracted_text": "",
                "parsed_data": [],
                "calculated_data": [],
                "excel_data": None,
                "df_calc": pd.DataFrame()
            }

        st.success(f"✅ Text extracted successfully! Length: {len(extracted_text)} chars")

        st.info("🧠 Parsing policy data with AI...")
        
        parse_prompt = f"""Analyze this insurance policy text and extract structured data.

Company Name: {company_name}

CRITICAL INSTRUCTIONS:
1. ALWAYS return a valid JSON ARRAY (list) of objects, even if there's only one record or no data is found. If multiple lines, tables, emails, or messages are present (e.g., grids with rows, forwarded messages), create a separate object for each row or entry.
2. Each object must have these EXACT field names:
   - "Segment": Standardized LOB + policy type (e.g., "TW TP", "PVT CAR COMP + SAOD", "Upto 2.5 GVW"). If unsure or no data, use "Unknown".
   - "Location": location/region information (e.g., "Mumbai, Pune", "Nagpur", "RTO Cluster: GOA", states like "GA", "JH", RTOs like "R13", "MP1, MP2", use "N/A" if not found).
   - "Policy Type": policy type details (use "N/A" if not specified, or "COMP/TP" if mixed, "SATP" maps to "TP").
   - "Payin": percentage value (convert decimals: 0.625 → 62.5%, or keep as is: 34%, use "0%" if not found; extract petrol/diesel separately if present).
   - "Doable District": district info or RTO-related (use "N/A" if not found).
   - "Remarks": additional info including vehicle makes, age (e.g., ">6 years"), transaction type, validity, fuel type (e.g., "Petrol: 50%, Diesel: 45%"), notes (e.g., "Decrease by 3%", "No additional deal", "irrespective of Institution"), month/year (e.g., "June 25", "FY 25-26"), and raw text snippets if ambiguous.

3. For Segment field:
   - First, identify the LOB: TW (Two Wheeler, including MCY for motorcycles or scooters), PVT CAR (Private Car or PCI), CV (Commercial Vehicle), BUS, TAXI, MISD (Miscellaneous).
   - Then, determine policy type: TP (Third Party, including SATP), COMP (Comprehensive, or synonyms like package, 1st party, 1+1), SAOD (Stand Alone Own Damage), etc.
   - Use your intelligence to standardize the "Segment" to EXACTLY match one of these predefined values based on the text content, keywords, tonnage, or descriptions. If no match or unclear, use "Unknown":
     - TW-related (including MCY for motorcycles/scooters):
       - If mentions 1+5 year plan, long-term policy, new, or fresh: "1+5"
       - If SAOD + COMP, Comprehensive + Own Damage, package, 1st party, 1+1, or similar full coverage: "TW SAOD + COMP"
       - If TP only, Third Party, or liability-only (including SATP): "TW TP"
     - PVT CAR-related (or PCI):
       - If COMP + SAOD, Comprehensive + Own Damage, package, 1st party, 1+1, or full coverage: "PVT CAR COMP + SAOD"
       - If TP only or Third Party (including SATP): "PVT CAR TP"
     - CV-related (Commercial Vehicle):
       - If upto 2.5 Tn, 2.5 GVW, or similar low tonnage/weight (e.g., "upto 2.5", "2.5 Tn"): "Upto 2.5 GVW"
       - For all other CV cases, higher tonnage (e.g., "2.5 - 3.5 Tn", "3.5-12 Tn", ">3.5-45 Tn", "12-45 Tn"), PCV 3W, GCV 3W, or general CV: "All GVW & PCV 3W, GCV 3W"
     - BUS-related:
       - If School Bus or educational (e.g., "School bus >14 seater", "<11 SC"): "SCHOOL BUS"
       - If Staff Bus, corporate, or other bus types: "STAFF BUS"
     - TAXI-related: "TAXI"
     - MISD-related (Miscellaneous, Tractor, ambulance, police van, garbage van, etc.): "Misd, Tractor"
   - If ambiguous, infer from context (e.g., tonnage for CV, "Private Car SATP" to "PVT CAR TP") and default to "Unknown".

4. For Payin field:
   - If you see decimals like 0.625, convert to 62.5%
   - If you see whole numbers like 34, add % to make 34%
   - If you see percentages, keep them as is (e.g., "50%", "45%", " -72%")
   - Use the value from the "PO" column, "Payin" column, or any percentage in grids (e.g., Petrol/Diesel columns, or "2.5%").
   - If multiple (e.g., Petrol and Diesel), use "Petrol: X%, Diesel: Y%" in Payin or move to Remarks.
   - If not found, use "0%"
   - Do not use values from "Discount" column

5. For Remarks field - extract ALL additional info:
   - Vehicle makes (Tata, Maruti, etc.) → "Vehicle Makes: Tata, Maruti"
   - Age info (>5 years, etc.) → "Age: >5 years"
   - Transaction type (New/Old/Renewal) → "Transaction: New"
   - Validity dates → "Validity till: [date]" (e.g., "1st June 2025 to 30th June 2025", "FY 25-26")
   - Decline RTO information (e.g., "Decline RTO: Dhar, Jhabua")
   - Fuel type (diesel, petrol, etc.) → "Fuel Type: Diesel" (if present)
   - Notes like "Decrease by 3%", "No additional deal", "irrespective of Institution", "Below 11 seater part of YTD target"
   - Combine with semicolons: "Vehicle Makes: Tata; Age: >5 years; Transaction: New; Fuel Type: Diesel" (use "N/A" if nothing found)

IMPORTANT: 
- If a field is not found, use "N/A"
- Return ONLY the JSON array, no other text
- Ensure the JSON is valid and parseable
- Do not extract or include the "Discount" column or its values in any field. Ignore it completely.
- The "PO" column contains the Payin values - use that for the "Payin" field.
- The table may have a "Discount" column - IGNORE it completely. Do not include its values anywhere, not even in remarks.
IGNORE these columns completely - DO NOT extract them:
   - Discount
   - CD1
   - Any column containing "discount" or "cd1" 
   - These are not needed for our analysis

Point to be Noted:
- SATP means TP , so Whenever Private Car SATP is mentioned then it means PVT CAR TP
- sometimes the two columns are also there in the input which are in values , like diesel or petrol, means fuel type also , so Extract that too and parse that too
- Sometimes MCY means motorcycle  , so consider them in TW LOB , be it Motorcycles, scooters
- Handle various formats: line-separated grids (e.g., tonnage with locations and percentages), tables with State/RTO/Fuel, emails with forwarded messages, incentives for YTD targets.
- Extract month/year (e.g., "June 25", "July 25", "Aug 25", "FY 25-26") in Remarks.
- For grids, create one object per row or entry (e.g., each state/RTO as a record).
- If text is ambiguous, infer LOB/Segment from context (e.g., "School bus >14 seater" to "SCHOOL BUS").
- Also if in the put % is mentioned in "-" , this sign is not subtraction but infact it is hyphen symbol 
- Sometimes the values given in input are for example -68% , so in that case also consider it as 68% only and ignore the - sign
- Sometimes segment is also mentioned as Product in the input , and remember PCI means Private Car
- sometimes the two columns are also there in the input which are in values , like diesel or petrol, means fuel type also , so Extract that too and parse that too


### In-Depth Description of the Data in the Provided Screenshots

I'll describe the data from each screenshot in depth, based on the visible content. I've numbered them for clarity based on the order they appear in your query. Each description covers:
- **Format/Structure**: How the data is presented (e.g., table, list, email, message).
- **Content Overview**: Key themes, products, and entities mentioned (e.g., insurance types, locations, percentages).
- **Detailed Breakdown**: Line-by-line or row-by-row analysis, including any notes, dates, and implications.
- **Context and Implications**: What the data seems to represent (e.g., insurance payout grids, revisions, incentives), potential use cases, and patterns.

This is a comprehensive analysis to help you understand the data's organization, patterns, and potential for extraction/processing in your code.

#### **Screenshot 1: Email about PCI Grid for June 2025**
- **Format/Structure**: This is a formal email with a subject line, body text, and a small table. The table is a standard grid with headers and rows, formatted in a word processor or email client (likely Outlook or Gmail). The table has 8 columns and 3 rows (1 header + 2 data rows). It's compact, with some cells merged or spanning multiple columns. Below the table, there's a sign-off and signature.
- **Content Overview**: The email discusses a "PCI Grid" (likely "Private Car Insurance Grid") effective from June 1 to June 30, 2025. It focuses on "PO on OD" (Payout on Own Damage) rates for different fuel types (Diesel vs. Other Than Diesel). Key entities: Probus Insurance Broker, Tata AIG (implied from Cc), locations like "PAN India", and terms like "NCB" (No Claim Bonus). Percentages are positive, indicating payouts or rates.
- **Detailed Breakdown**:
  - **Header**: "Business Type", "Segment", "Fuel Type", "Section Text", "Discount", "RTO", "NCB", "PO on OD".
  - **Row 1**: Business Type: "All", Segment: "All", Fuel Type: "Other Than Diesel", Section Text: "Package/SAOD", Discount: "All", RTO: "PAN India", NCB: "No", PO on OD: "19.50".
  - **Row 2**: Business Type: "All", Segment: "All", Fuel Type: "Other Than Diesel", Section Text: "Package/SAOD", Discount: "All", RTO: "PAN India", NCB: "Yes", PO on OD: "25.00".
  - Body Text: "As discussed over call please find the below PCI Grid effective 1st June 2025 to 30th June 2025." Followed by "Product: PCI".
  - Sign-off: "Thanks, Bhavessh Mistry, Regional Sales Manager, Ahmedabad".
- **Context and Implications**: This appears to be an update on insurance payout rates for private car own damage coverage, differentiated by NCB status. "PAN India" indicates nationwide applicability except possibly specified RTOs. The data is tabular, easy to extract, but note the repeated "Other Than Diesel" – it might imply a focus on non-diesel vehicles. Percentages (19.50, 25.00) are likely commission or payout rates. Patterns: No variation by segment or business type (all "All"), suggesting a uniform policy. For code extraction, focus on table rows, ignoring "Discount" as per your prompt.

#### **Screenshot 2: WhatsApp Message about MH Grid for June 25**
- **Format/Structure**: This is a forwarded WhatsApp message in a dark mode chat interface. The data is presented as a line-separated list (not a table), with each entry on a new line. It's unstructured text, with tonnage categories followed by locations and percentages (e.g., "<2.5 T Mumbai, Pune -6.50x"). No headers; it's a bullet-less list. The message is from "Krunal Sir Probus" and includes a timestamp (11:32).
- **Content Overview**: Titled "Grid Applicable for June 25", focused on "MH" (likely Maharashtra) with CV (Commercial Vehicle) categories by tonnage (e.g., <2.5 T, 2.5-3.5 T). Percentages are multipliers (e.g., -6.50x, -6.20x), possibly payout rates or adjustments. Locations: Mumbai, Pune, Nagpur. Includes "SATP" (Stand Alone Third Party) and age-based entries (>6 years).
- **Detailed Breakdown**:
  - "<2.5 T Mumbai, Pune -6.50x Nagpur -6.20x"
  - "2.5-3.5 T Mumbai, Pune -5.30x"
  - "3.5-12 T Mumbai, Pune -3.10x"
  - "PCV 3W Mumbai, Pune -6.70x"
  - ">3.5-45 T SATP Mumbai, Pune, Nagpur -4.60x"
  - "12-45 T (>6 years) Mumbai, Pune -3.10x"
  - "PO rate for MH - Tata" (possibly a header or note).
  - The message is forwarded, with "Forwarded" label.
- **Context and Implications**: This is a quick update on payout (PO) rates for commercial vehicles in Maharashtra for June 2025. Percentages are negative multipliers, suggesting deductions or rates (e.g., -6.50x might mean a 6.5% deduction scaled by x). Patterns: Rates decrease with higher tonnage, and locations have varying rates. "SATP" maps to TP as per your notes. For code, parse as key-value pairs from lines, mapping to "Segment" (tonnage as CV sub-type), "Location", "Payin" (percentage), "Remarks" (age like ">6 years").

#### **Screenshot 3: Email about Private Car SATP Changes**
- **Format/Structure**: Another formal email with a subject, body, and a table. The table is colored (green for "OPEN"), with 5 columns and 8 rows (1 header + 7 data rows). It's well-structured, likely from a word processor. Sign-off is simple ("Dear Sir").
- **Content Overview**: Subject: "FW: Probus Insurance Broker, Approval for TW Scooter Mopped, Package and SATP". Body discusses implementing "highlighted changes" from June 5, 2025. Focus on "Private Car SATP" (Third Party) rates by fuel (Petrol, Diesel), for different states/RTO clusters. All guidelines are "OPEN" (possibly open for business).
- **Detailed Breakdown**:
  - **Header**: "State", "RTO Cluster", "Current Guidelines", "Petrol", "Diesel".
  - **Row 1**: State: "GA", RTO Cluster: "GOA", Current: "OPEN", "OPEN", Petrol: "55%", Diesel: "45%".
  - **Row 2**: "JH", "JHARKHAND", "OPEN", "OPEN", "50%", "50%".
  - **Row 3**: "MH", "MUMBAI", "OPEN", "OPEN", "58%", "47%".
  - **Row 4**: "MH", "NAGPUR", "OPEN", "OPEN", "45%", "45%".
  - **Row 5**: "TN", "ANDAMAN", "OPEN", "OPEN", "50%", "45%".
  - **Row 6**: "TS", "HYDERABAD", "OPEN", "OPEN", "46%", "45%".
  - **Row 7**: "WB", "ROWB B", "OPEN", "OPEN", "55%", "50%".
  - Body: "As discussed, please implement below highlighted changes from today(5th June 2025)."
  - Forwarded from Bhavessh Mistry, with Cc to Rahul Bansal.
- **Context and Implications**: This is a revision for Private Car Third Party rates, differentiated by fuel type and state/RTO. "OPEN" likely means the cluster is active for sales. Patterns: Petrol rates are higher than Diesel in most cases (e.g., 58% vs 47% for Mumbai). For code, extract as multiple records (one per row), mapping "State" to "Doable District" or "Location", "Petrol/Diesel" to "Payin" with fuel in "Remarks" (e.g., "Fuel Type: Petrol 58%; Diesel 47%").

#### **Screenshot 4: WhatsApp Message about MH Grid for Aug 25**
- **Format/Structure**: Forwarded WhatsApp message in dark mode, similar to Screenshot 2. Line-separated list, no table. Titled "MH Grid Applicable for Aug 25". Timestamp: 09:44.
- **Content Overview**: Similar to June/July grids, but for August 2025. Focus on tonnage categories for CV, with locations and percentages. Includes SATP and age-based entries.
- **Detailed Breakdown**:
  - "<2.5 T Mumbai, Pune -60x Nagpur -60x"
  - "2.5-3.5 T Mumbai, Pune -56x"
  - "3.5-12 T Mumbai, Pune -35x"
  - ">3.5-45 T SATP Mumbai, Pune, Nagpur -51x"
  - "12-45 T (>6 years) Mumbai, Pune -35x"
- **Context and Implications**: Update for August, with rates like -60x (higher deductions than June). Patterns: Consistent structure across months, with slight rate changes (e.g., -6.50x in June to -60x in Aug – possibly a notation change, e.g., -60% or -6.0x). For code, parse lines to "Segment" (tonnage), "Location", "Payin" (percentage), "Remarks" (age).

#### **Screenshot 5: Email about School Bus Incentive**
- **Format/Structure**: Formal email with a table (6 columns, 3 rows: header + 2 data). Table is highlighted in yellow for emphasis. Body text explains "exclusive Call" and notes.
- **Content Overview**: Approval for Probus Insurance Brokers for FY 25-26, focused on "School bus >11 SC" incentives. YTD targets, quarterly rates, and distinctions for Institution vs Individual. Locations: "Pan India except ROTN, MP 1.2 & 3".
- **Detailed Breakdown**:
  - **Header**: "School bus >11 SC", "Pan India except ROTN, MP 1.2 & 3".
  - Sub-headers: "YTD Target (in Cr)", "Additional-Yearly / qtry", "Institution", "Individual", "TATA <11 SC on all RTO's".
  - **Row 1**: YTD: "<35", Yearly/qtry: "2.5%", Institution: "75.00%", Individual: "72.00%", TATA: "50%".
  - **Row 2**: Highlighted: "ROTN-65%, MP1, MP2 & MP3 -67% <11 Seating 50% max"
  - Note: "No additional deal applicable for ROTN and MP1, MP2 and MP3 RTO."
  - Body: "Pls approve for Probus Insurance Brokers Private Limited for FY 25-26."
  - Additional Bullet: "Below 11 seater will be part of over YTD target and will be counted for additional incentive as well."
- **Context and Implications**: Incentive structure for school bus insurance, with targets in Cr (Crores) and rates for institution vs individual. "ROTN-65%" might be a special rate. Patterns: Tiered incentives, with max caps. For code, extract as records for "Institution" and "Individual", "Segment": "SCHOOL BUS", "Payin": percentages, "Remarks": notes and locations.

#### **Screenshot 6: WhatsApp Message about MH Grid for July 25**
- **Format/Structure**: Forwarded WhatsApp message, line-separated list, similar to previous grids. Timestamp: 17:05.
- **Content Overview**: "MH Grid Applicable for July 25", CV tonnage rates for Maharashtra.
- **Detailed Breakdown**:
  - "<2.5 T Mumbai, Pune -68x Nagpur -65x"
  - "2.5-3.5 T Mumbai, Pune -58x"
  - "3.5-12 T Mumbai, Pune -35x"
  - ">3.5-45 T SATP Mumbai, Pune, Nagpur -51x"
  - "12-45 T (>6 years) Mumbai, Pune -35x"
- **Context and Implications**: July update, rates between June and August (e.g., -68x for <2.5 T Mumbai). Patterns: Monthly revisions, decreasing deductions over time? For code, same parsing as other grids.

#### **Screenshot 7: WhatsApp Message about Rajasthan CV SATP Revised Grid**
- **Format/Structure**: Forwarded WhatsApp message, short list with 2 entries. Timestamp: 22:25.
- **Content Overview**: "Dear Sir, Rajasthan CV SATP revised grid". RTO clusters with percentages.
- **Detailed Breakdown**:
  - "R13 RTO cluster 12 to 45T -26%"
  - "R14 RTO cluster 12 to 35T -26%"
  - Note: "Decrease by 3% From Tata"
- **Context and Implications**: Revision for CV SATP in Rajasthan, focused on tonnage and RTO clusters. "Decrease by 3%" implies rate reduction. Patterns: Specific to state, tonnage-based. For code, map to "Segment": "All GVW & PCV 3W, GCV 3W", "Location": "R13, R14", "Payin": "-26%".

#### **Screenshot 8: WhatsApp Message with UP Percentages and MCY TW SATP**
- **Format/Structure**: WhatsApp message in dark mode, line-separated list. Timestamp: 15:36.
- **Content Overview**: UP rates (UP1 to UP4) and "MCY TW SATP", with confirmation request.
- **Detailed Breakdown**:
  - "UP 1 38%"
  - "UP 2 30%"
  - "UP 3 30%"
  - "Up4 38%"
  - "MCY TW SATP"
  - "pl confirm"
  - Response: "Ok"
- **Context and Implications**: UP (Uttar Pradesh?) rates for TW SATP (Two Wheeler TP). Patterns: Location-specific rates, with "MCY" as motorcycle. For code, map to "Segment": "TW TP", "Location": "UP1", "Payin": "38%", etc.

#### **Screenshot 9: WhatsApp Chat Context**
- **Format/Structure**: Chat history with messages, timestamps, and avatars. Not data-heavy, more conversational.
- **Content Overview**: Discussion about meetings, policies, and API improvements. Mentions "10 days ma 1000 policies vadhi.. infact tw bike tp occhi thai chee" (Gujarati for "increased 1000 policies in 10 days, in fact TW bike TP has become less").
- **Detailed Breakdown**:
  - Messages: Scheduling meetings, "no today meeting scheduled", "overall no improved. its of API Only".
  - No grid data; contextual chat.
- **Context and Implications**: Background conversation, possibly about performance metrics. Not core data, but useful for date context (e.g., current discussions).

Overall Patterns Across Screenshots:
- **Data Types**: Mostly grids/lists of insurance rates by segment (CV tonnage, BUS seater, PVT CAR fuel), location (states/RTOs), and month (June-Aug 2025).
- **Common Elements**: Percentages (positive/negative), locations (Mumbai, Pune, Nagpur, PAN India, states like MH, RJ, MP), fuel types, age/tonnage conditions.
- **Variations**: Emails are tabular/formal, WhatsApp are list-based/informal. Negative percentages likely deductions, positive are payouts/incentives.
- **Use for Code**: Your prompt should parse tables/lists into multiple records, handle negatives, extract fuel/age to Remarks, map SATP to TP, and infer LOB from context (e.g., "School bus" to BUS, "MCY" to TW).

If you need more specifics or code adjustments based on this, let me know!

### In-Depth Description of the Data in the Provided Screenshot

This screenshot appears to be a formal email or document (likely from an insurance approval process) requesting approval for incentive structures related to school bus insurance for the financial year (FY) 25-26. It's focused on Probus Insurance Brokers Private Limited. I'll break it down comprehensively, similar to my previous analysis, covering the overall structure, content, detailed elements, and implications. The data is tabular with supporting text, emphasizing targets, rates, and exceptions.

#### **Format/Structure**
- **Overall Layout**: This is a semi-formal email or approval document, likely created in Microsoft Word or Outlook and shared as an image/PDF. It starts with a greeting ("Dear Sir,"), followed by a request paragraph, a table, a highlighted note, and a bullet point. The table is the core data element, with 5 columns and 2 rows (1 header + 1 data row), plus a sub-note. The layout is clean but has some formatting artifacts (e.g., repeated text like "School bus >11 SC" due to possible OCR extraction or image compression). There's a smiley emoji at the end, suggesting it was copied from a chat or email with casual elements. The document is vertical, with the table spanning the width.
- **Table Structure**: 
  - **Columns**: 5 (merged in places for readability).
  - **Rows**: 1 header row + 1 data row + 1 highlighted sub-row/note.
  - **Styling**: The sub-note is highlighted in yellow, indicating emphasis. Percentages are formatted with decimals (e.g., "75.00%"). Text is bolded for headers and key notes.
- **Length and Density**: Compact (fits one screen), but dense with financial terms. No images or attachments visible; it's text-heavy.

#### **Content Overview**
- **Theme**: This is an incentive and target approval request for school bus insurance sales, specifically for buses with more than 11 seats ("School bus >11 SC"). It covers FY 25-26 (April 2025–March 2026) and includes Year-to-Date (YTD) targets in crores (₹ Cr), quarterly/yearly additional incentives, and differentiated rates for "Institution" (e.g., schools) vs. "Individual" (private buyers), plus a special rate for TATA vehicles. The focus is on "Pan India except ROTN, MP1 & 2 & 3" (nationwide except specific RTO regions in Madhya Pradesh). Key numbers are percentages (e.g., 75.00%, 72.00%) representing incentive rates, with caps and exceptions. The document emphasizes no additional deals in certain regions and counts smaller buses toward overall targets.
- **Key Entities**:
  - **Company/Broker**: Probus Insurance Brokers Private Limited.
  - **Product/Lob**: School Bus (>11 SC), likely under BUS LOB in your formula data.
  - **Rates/Incentives**: Percentages for institutions/individuals (75-72%), TATA special (50%), with caps like "<11 Seating 50% max".
  - **Locations/Exclusions**: Pan India, excluding ROTN (possibly "Rest of Tamil Nadu" or a specific RTO code) and MP1, MP2, MP3 (Madhya Pradesh RTO clusters).
  - **Dates**: FY 25-26; targets are ">35 Cr" (more than 35 crores).

#### **Detailed Breakdown**
- **Greeting and Request Paragraph**:
  - "Dear Sir," (standard formal greeting).
  - "Pls approve for Probus Insurance Brokers Private Limited for FY 25-26." (Core request for approval of the incentive structure).
  - Repeated text ("Pls approve... for FY 25-26.") – likely a duplication from OCR or copy-paste error in the image.

- **Table**:
  - **Header Row** (merged columns):
    - Left: "School bus >11 SC" (Segment/Lob identifier).
    - Right (merged): "Pan India except ROTN, MP1 & 2 & 3" (Location/Scope).
  - **Sub-Headers** (under the right merged cell):
    - "YTD Target (in Cr)", ">35" (target: more than 35 crores for the year).
    - "Additional-Yearly / qtry" (quarterly/yearly incentives), value: "2.5%" (2.5% on quarterly sales).
    - "Institution" (for institutional buyers like schools), value: "75.00%" (75% incentive rate).
    - "Individual" (for private buyers), value: "72.00%" (72% incentive rate).
    - "TATA <11 SC on all RTO's" (special rate for TATA vehicles under 11 seats, across all Regional Transport Offices), value: "50%" (50% rate).
  - **Highlighted Sub-Row/Note** (yellow background, under the table):
    - "* ROTN-65%, MP1, MP2 & MP3 -67%" (Asterisk for footnote; reduced rates: 65% for ROTN, 67% for MP clusters – likely for institutions/individuals).
    - "MP1, MP2 & MP3 -67%, <11 Seating 50% max" (Cap for smaller buses: max 50% incentive in MP regions).
    - "No additional deal applicable for ROTN and MP1, MP2 and MP3 RTO." (Exclusion: No extra incentives in these regions).

- **Bullet Point**:
  - "• Below 11 seater will be part of over YTD target and will be counted for additional incentive as well." (Note: Smaller buses (<11 seats) contribute to the overall target and qualify for incentives, despite the cap).

- **Footer/Contextual Elements**:
  - Repeated table snippets (e.g., "School bus >11 SC") – likely image artifacts.
  - Emoji (😊) at the end – suggests this was shared in a chat or email with a friendly tone.

#### **Context and Implications**
- **Business Context**: This is an internal approval document for sales incentives in the school bus insurance segment (likely under "BUS" LOB in your formula data, mapping to "SCHOOL BUS" segment). It's aimed at brokers like Probus to meet FY targets through higher incentives for larger vehicles (>11 seats). The structure encourages volume sales (YTD >35 Cr) with tiered rates: higher for institutions (75%) than individuals (72%), and lower for TATA small vehicles (50%). Exclusions in specific RTOs (ROTN, MP clusters) prevent over-incentivization in low-margin areas. The "2.5%" is a quarterly bonus on top of base rates.
- **Data Patterns**:
  - **Numerical**: Targets in Cr (₹), rates as percentages (75.00%, 72.00%, 50%, 65%, 67%). All positive, indicating incentives (not deductions like in CV grids).
  - **Hierarchical**: Targets → Quarterly bonuses → Buyer-type rates → Caps/Exclusions.
  - **Geographic**: Pan India with exclusions (ROTN, MP1-3), implying regional variations.
  - **Conditional**: ">11 SC" (seats), "<11 Seating 50% max", "irrespective of Institution/non Institution" (from previous screenshots, but similar here).
- **Implications for Processing**:
  - **Extraction**: Parse the table as one primary record for "SCHOOL BUS", with sub-details in "Remarks" (e.g., "Institution: 75.00%; Individual: 72.00%; TATA <11 SC: 50%; ROTN: 65%; MP1-3: 67%; Cap: <11 Seating 50% max; No additional deal in ROTN/MP; Below 11 seater counts toward YTD target").
  - **Mapping to Your System**:
    - **Segment**: "SCHOOL BUS" (direct match).
    - **Payin**: Use the highest rate (75.00%) or average (72.50%), but since it's incentives, treat as "Payin" for calculation; include variants in Remarks.
    - **Location**: "Pan India except ROTN, MP1, MP2, MP3".
    - **Doable District**: "MP1, MP2, MP3" (as exclusions).
    - **Remarks**: Age/condition: N/A; Fuel: N/A; Notes: "YTD Target >35 Cr; Quarterly 2.5%; Below 11 seater counts for additional incentive".
  - **Potential Issues**: Repeated text could confuse OCR; yellow highlight might be lost in extraction – prompt should ignore styling.
  - **Business Insight**: This incentivizes large-volume school bus sales (institutional focus), with safeguards in specific regions. Total potential: If target is >35 Cr at 72% average rate, incentives could be ~25 Cr payout.

This screenshot fits the pattern of incentive approvals, contrasting with rate grids in other images (deductions vs. bonuses). If you need a code update to handle this specific format or extract it as multiple records (e.g., one per rate type), let me know!
Text to analyze:
{extracted_text}
"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a data extraction expert. Extract policy data as a JSON array. Convert all Payin values to percentage format. Always return valid JSON array with complete field names. Extract all additional information for remarks."
                    },
                    {"role": "user", "content": parse_prompt}
                ],
                temperature=0.0,
                max_tokens=4000
            )
            
            parsed_json = response.choices[0].message.content.strip()
            logger.info(f"Raw parsing response: {parsed_json[:500]}...")
            
            cleaned_json = clean_json_response(parsed_json)
            logger.info(f"Cleaned JSON: {cleaned_json[:500]}...")
            
            try:
                policy_data = json.loads(cleaned_json)
                policy_data = ensure_list_format(policy_data)
                
                if not policy_data or len(policy_data) == 0:
                    raise ValueError("Parsed data is empty")
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)} with cleaned JSON: {cleaned_json}")
                st.warning("⚠️ AI response was not valid JSON. Creating fallback structure...")
                policy_data = [{
                    "Segment": "Unknown",
                    "Location": "N/A",
                    "Policy Type": "N/A", 
                    "Payin": "0%",
                    "Doable District": "N/A",
                    "Remarks": f"Failed to parse - please check image quality. Extract manually from: {extracted_text[:200]}"
                }]
        
        except Exception as e:
            logger.error(f"Error in AI parsing: {str(e)} with raw response: {parsed_json[:500]}...")
            st.warning("⚠️ AI parsing failed. Creating fallback structure...")
            policy_data = [{
                "Segment": "Unknown",
                "Location": "N/A",
                "Policy Type": "N/A",
                "Payin": "0%",
                "Doable District": "N/A",
                "Remarks": f"Parsing error: {str(e)}"
            }]

        st.success(f"✅ Successfully parsed {len(policy_data)} policy records")

        for record in policy_data:
            try:
                if 'Discount' in record:
                    del record['Discount']
                payin_val, payin_cat = classify_payin(record.get('Payin', '0%'))
                record['Payin_Value'] = payin_val
                record['Payin_Category'] = payin_cat
            except Exception as e:
                logger.warning(f"Error classifying payin: {e}")
                record['Payin_Value'] = 0.0
                record['Payin_Category'] = "Payin Below 20%"

        st.info("🧮 Applying formulas and calculating payouts...")
        calculated_data = apply_formula_directly(policy_data, company_name)
        
        if not calculated_data or len(calculated_data) == 0:
            st.error("❌ No data after formula application")
            return {
                "extracted_text": extracted_text,
                "parsed_data": policy_data,
                "calculated_data": [],
                "excel_data": None,
                "df_calc": pd.DataFrame()
            }

        st.success(f"✅ Successfully calculated {len(calculated_data)} records")

        st.info("📊 Creating Excel file...")
        
        df_calc = pd.DataFrame(calculated_data)
        
        if df_calc.empty:
            st.error("❌ DataFrame is empty")
            return {
                "extracted_text": extracted_text,
                "parsed_data": policy_data,
                "calculated_data": calculated_data,
                "excel_data": None,
                "df_calc": df_calc
            }

        output = BytesIO()
        try:
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_calc.to_excel(writer, sheet_name='Policy Data', startrow=2, index=False)
                worksheet = writer.sheets['Policy Data']

                headers = list(df_calc.columns)
                for col_num, value in enumerate(headers, 1):
                    cell = worksheet.cell(row=3, column=col_num, value=value)
                    cell.font = cell.font.copy(bold=True)

                if len(headers) > 1:
                    company_cell = worksheet.cell(row=1, column=1, value=company_name)
                    worksheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
                    company_cell.font = company_cell.font.copy(bold=True, size=14)
                    company_cell.alignment = company_cell.alignment.copy(horizontal='center')

                    title_cell = worksheet.cell(row=2, column=1, value='Policy Data with Payin and Calculated Payouts')
                    worksheet.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(headers))
                    title_cell.font = title_cell.font.copy(bold=True, size=12)
                    title_cell.alignment = title_cell.alignment.copy(horizontal='center')
                else:
                    worksheet.cell(row=1, column=1, value=company_name)
                    worksheet.cell(row=2, column=1, value='Policy Data with Payin and Calculated Payouts')

        except Exception as e:
            logger.error(f"Error creating Excel file: {str(e)}")
            st.error(f"❌ Error creating Excel: {str(e)}")
            return {
                "extracted_text": extracted_text,
                "parsed_data": policy_data,
                "calculated_data": calculated_data,
                "excel_data": None,
                "df_calc": df_calc
            }

        output.seek(0)
        excel_data = output.read()

        return {
            "extracted_text": extracted_text,
            "parsed_data": policy_data,
            "calculated_data": calculated_data,
            "excel_data": excel_data,
            "df_calc": df_calc
        }

    except Exception as e:
        logger.error(f"Unexpected error in process_files: {str(e)}", exc_info=True)
        st.error(f"❌ Processing error: {str(e)}")
        return {
            "extracted_text": "",
            "parsed_data": [],
            "calculated_data": [],
            "excel_data": None,
            "df_calc": pd.DataFrame()
        }
def main():
    st.set_page_config(
        page_title="Insurance Policy Processing", 
        page_icon="📋", 
        layout="wide"
    )
    
    st.title("🏢 Insurance Policy Processing System")
    st.markdown("---")
    
    with st.sidebar:
        st.header("📁 File Upload")
        
        company_name = st.text_input(
            "Company Name", 
            value="Unknown Company",
            help="Enter the insurance company name"
        )
        
        policy_file = st.file_uploader(
            "📄 Upload Policy Image",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'],
            help="Upload your insurance policy image"
        )
        
        st.info("📊 Formula rules are embedded in the system and will be automatically applied.")
        
        process_button = st.button(
            "🚀 Process Policy File", 
            type="primary",
            disabled=not policy_file
        )
    
    if not policy_file:
        st.info("👆 Please upload a policy image to begin processing.")
        st.markdown("""
        ### 📋 Instructions:
        1. **Company Name**: Enter the insurance company name
        2. **Policy Image**: Upload the image containing policy data (PNG, JPG, etc.)
        3. **Process**: Click the process button to extract data and calculate payouts
        
        ### 🎯 Features:
        - **Image Support**: PNG, JPG, JPEG, GIF, BMP, TIFF
        - **AI-Powered Extraction**: Uses GPT-4o for intelligent text extraction via OCR
        - **Enhanced Remarks Extraction**: Automatically detects and extracts:
          - Vehicle make information (Tata, Maruti, etc.)
          - Age information (>5 years, etc.)
          - Transaction type (New/Old/Renewal)
          - Validity dates
        - **Smart Formula Application**: Uses embedded formula rules for accurate calculations
        - **Excel Export**: Download processed data as formatted Excel file
        """)
        return
    
    if process_button:
        try:
            policy_file_bytes = policy_file.read()
            
            if len(policy_file_bytes) == 0:
                st.error("❌ Uploaded file is empty. Please upload a valid image.")
                return
            
            with st.spinner("Processing policy image... This may take a few moments."):
                results = process_files(
                    policy_file_bytes, 
                    policy_file.name, 
                    policy_file.type,
                    company_name
                )
            
            if not results["calculated_data"] or len(results["calculated_data"]) == 0:
                st.error("❌ No data was extracted or processed. Please check the image quality and try again.")
                if results["extracted_text"]:
                    with st.expander("📝 View Extracted Text"):
                        st.text_area("Extracted Text", results["extracted_text"], height=300)
                if results["parsed_data"]:
                    with st.expander("🧾 View Parsed Data"):
                        st.json(results["parsed_data"])
                return
            
            st.success("🎉 Processing completed successfully!")
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Final Results", 
                "📝 Extracted Text", 
                "🧾 Parsed Data", 
                "🧮 Calculated Data",
                "📥 Download"
            ])
            
            with tab1:
                st.subheader("📊 Final Processed Data")
                st.dataframe(results["df_calc"], use_container_width=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", len(results["calculated_data"]))
                with col2:
                    if len(results["calculated_data"]) > 0:
                        avg_payin = sum([r.get('Payin_Value', 0) for r in results["calculated_data"]]) / len(results["calculated_data"])
                        st.metric("Avg Payin", f"{avg_payin:.1f}%")
                    else:
                        st.metric("Avg Payin", "0.0%")
                with col3:
                    segments = set([r.get('Segment', 'N/A') for r in results["calculated_data"]])
                    st.metric("Unique Segments", len(segments))
                with col4:
                    st.metric("Company", company_name)
                
                st.subheader("🔧 Formula Rules Applied")
                formula_summary = {}
                for record in results["calculated_data"]:
                    formula = record.get('Formula Used', 'Unknown')
                    formula_summary[formula] = formula_summary.get(formula, 0) + 1
                
                for formula, count in formula_summary.items():
                    st.write(f"• **{formula}**: Applied to {count} record(s)")

            with tab2:
                st.subheader("📝 Extracted Text from Policy Image")
                st.text_area("Policy Text", results["extracted_text"], height=400, key="policy_text")
                
                st.subheader("📊 Embedded Formula Rules")
                df_formula = pd.DataFrame(FORMULA_DATA)
                st.dataframe(df_formula, use_container_width=True)
            
            with tab3:
                st.subheader("🧾 Parsed Policy Data")
                st.json(results["parsed_data"])
            
            with tab4:  
                st.subheader("🧮 Calculated Data with Formulas")
                st.json(results["calculated_data"])
                
                st.subheader("🔍 Rule Explanations")
                for i, record in enumerate(results["calculated_data"]):
                    with st.expander(f"Record {i+1}: {record.get('Segment', 'Unknown')}"):
                        st.write(f"**Payin**: {record.get('Payin', 'N/A')}")
                        st.write(f"**Calculated Payout**: {record.get('Calculated Payout', 'N/A')}")
                        st.write(f"**Formula Used**: {record.get('Formula Used', 'N/A')}")
                        st.write(f"**Rule Explanation**: {record.get('Rule Explanation', 'N/A')}")
            
            with tab5:
                st.subheader("📥 Download Results")
                
                if results["excel_data"]:
                    st.download_button(
                        label="📊 Download Excel File",
                        data=results["excel_data"],
                        file_name=f"{company_name}_processed_policies.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.warning("⚠️ Excel file could not be generated")
                
                json_data = json.dumps(results["calculated_data"], indent=2)
                st.download_button(
                    label="📄 Download JSON Data",
                    data=json_data,
                    file_name=f"{company_name}_processed_data.json",
                    mime="application/json"
                )
                
                if not results["df_calc"].empty:
                    csv_data = results["df_calc"].to_csv(index=False)
                    st.download_button(
                        label="📋 Download CSV File",
                        data=csv_data,
                        file_name=f"{company_name}_processed_policies.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("⚠️ CSV file could not be generated")
                
                st.info("💡 The Excel file contains formatted data with company header and calculated payouts.")
                
        except Exception as e:
            st.error(f"❌ Error processing files: {str(e)}")
            logger.error(f"Main processing error: {str(e)}", exc_info=True)
            st.exception(e)


if __name__ == "__main__":
    main()
