
# import streamlit as st
# import pandas as pd
# from io import BytesIO
# import base64
# import json
# import os
# from dotenv import load_dotenv
# import logging
# import re

# # Check if required packages are available
# try:
#     from openai import OpenAI
# except ImportError:
#     st.error("OpenAI package not found. Please install it using 'pip install openai'")
#     st.stop()

# try:
#     import PyPDF2
# except ImportError:
#     st.warning("PyPDF2 not found. PDF text extraction will use OpenAI vision only.")
#     PyPDF2 = None

# try:
#     import pdfplumber
# except ImportError:
#     st.warning("pdfplumber not found. PDF text extraction will use PyPDF2 or OpenAI vision only.")
#     pdfplumber = None

# # Load environment variables
# load_dotenv()

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load OpenAI API key
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     st.error("OPENAI_API_KEY environment variable not set. Please set it in your .env file or environment variables.")
#     st.stop()

# # Initialize OpenAI client
# try:
#     client = OpenAI(api_key=OPENAI_API_KEY)
# except Exception as e:
#     st.error(f"Failed to initialize OpenAI client: {str(e)}")
#     st.stop()

# # Embedded Formula Data
# FORMULA_DATA = [
#     {"LOB": "TW", "SEGMENT": "1+5", "INSURER": "All Companies", "PO": "90% of Payin", "REMARKS": "NIL"},
#     {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "INSURER": "All Companies", "PO": "90% of Payin", "REMARKS": "NIL"},
#     {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "INSURER": "DIGIT", "PO": "-2%", "REMARKS": "Payin Below 20%"},
#     {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "INSURER": "DIGIT", "PO": "-3%", "REMARKS": "Payin 21% to 30%"},
#     {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "INSURER": "DIGIT", "PO": "-4%", "REMARKS": "Payin 31% to 50%"},
#     {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "INSURER": "DIGIT", "PO": "-5%", "REMARKS": "Payin Above 50%"},
#     {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Bajaj, Digit, ICICI", "PO": "-2%", "REMARKS": "Payin Below 20%"},
#     {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Bajaj, Digit, ICICI", "PO": "-3%", "REMARKS": "Payin Above 20%"},
#     {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Rest of Companies", "PO": "-2%", "REMARKS": "Payin Below 20%"},
#     {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Rest of Companies", "PO": "-3%", "REMARKS": "Payin 21% to 30%"},
#     {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Rest of Companies", "PO": "-4%", "REMARKS": "Payin 31% to 50%"},
#     {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Rest of Companies", "PO": "-5%", "REMARKS": "Payin Above 50%"},
#     {"LOB": "PVT CAR", "SEGMENT": "PVT CAR COMP + SAOD", "INSURER": "All Companies", "PO": "90% of Payin", "REMARKS": "All Fuel"},
#     {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "Bajaj, Digit, SBI", "PO": "-2%", "REMARKS": "Payin Below 20%"},
#     {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "Bajaj, Digit, SBI", "PO": "-3%", "REMARKS": "Payin Above 20%"},
#     {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "Rest of Companies", "PO": "90% of Payin", "REMARKS": "Zuno -  21"},
#     {"LOB": "CV", "SEGMENT": "Upto 2.5 GVW", "INSURER": "Reliance, SBI, Tata", "PO": "-2%", "REMARKS": "NIL"},
#     {"LOB": "CV", "SEGMENT": "All GVW & PCV 3W, GCV 3W", "INSURER": "Rest of Companies", "PO": "-2%", "REMARKS": "Payin Below 20%"},
#     {"LOB": "CV", "SEGMENT": "All GVW & PCV 3W, GCV 3W", "INSURER": "Rest of Companies", "PO": "-3%", "REMARKS": "Payin 21% to 30%"},
#     {"LOB": "CV", "SEGMENT": "All GVW & PCV 3W, GCV 3W", "INSURER": "Rest of Companies", "PO": "-4%", "REMARKS": "Payin 31% to 50%"},
#     {"LOB": "CV", "SEGMENT": "All GVW & PCV 3W, GCV 3W", "INSURER": "Rest of Companies", "PO": "-5%", "REMARKS": "Payin Above 50%"},
#     {"LOB": "BUS", "SEGMENT": "SCHOOL BUS", "INSURER": "TATA, Reliance, Digit, ICICI", "PO": "Less 2% of Payin", "REMARKS": "NIL"},
#     {"LOB": "BUS", "SEGMENT": "SCHOOL BUS", "INSURER": "Rest of Companies", "PO": "88% of Payin", "REMARKS": "NIL"},
#     {"LOB": "BUS", "SEGMENT": "STAFF BUS", "INSURER": "All Companies", "PO": "88% of Payin", "REMARKS": "NIL"},
#     {"LOB": "TAXI", "SEGMENT": "TAXI", "INSURER": "All Companies", "PO": "-2%", "REMARKS": "Payin Below 20%"},
#     {"LOB": "TAXI", "SEGMENT": "TAXI", "INSURER": "All Companies", "PO": "-3%", "REMARKS": "Payin 21% to 30%"},
#     {"LOB": "TAXI", "SEGMENT": "TAXI", "INSURER": "All Companies", "PO": "-4%", "REMARKS": "Payin 31% to 50%"},
#     {"LOB": "TAXI", "SEGMENT": "TAXI", "INSURER": "All Companies", "PO": "-5%", "REMARKS": "Payin Above 50%"},
#     {"LOB": "MISD", "SEGMENT": "Misd, Tractor", "INSURER": "All Companies", "PO": "88% of Payin", "REMARKS": "NIL"}
# ]


# def extract_text_from_pdf_file(pdf_bytes: bytes) -> str:
#     """Extract text from PDF using multiple methods"""
#     extracted_text = ""
    
#     # Method 1: Try pdfplumber first (most accurate)
#     if pdfplumber:
#         try:
#             with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
#                 for page in pdf.pages:
#                     page_text = page.extract_text()
#                     if page_text:
#                         extracted_text += page_text + "\n"
#             if extracted_text.strip():
#                 logger.info("PDF text extracted using pdfplumber")
#                 return extracted_text.strip()
#         except Exception as e:
#             logger.warning(f"pdfplumber extraction failed: {str(e)}")
    
#     # Method 2: Try PyPDF2 as fallback
#     if PyPDF2:
#         try:
#             pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
#             for page in pdf_reader.pages:
#                 page_text = page.extract_text()
#                 if page_text:
#                     extracted_text += page_text + "\n"
#             if extracted_text.strip():
#                 logger.info("PDF text extracted using PyPDF2")
#                 return extracted_text.strip()
#         except Exception as e:
#             logger.warning(f"PyPDF2 extraction failed: {str(e)}")
    
#     # Method 3: Return empty string if all methods fail
#     logger.warning("All PDF text extraction methods failed")
#     return ""


# def extract_text_from_file(file_bytes: bytes, filename: str, content_type: str) -> str:
#     """Extract text from uploaded file"""
#     file_extension = filename.split('.')[-1].lower() if '.' in filename else ''
#     file_type = content_type if content_type else file_extension

#     # Image-based extraction with enhanced OCR
#     image_extensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']
#     if file_extension in image_extensions or file_type.startswith('image/'):
#         image_base64 = base64.b64encode(file_bytes).decode('utf-8')
        
#         prompt = """Extract all insurance policy text accurately from this image using OCR.
#         Pay special attention to identifying:
#         - Segment information (like TW, PVT CAR, CV, BUS, TAXI, MISD)
#         - Company names
#         - Policy types
#         - Payin/Payout percentages
#         - Any numerical values
#         - Location information
#         Extract all text exactly as it appears."""
            
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[{
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": prompt},
#                     {"type": "image_url", "image_url": {"url": f"data:image/{file_extension};base64,{image_base64}"}}
#                 ]
#             }],
#             temperature=0.1
#         )
#         return response.choices[0].message.content.strip()

#     # Enhanced PDF extraction
#     if file_extension == 'pdf':
#         # First try to extract text directly from PDF
#         pdf_text = extract_text_from_pdf_file(file_bytes)
        
#         if pdf_text and len(pdf_text.strip()) > 50:
#             # If we got good text extraction, use it
#             logger.info("Using direct PDF text extraction")
#             return pdf_text
#         else:
#             # Fallback to OpenAI vision for PDF
#             logger.info("Using OpenAI vision for PDF")
#             pdf_base64 = base64.b64encode(file_bytes).decode("utf-8")
            
#             prompt = """Extract insurance policy details from this PDF.
#             Focus on identifying segments, company names, policy types, and payout percentages."""
                
#             response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[{
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": prompt},
#                         {"type": "image_url", "image_url": {"url": f"data:application/pdf;base64,{pdf_base64}"}}
#                     ]
#                 }],
#                 temperature=0.1
#             )
#             return response.choices[0].message.content.strip()

#     # Text files
#     if file_extension == 'txt':
#         return file_bytes.decode('utf-8', errors='ignore')

#     # CSV files
#     if file_extension == 'csv':
#         df = pd.read_csv(BytesIO(file_bytes), encoding='utf-8', errors='ignore')
#         return df.to_string()

#     # Excel files
#     if file_extension in ['xlsx', 'xls']:
#         all_sheets = pd.read_excel(BytesIO(file_bytes), sheet_name=None)
#         dfs = []
#         for sheet_name, df_sheet in all_sheets.items():
#             df_sheet["Source_Sheet"] = sheet_name
#             dfs.append(df_sheet)
#         df = pd.concat(dfs, ignore_index=True, join="outer")
#         return df.to_string(index=False)

#     raise ValueError(f"Unsupported file type for {filename}")


# def clean_json_response(response_text: str) -> str:
#     """Clean and extract JSON from OpenAI response"""
#     # Remove markdown code blocks
#     cleaned = re.sub(r'```json\s*', '', response_text)
#     cleaned = re.sub(r'```\s*$', '', cleaned)
    
#     # Remove any text before the first [ or {
#     json_start = -1
#     for i, char in enumerate(cleaned):
#         if char in '[{':
#             json_start = i
#             break
    
#     if json_start != -1:
#         cleaned = cleaned[json_start:]
    
#     # Remove any text after the last ] or }
#     json_end = -1
#     for i in range(len(cleaned) - 1, -1, -1):
#         if cleaned[i] in ']}':
#             json_end = i + 1
#             break
    
#     if json_end != -1:
#         cleaned = cleaned[:json_end]
    
#     return cleaned.strip()


# def ensure_list_format(data) -> list:
#     """Ensure data is in list format"""
#     if isinstance(data, list):
#         return data
#     elif isinstance(data, dict):
#         return [data]  # Convert single object to list
#     else:
#         raise ValueError(f"Expected list or dict, got {type(data)}")


# def classify_payin(payin_str):
#     """
#     Converts Payin string (e.g., '50%') to float and classifies its range.
#     """
#     try:
#         # Handle various formats
#         payin_clean = str(payin_str).replace('%', '').replace(' ', '')
#         payin_value = float(payin_clean)
        
#         if payin_value <= 20:
#             category = "Payin Below 20%"
#         elif 21 <= payin_value <= 30:
#             category = "Payin 21% to 30%"
#         elif 31 <= payin_value <= 50:
#             category = "Payin 31% to 50%"
#         else:
#             category = "Payin Above 50%"
#         return payin_value, category
#     except (ValueError, TypeError):
#         logger.warning(f"Could not parse payin value: {payin_str}")
#         return 0.0, "Payin Below 20%"


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


# def process_files(policy_file_bytes, policy_filename, policy_content_type, company_name):
#     """Main processing function"""
#     try:
#         st.info("🔍 Extracting text from policy file...")
        
#         # Extract text with enhanced intelligence
#         extracted_text = extract_text_from_file(policy_file_bytes, policy_filename, policy_content_type)

#         logger.info(f"Extracted text length: {len(extracted_text)}")

#         st.success(f"✅ Text extracted successfully! Policy: {len(extracted_text)} chars")

#         # Parse policy data with enhanced segment identification
#         st.info("🧠 Parsing policy data with AI...")
        
#         parse_prompt = f"""
#         Analyze the following text, which contains insurance policy details.
#         Use your intelligence to identify and extract the data accurately.

#         Company Name: {company_name}

#         IMPORTANT: Always return a valid JSON array (list) of records, even if there's only one record.

#         Extract into JSON records with these exact fields:
#         - "Segment": identify LOB like TW, PVT CAR, CV, BUS, TAXI, MISD and policy type like TP, COMP, SAOD. 
#           For CV (Commercial Vehicle), also extract weight/tonnage information like "upto 2.5 Tn", "2.5 GVW", etc.
#           PRESERVE EXACT TONNAGE/WEIGHT INFORMATION as mentioned in the text.
#         - "Location": location information
#         - "Policy Type": policy type information
#         - "Payin": convert payout/payrate values to percentage format, e.g. 0.625 → 62.5%, 34 → 34%
#         - "Doable District": district information
#         - "Remarks": any additional information INCLUDING vehicle make information

#         ENHANCED SEGMENT EXTRACTION FOR CV:
#         - If text mentions "upto 2.5 Tn", "2.5 GVW", "upto 2.5", extract as "CV upto 2.5 Tn"
#         - If text mentions larger tonnages or generic CV, extract as "CV"
#         - PRESERVE the exact tonnage/weight specification from the original text

#         ENHANCED REMARKS FIELD INSTRUCTIONS:
#         - Extract ALL additional information that doesn't fit in other specific fields
#         - Include any notes, conditions, special instructions, exclusions, etc.
#         - ALSO look for vehicle make information in the text (e.g., "Make – Tata, Maruti" or similar patterns)
#         - Vehicle makes include brands like: TATA, MARUTI, SUZUKI, HYUNDAI, HONDA, TOYOTA, MAHINDRA, BAJAJ, HERO, TVS, YAMAHA, etc.
#         - ALSO look for AGE information (e.g., ">5 years", "Age: 3 years", "Below 5 years", etc.)
#         - ALSO look for TRANSACTION information (e.g., "New", "Old", "Renewal", "Fresh", "Transaction: New/Old", etc.)
#         - ALSO look for VALIDITY information (e.g., "Validity till 30th Sep'25", etc.)
#         - If vehicle makes are found, include them: "Vehicle Makes: [make1, make2, ...]"
#         - If age information is found, include it: "Age: [age_info]"
#         - If transaction type is found, include it: "Transaction: [New/Old/Renewal]"
#         - If validity is found, include it: "Validity till [date]"
#         - Combine all relevant additional information into the remarks field, separated by semicolons
#         - Example format: "Vehicle Makes: Tata, Maruti; Age: >5 years; Transaction: Old; Validity till 30th Sep'25"
#         - If no other remarks exist, just include the available information

#         Be intelligent in identifying segments even if the format varies.
#         Look for patterns like:
#         - CV with tonnage specifications (upto 2.5 Tn, 2.5 GVW, etc.)
#         - TW (Two Wheeler) related segments
#         - PVT CAR (Private Car) segments  
#         - BUS segments
#         - TAXI segments
#         - MISD (Miscellaneous) segments

#         If you cannot find specific information for any field, use "N/A" or reasonable defaults.
#         Note : If Policy is not defined or told , then consider it as COMP/TP
        
#         Text to analyze:
#         {extracted_text}
        
#         Return ONLY a valid JSON array, no other text.
        
#         """
        
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "Extract policy data as JSON array. Always rename 'Payout' → 'Payin'. Normalize all Payin values to percentage format. Use intelligence to identify segments accurately. Extract ALL additional information for remarks including vehicle makes, age information, and transaction type ONLY when explicitly found in text. Do NOT infer or assume transaction type if not clearly stated. Return ONLY valid JSON array."},
#                 {"role": "user", "content": parse_prompt}
#             ],
#             temperature=0.1
#         )
        
#         parsed_json = response.choices[0].message.content.strip()
#         logger.info(f"Raw OpenAI response: {parsed_json[:500]}...")
        
#         # Clean the JSON response
#         cleaned_json = clean_json_response(parsed_json)
#         logger.info(f"Cleaned JSON: {cleaned_json[:500]}...")
        
#         try:
#             policy_data = json.loads(cleaned_json)
#             policy_data = ensure_list_format(policy_data)
#         except json.JSONDecodeError as e:
#             logger.error(f"JSON decode error: {str(e)}")
#             logger.error(f"Problematic JSON: {cleaned_json}")
            
#             # Fallback: create a basic structure
#             policy_data = [{
#                 "Segment": "Unknown",
#                 "Location": "N/A",
#                 "Policy Type": "N/A", 
#                 "Payin": "0%",
#                 "Doable District": "N/A",
#                 "Remarks": f"Error parsing data: {str(e)}"
#             }]

#         st.success(f"✅ Successfully parsed {len(policy_data)} policy records")

#         # Pre-classify Payin values
#         for record in policy_data:
#             try:
#                 payin_val, payin_cat = classify_payin(record.get('Payin', '0%'))
#                 record['Payin_Value'] = payin_val
#                 record['Payin_Category'] = payin_cat
#             except Exception as e:
#                 logger.warning(f"Error classifying payin for record: {record}, error: {str(e)}")
#                 record['Payin_Value'] = 0.0
#                 record['Payin_Category'] = "Payin Below 20%"

#         # Apply formulas directly using Python logic
#         st.info("🧮 Applying formulas and calculating payouts...")
        
#         calculated_data = apply_formula_directly(policy_data, company_name)

#         st.success(f"✅ Successfully calculated {len(calculated_data)} records with payouts")

#         # Create Excel file
#         st.info("📊 Creating Excel file...")
        
#         df_calc = pd.DataFrame(calculated_data)
#         output = BytesIO()
#         with pd.ExcelWriter(output, engine='openpyxl') as writer:
#             df_calc.to_excel(writer, sheet_name='Policy Data', startrow=2, index=False)
#             worksheet = writer.sheets['Policy Data']

#             headers = list(df_calc.columns)
#             for col_num, value in enumerate(headers, 1):
#                 worksheet.cell(row=3, column=col_num, value=value)
#                 worksheet.cell(row=3, column=col_num).font = worksheet.cell(row=3, column=col_num).font.copy(bold=True)

#             company_cell = worksheet.cell(row=1, column=1, value=company_name)
#             worksheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
#             company_cell.font = company_cell.font.copy(bold=True, size=14)
#             company_cell.alignment = company_cell.alignment.copy(horizontal='center')

#             title_cell = worksheet.cell(row=2, column=1, value='Policy Data with Payin and Calculated Payouts')
#             worksheet.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(headers))
#             title_cell.font = title_cell.font.copy(bold=True, size=12)
#             title_cell.alignment = title_cell.alignment.copy(horizontal='center')

#         output.seek(0)
#         excel_data = output.read()

#         return {
#             "extracted_text": extracted_text,
#             "parsed_data": policy_data,
#             "calculated_data": calculated_data,
#             "excel_data": excel_data,
#             "df_calc": df_calc
#         }

#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}", exc_info=True)
#         raise Exception(f"An error occurred: {str(e)}")


# # Streamlit App
# def main():
#     st.set_page_config(
#         page_title="Insurance Policy Processing", 
#         page_icon="📋", 
#         layout="wide"
#     )
    
#     st.title("🏢 Insurance Policy Processing System")
#     st.markdown("---")
    
#     # Sidebar for inputs
#     with st.sidebar:
#         st.header("📁 File Upload")
        
#         # Company name input
#         company_name = st.text_input(
#             "Company Name", 
#             value="Unknown Company",
#             help="Enter the insurance company name"
#         )
        
#         # Policy file upload (only need policy file now)
#         policy_file = st.file_uploader(
#             "📄 Upload Policy File",
#             type=['pdf', 'png', 'jpg', 'jpeg', 'txt', 'csv', 'xlsx', 'xls'],
#             help="Upload your insurance policy document"
#         )
        
#         # Show formula data info
#         st.info("📊 Formula rules are embedded in the system and will be automatically applied.")
        
#         # Process button
#         process_button = st.button(
#             "🚀 Process Policy File", 
#             type="primary",
#             disabled=not policy_file
#         )
    
#     # Main content area
#     if not policy_file:
#         st.info("👆 Please upload a policy file to begin processing.")
#         st.markdown("""
#         ### 📋 Instructions:
#         1. **Company Name**: Enter the insurance company name
#         2. **Policy File**: Upload the document containing policy data (PDF, Image, Excel, CSV, etc.)
#         3. **Process**: Click the process button to extract data and calculate payouts
        
#         ### 🎯 Features:
#         - **Multi-format Support**: PDF, Images, Excel, CSV, Text files
#         - **AI-Powered Extraction**: Uses GPT-4 for intelligent text extraction
#         - **Enhanced Remarks Extraction**: Automatically detects and extracts:
#           - Vehicle make information (Tata, Maruti, etc.)
#           - Age information (>5 years, etc.)
#           - Transaction type (New/Old/Renewal)
#         - **Smart Formula Application**: Uses embedded formula rules for accurate calculations
#         - **Excel Export**: Download processed data as formatted Excel file
        
#         ### 📊 Formula Rules:
#         The system uses pre-configured formula rules for different LOBs:
#         - **TW**: Two Wheeler segments with company-specific rules
#         - **PVT CAR**: Private Car segments (COMP+SAOD, TP)
#         - **CV**: Commercial Vehicle segments
#         - **BUS**: School Bus and Staff Bus segments
#         - **TAXI**: Taxi segments with payin-based rules
#         - **MISD**: Miscellaneous segments including tractors
#         """)
#         return
    
#     if process_button:
#         try:
#             # Read file contents
#             policy_file_bytes = policy_file.read()
            
#             # Process files
#             with st.spinner("Processing policy file... This may take a few moments."):
#                 results = process_files(
#                     policy_file_bytes, policy_file.name, policy_file.type,
#                     company_name
#                 )
            
#             st.success("🎉 Processing completed successfully!")
            
#             # Display results in tabs
#             tab1, tab2, tab3, tab4, tab5 = st.tabs([
#                 "📊 Final Results", 
#                 "📝 Extracted Text", 
#                 "🧾 Parsed Data", 
#                 "🧮 Calculated Data",
#                 "📥 Download"
#             ])
            
#             with tab1:
#                 st.subheader("📊 Final Processed Data")
#                 st.dataframe(results["df_calc"], use_container_width=True)
                
#                 # Summary statistics
#                 col1, col2, col3, col4 = st.columns(4)
#                 with col1:
#                     st.metric("Total Records", len(results["calculated_data"]))
#                 with col2:
#                     avg_payin = sum([r.get('Payin_Value', 0) for r in results["calculated_data"]]) / len(results["calculated_data"])
#                     st.metric("Avg Payin", f"{avg_payin:.1f}%")
#                 with col3:
#                     segments = set([r.get('Segment', 'N/A') for r in results["calculated_data"]])
#                     st.metric("Unique Segments", len(segments))
#                 with col4:
#                     st.metric("Company", company_name)
                
#                 # Display formula rules used
#                 st.subheader("🔧 Formula Rules Applied")
#                 formula_summary = {}
#                 for record in results["calculated_data"]:
#                     formula = record.get('Formula Used', 'Unknown')
#                     if formula not in formula_summary:
#                         formula_summary[formula] = 0
#                     formula_summary[formula] += 1
                
#                 for formula, count in formula_summary.items():
#                     st.write(f"• **{formula}**: Applied to {count} record(s)")
            
#             with tab2:
#                 st.subheader("📝 Extracted Text from Policy File")
#                 st.text_area(
#                     "Policy Text", 
#                     results["extracted_text"], 
#                     height=400,
#                     key="policy_text"
#                 )
                
#                 st.subheader("📊 Embedded Formula Rules")
#                 st.write("The following formula rules are embedded in the system:")
#                 df_formula = pd.DataFrame(FORMULA_DATA)
#                 st.dataframe(df_formula, use_container_width=True)
            
#             with tab3:
#                 st.subheader("🧾 Parsed Policy Data")
#                 st.json(results["parsed_data"])
            
#             with tab4:
#                 st.subheader("🧮 Calculated Data with Formulas")
#                 st.json(results["calculated_data"])
                
#                 # Show detailed rule explanations
#                 st.subheader("🔍 Rule Explanations")
#                 for i, record in enumerate(results["calculated_data"]):
#                     with st.expander(f"Record {i+1}: {record.get('Segment', 'Unknown')}"):
#                         st.write(f"**Payin**: {record.get('Payin', 'N/A')}")
#                         st.write(f"**Calculated Payout**: {record.get('Calculated Payout', 'N/A')}")
#                         st.write(f"**Formula Used**: {record.get('Formula Used', 'N/A')}")
#                         st.write(f"**Rule Explanation**: {record.get('Rule Explanation', 'N/A')}")
            
#             with tab5:
#                 st.subheader("📥 Download Results")
                
#                 # Excel download
#                 st.download_button(
#                     label="📊 Download Excel File",
#                     data=results["excel_data"],
#                     file_name=f"{company_name}_processed_policies.xlsx",
#                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#                 )
                
#                 # JSON download
#                 json_data = json.dumps(results["calculated_data"], indent=2)
#                 st.download_button(
#                     label="📄 Download JSON Data",
#                     data=json_data,
#                     file_name=f"{company_name}_processed_data.json",
#                     mime="application/json"
#                 )
                
#                 # CSV download
#                 csv_data = results["df_calc"].to_csv(index=False)
#                 st.download_button(
#                     label="📋 Download CSV File",
#                     data=csv_data,
#                     file_name=f"{company_name}_processed_policies.csv",
#                     mime="text/csv"
#                 )
                
#                 st.info("💡 The Excel file contains formatted data with company header and calculated payouts.")
                
#         except Exception as e:
#             st.error(f"❌ Error processing files: {str(e)}")
#             st.exception(e)


# if __name__ == "__main__":
#     main()




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


def extract_text_from_file(file_bytes: bytes, filename: str, content_type: str) -> str:
    """Extract text from uploaded file"""
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
        return response.choices[0].message.content.strip()

    # Enhanced PDF extraction
    if file_extension == 'pdf':
        # First try to extract text directly from PDF
        pdf_text = extract_text_from_pdf_file(file_bytes)
        
        if pdf_text and len(pdf_text.strip()) > 50:
            # If we got good text extraction, use it
            logger.info("Using direct PDF text extraction")
            return pdf_text
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
            return response.choices[0].message.content.strip()

    # Text files
    if file_extension == 'txt':
        return file_bytes.decode('utf-8', errors='ignore')

    # CSV files
    if file_extension == 'csv':
        df = pd.read_csv(BytesIO(file_bytes), encoding='utf-8', errors='ignore')
        return df.to_string()

    # Excel files
    if file_extension in ['xlsx', 'xls']:
        all_sheets = pd.read_excel(BytesIO(file_bytes), sheet_name=None)
        dfs = []
        for sheet_name, df_sheet in all_sheets.items():
            df_sheet["Source_Sheet"] = sheet_name
            dfs.append(df_sheet)
        df = pd.concat(dfs, ignore_index=True, join="outer")
        return df.to_string(index=False)

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


def process_files(policy_file_bytes, policy_filename, policy_content_type, company_name):
    """Main processing function"""
    try:
        st.info("🔍 Extracting text from policy file...")
        
        # Extract text with enhanced intelligence
        extracted_text = extract_text_from_file(policy_file_bytes, policy_filename, policy_content_type)

        logger.info(f"Extracted text length: {len(extracted_text)}")

        st.success(f"✅ Text extracted successfully! Policy: {len(extracted_text)} chars")

        # Parse policy data with enhanced segment identification
        st.info("🧠 Parsing policy data with AI...")
        
        parse_prompt = f"""
        Analyze the following text, which contains insurance policy grid details, likely in a tabular format. Use your intelligence to identify and extract the data accurately, treating it as a table with rows for regions/states/circles and columns for segments, sub-segments, vehicle ages, policy types, and payin values.

Company Name: {company_name}

IMPORTANT: Always return a valid JSON array (list) of records, even if there's only one record. Create one JSON record per payin value (each numerical cell in the table), associating it with the corresponding row headers (region, state, circle) and column headers (segment, sub-segment, age, policy type). Generalize segment identification: Segments, including tonnage, makes, and other details, are typically embedded in the column headers or sub-headers (e.g., "GCV 4W-Upto 2.5T GCV (except EV)", "12t to 20T (TATA & Ashok leyland)").

Extract into JSON records with these exact fields:
- "Segment": Identify the main segment from column headers, e.g., "GCV 4W-Upto 2.5T GCV (except EV)", "12t to 20T (TATA & Ashok leyland)", "20t to 40T (TATA & Ashok leyland)". Include tonnage/weight/make details exactly as mentioned (e.g., "Upto 2.5T", "12t to 20T"). For sub-segments like makes (e.g., "Upto 2.0 T-All makes & 2.0-2.5T Tata makes", "2.0-2.5T-other than Tata makes"), append them to the segment or include in remarks if separate. Recognize this as CV (Commercial Vehicle) or GCV (Goods Carrying Vehicle) LOB, and prefix with "CV - " or "GCV - " if appropriate based on context.
- "Location": Combine region, state, and circle, e.g., "Region: North 2; State: UTTAR PRADESH (Rest); Circle: Lucknow". Use "N/A" for missing parts.
- "Policy Type": From sub-column headers, e.g., "New-Comp (Net)", "Comp (Net)", "SATP (Net)". If not specified, default to "COMP/TP" based on context.
- "Payin": The numerical value converted to percentage format, e.g., "56" → "56%", "45.5" → "45.5%". Ensure it's a string with "%".
- "Doable District": Extract any RTO lists or district information from notes, e.g., "KA22/KA23/.../KA71" or "UP13/UP20/.../UP94". If specific to a location or segment, associate accordingly; otherwise, "N/A".
- "Remarks": Include vehicle age from headers (e.g., "Age: 1-5 Years", "Age: above 5 years"), vehicle makes (e.g., "Vehicle Makes: TATA, Ashok Leyland"), transaction types if mentioned (e.g., "New", "Comp" implying comprehensive), validity (e.g., "Effective from 21st Sep25 to 31st Oct25"), and any other notes (e.g., "Above mentioned Club applicable from 21st Sep 25 to 31st Oct25", "For doable & declined RTOs & model list-Same as previous month guidelines", "Note-For less than 25 lac Grid would be lower by 2% & for less than 1 L only IRDA applicable", "Festival Dhamaka Grid"). Combine with semicolons, e.g., "Vehicle Makes: TATA, Ashok Leyland; Age: 1-5 Years; Validity: from 21st Sep25 to 31st Oct25; Note-For less than 25 lac Grid would be lower by 2% & for less than 1 L only IRDA applicable".

ENHANCED EXTRACTION FOR GRIDS:
- Treat the text as a table: Rows start with Region/State/Circle, followed by values.
- Columns represent segments (tonnage/makes), ages, and policy types (New-Comp/Comp/SATP).
- For each numerical value, create a record linking it to the row's location and the full column path (segment + sub + age + policy type).
- Preserve exact terminology: e.g., "SATP (Net)" instead of just "TP".
- If the grid is a main grid like "Festival Dhamaka Grid" or addendum, note it in remarks.
- For CV/GCV: Always include weight/tonnage/make in segment, e.g., "GCV - Upto 2.5T (except EV)".
- Look for patterns: Segments in top headers, sub-makes/ages as sub-headers, policy types as innermost.
- Extract global notes (e.g., RTO lists, applicability) and apply to relevant records or duplicate if needed.
- Generalize for variations: Column headers may contain the full segment description, so parse them intelligently to build the "Segment" field.

Be intelligent in identifying even if format varies (e.g., from OCR text of image/PDF/Excel/CSV).
If you cannot find specific information, use "N/A".
        
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

        st.success(f"✅ Successfully parsed {len(policy_data)} policy records")

        # Pre-classify Payin values
        for record in policy_data:
            try:
                payin_val, payin_cat = classify_payin(record.get('Payin', '0%'))
                record['Payin_Value'] = payin_val
                record['Payin_Category'] = payin_cat
            except Exception as e:
                logger.warning(f"Error classifying payin for record: {record}, error: {str(e)}")
                record['Payin_Value'] = 0.0
                record['Payin_Category'] = "Payin Below 20%"

        # Apply formulas directly using Python logic
        st.info("🧮 Applying formulas and calculating payouts...")
        
        calculated_data = apply_formula_directly(policy_data, company_name)

        st.success(f"✅ Successfully calculated {len(calculated_data)} records with payouts")

        # Create Excel file
        st.info("📊 Creating Excel file...")
        
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
        excel_data = output.read()

        return {
            "extracted_text": extracted_text,
            "parsed_data": policy_data,
            "calculated_data": calculated_data,
            "excel_data": excel_data,
            "df_calc": df_calc
        }

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
        
        # Policy file upload (only need policy file now)
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
        - **Excel Export**: Download processed data as formatted Excel file
        
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
                st.dataframe(results["df_calc"], use_container_width=True)
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", len(results["calculated_data"]))
                with col2:
                    avg_payin = sum([r.get('Payin_Value', 0) for r in results["calculated_data"]]) / len(results["calculated_data"])
                    st.metric("Avg Payin", f"{avg_payin:.1f}%")
                with col3:
                    segments = set([r.get('Segment', 'N/A') for r in results["calculated_data"]])
                    st.metric("Unique Segments", len(segments))
                with col4:
                    st.metric("Company", company_name)
                
                # Display formula rules used
                st.subheader("🔧 Formula Rules Applied")
                formula_summary = {}
                for record in results["calculated_data"]:
                    formula = record.get('Formula Used', 'Unknown')
                    if formula not in formula_summary:
                        formula_summary[formula] = 0
                    formula_summary[formula] += 1
                
                for formula, count in formula_summary.items():
                    st.write(f"• **{formula}**: Applied to {count} record(s)")
            
            with tab2:
                st.subheader("📝 Extracted Text from Policy File")
                st.text_area(
                    "Policy Text", 
                    results["extracted_text"], 
                    height=400,
                    key="policy_text"
                )
                
                st.subheader("📊 Embedded Formula Rules")
                st.write("The following formula rules are embedded in the system:")
                df_formula = pd.DataFrame(FORMULA_DATA)
                st.dataframe(df_formula, use_container_width=True)
            
            with tab3:
                st.subheader("🧾 Parsed Policy Data")
                st.json(results["parsed_data"])
            
            with tab4:
                st.subheader("🧮 Calculated Data with Formulas")
                st.json(results["calculated_data"])
                
                # Show detailed rule explanations
                st.subheader("🔍 Rule Explanations")
                for i, record in enumerate(results["calculated_data"]):
                    with st.expander(f"Record {i+1}: {record.get('Segment', 'Unknown')}"):
                        st.write(f"**Payin**: {record.get('Payin', 'N/A')}")
                        st.write(f"**Calculated Payout**: {record.get('Calculated Payout', 'N/A')}")
                        st.write(f"**Formula Used**: {record.get('Formula Used', 'N/A')}")
                        st.write(f"**Rule Explanation**: {record.get('Rule Explanation', 'N/A')}")
            
            with tab5:
                st.subheader("📥 Download Results")
                
                # Excel download
                st.download_button(
                    label="📊 Download Excel File",
                    data=results["excel_data"],
                    file_name=f"{company_name}_processed_policies.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                # JSON download
                json_data = json.dumps(results["calculated_data"], indent=2)
                st.download_button(
                    label="📄 Download JSON Data",
                    data=json_data,
                    file_name=f"{company_name}_processed_data.json",
                    mime="application/json"
                )
                
                # CSV download
                csv_data = results["df_calc"].to_csv(index=False)
                st.download_button(
                    label="📋 Download CSV File",
                    data=csv_data,
                    file_name=f"{company_name}_processed_policies.csv",
                    mime="text/csv"
                )
                
                st.info("💡 The Excel file contains formatted data with company header and calculated payouts.")
                
        except Exception as e:
            st.error(f"❌ Error processing files: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()

