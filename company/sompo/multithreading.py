# # import streamlit as st
# # import pandas as pd
# # from io import BytesIO
# # import base64
# # import json
# # import os
# # from dotenv import load_dotenv
# # import logging
# # import re
# # from typing import Dict, List, Tuple
# # import time

# # # Check required packages
# # try:
# #     from openai import OpenAI
# # except ImportError:
# #     st.error("OpenAI package not found. Please install it using 'pip install openai'")
# #     st.stop()

# # try:
# #     import PyPDF2
# # except ImportError:
# #     st.warning("PyPDF2 not found. PDF text extraction will use OpenAI vision only.")
# #     PyPDF2 = None

# # try:
# #     import pdfplumber
# # except ImportError:
# #     st.warning("pdfplumber not found. PDF text extraction will use PyPDF2 or OpenAI vision only.")
# #     pdfplumber = None

# # # Load environment variables
# # load_dotenv()

# # # Configure logging
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)

# # # Load OpenAI API key
# # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# # if not OPENAI_API_KEY:
# #     st.error("OPENAI_API_KEY environment variable not set.")
# #     st.stop()

# # # Initialize OpenAI client
# # try:
# #     client = OpenAI(api_key=OPENAI_API_KEY)
# # except Exception as e:
# #     st.error(f"Failed to initialize OpenAI client: {str(e)}")
# #     st.stop()

# # # CONFIGURATION
# # CHUNK_SIZE = 25  # Process 25 rows at a time (configurable)
# # MAX_RETRIES = 3  # Retry failed chunks
# # RETRY_DELAY = 2  # Seconds between retries

# # # Embedded Formula Data
# # FORMULA_DATA = [
# #     {"LOB": "TW", "SEGMENT": "1+5", "INSURER": "All Companies", "PO": "90% of Payin", "REMARKS": "NIL"},
# #     {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "INSURER": "All Companies", "PO": "90% of Payin", "REMARKS": "NIL"},
# #     {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "INSURER": "DIGIT", "PO": "-2%", "REMARKS": "Payin Below 20%"},
# #     {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "INSURER": "DIGIT", "PO": "-3%", "REMARKS": "Payin 21% to 30%"},
# #     {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "INSURER": "DIGIT", "PO": "-4%", "REMARKS": "Payin 31% to 50%"},
# #     {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "INSURER": "DIGIT", "PO": "-5%", "REMARKS": "Payin Above 50%"},
# #     {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Bajaj, Digit, ICICI", "PO": "-2%", "REMARKS": "Payin Below 20%"},
# #     {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Bajaj, Digit, ICICI", "PO": "-3%", "REMARKS": "Payin Above 20%"},
# #     {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Rest of Companies", "PO": "-2%", "REMARKS": "Payin Below 20%"},
# #     {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Rest of Companies", "PO": "-3%", "REMARKS": "Payin 21% to 30%"},
# #     {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Rest of Companies", "PO": "-4%", "REMARKS": "Payin 31% to 50%"},
# #     {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Rest of Companies", "PO": "-5%", "REMARKS": "Payin Above 50%"},
# #     {"LOB": "PVT CAR", "SEGMENT": "PVT CAR COMP + SAOD", "INSURER": "All Companies", "PO": "90% of Payin", "REMARKS": "All Fuel"},
# #     {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "Bajaj, Digit, SBI", "PO": "-2%", "REMARKS": "Payin Below 20%"},
# #     {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "Bajaj, Digit, SBI", "PO": "-3%", "REMARKS": "Payin Above 20%"},
# #     {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "Rest of Companies", "PO": "90% of Payin", "REMARKS": "Zuno - 21"},
# #     {"LOB": "CV", "SEGMENT": "Upto 2.5 GVW", "INSURER": "Reliance, SBI, Tata", "PO": "-2%", "REMARKS": "NIL"},
# #     {"LOB": "CV", "SEGMENT": "All GVW & PCV 3W, GCV 3W", "INSURER": "Rest of Companies", "PO": "-2%", "REMARKS": "Payin Below 20%"},
# #     {"LOB": "CV", "SEGMENT": "All GVW & PCV 3W, GCV 3W", "INSURER": "Rest of Companies", "PO": "-3%", "REMARKS": "Payin 21% to 30%"},
# #     {"LOB": "CV", "SEGMENT": "All GVW & PCV 3W, GCV 3W", "INSURER": "Rest of Companies", "PO": "-4%", "REMARKS": "Payin 31% to 50%"},
# #     {"LOB": "CV", "SEGMENT": "All GVW & PCV 3W, GCV 3W", "INSURER": "Rest of Companies", "PO": "-5%", "REMARKS": "Payin Above 50%"},
# #     {"LOB": "BUS", "SEGMENT": "SCHOOL BUS", "INSURER": "TATA, Reliance, Digit, ICICI", "PO": "Less 2% of Payin", "REMARKS": "NIL"},
# #     {"LOB": "BUS", "SEGMENT": "SCHOOL BUS", "INSURER": "Rest of Companies", "PO": "88% of Payin", "REMARKS": "NIL"},
# #     {"LOB": "BUS", "SEGMENT": "STAFF BUS", "INSURER": "All Companies", "PO": "88% of Payin", "REMARKS": "NIL"},
# #     {"LOB": "TAXI", "SEGMENT": "TAXI", "INSURER": "All Companies", "PO": "-2%", "REMARKS": "Payin Below 20%"},
# #     {"LOB": "TAXI", "SEGMENT": "TAXI", "INSURER": "All Companies", "PO": "-3%", "REMARKS": "Payin 21% to 30%"},
# #     {"LOB": "TAXI", "SEGMENT": "TAXI", "INSURER": "All Companies", "PO": "-4%", "REMARKS": "Payin 31% to 50%"},
# #     {"LOB": "TAXI", "SEGMENT": "TAXI", "INSURER": "All Companies", "PO": "-5%", "REMARKS": "Payin Above 50%"},
# #     {"LOB": "MISD", "SEGMENT": "Misd, Tractor", "INSURER": "All Companies", "PO": "88% of Payin", "REMARKS": "NIL"}
# # ]


# # def extract_text_from_pdf_file(pdf_bytes: bytes) -> str:
# #     """Extract text from PDF using multiple methods"""
# #     extracted_text = ""
    
# #     if pdfplumber:
# #         try:
# #             with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
# #                 for page in pdf.pages:
# #                     page_text = page.extract_text()
# #                     if page_text:
# #                         extracted_text += page_text + "\n"
# #             if extracted_text.strip():
# #                 logger.info("PDF text extracted using pdfplumber")
# #                 return extracted_text.strip()
# #         except Exception as e:
# #             logger.warning(f"pdfplumber extraction failed: {str(e)}")
    
# #     if PyPDF2:
# #         try:
# #             pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
# #             for page in pdf_reader.pages:
# #                 page_text = page.extract_text()
# #                 if page_text:
# #                     extracted_text += page_text + "\n"
# #             if extracted_text.strip():
# #                 logger.info("PDF text extracted using PyPDF2")
# #                 return extracted_text.strip()
# #         except Exception as e:
# #             logger.warning(f"PyPDF2 extraction failed: {str(e)}")
    
# #     logger.warning("All PDF text extraction methods failed")
# #     return ""


# # def extract_text_from_file(file_bytes: bytes, filename: str, content_type: str) -> dict:
# #     """Extract text from uploaded file"""
# #     file_extension = filename.split('.')[-1].lower() if '.' in filename else ''
# #     file_type = content_type if content_type else file_extension
   
# #     # Image-based extraction
# #     image_extensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']
# #     if file_extension in image_extensions or file_type.startswith('image/'):
# #         image_base64 = base64.b64encode(file_bytes).decode('utf-8')
        
# #         prompt = """Extract all insurance policy text accurately from this image using OCR.
# #         Pay special attention to identifying segments, company names, policy types, 
# #         Payin/Payout percentages, and numerical values. Extract all text exactly as it appears."""
            
# #         response = client.chat.completions.create(
# #             model="gpt-4o",
# #             messages=[{
# #                 "role": "user",
# #                 "content": [
# #                     {"type": "text", "text": prompt},
# #                     {"type": "image_url", "image_url": {"url": f"data:image/{file_extension};base64,{image_base64}"}}
# #                 ]
# #             }],
# #             temperature=0.1
# #         )
# #         return {"Sheet1": response.choices[0].message.content.strip()}

# #     # PDF extraction
# #     if file_extension == 'pdf':
# #         pdf_text = extract_text_from_pdf_file(file_bytes)
        
# #         if pdf_text and len(pdf_text.strip()) > 50:
# #             return {"Sheet1": pdf_text}
# #         else:
# #             pdf_base64 = base64.b64encode(file_bytes).decode("utf-8")
            
# #             prompt = """Extract insurance policy details from this PDF.
# #             Focus on segments, company names, policy types, and payout percentages."""
                
# #             response = client.chat.completions.create(
# #                 model="gpt-4o",
# #                 messages=[{
# #                     "role": "user",
# #                     "content": [
# #                         {"type": "text", "text": prompt},
# #                         {"type": "image_url", "image_url": {"url": f"data:application/pdf;base64,{pdf_base64}"}}
# #                     ]
# #                 }],
# #                 temperature=0.1
# #             )
# #             return {"Sheet1": response.choices[0].message.content.strip()}

# #     # Text files
# #     if file_extension == 'txt':
# #         return {"Sheet1": file_bytes.decode('utf-8', errors='ignore')}

# #     # CSV files
# #     if file_extension == 'csv':
# #         df = pd.read_csv(BytesIO(file_bytes), encoding='utf-8', errors='ignore')
# #         return {"Sheet1": df.to_string()}

# #     # Excel files - RETURN RAW DATAFRAMES
# #     if file_extension in ['xlsx', 'xls']:
# #         all_sheets = pd.read_excel(BytesIO(file_bytes), sheet_name=None)
# #         return all_sheets  # Return as DataFrames, not strings

# #     raise ValueError(f"Unsupported file type for {filename}")


# # def chunk_dataframe(df: pd.DataFrame, chunk_size: int = CHUNK_SIZE) -> List[pd.DataFrame]:
# #     """Split DataFrame into chunks"""
# #     chunks = []
# #     for i in range(0, len(df), chunk_size):
# #         chunks.append(df.iloc[i:i + chunk_size])
# #     return chunks


# # def clean_json_response(response_text: str) -> str:
# #     """Clean and extract JSON from OpenAI response"""
# #     cleaned = re.sub(r'```json\s*', '', response_text)
# #     cleaned = re.sub(r'```\s*$', '', cleaned)
    
# #     json_start = -1
# #     for i, char in enumerate(cleaned):
# #         if char in '[{':
# #             json_start = i
# #             break
    
# #     if json_start != -1:
# #         cleaned = cleaned[json_start:]
    
# #     json_end = -1
# #     for i in range(len(cleaned) - 1, -1, -1):
# #         if cleaned[i] in ']}':
# #             json_end = i + 1
# #             break
    
# #     if json_end != -1:
# #         cleaned = cleaned[:json_end]
    
# #     return cleaned.strip()


# # def ensure_list_format(data) -> list:
# #     """Ensure data is in list format"""
# #     if isinstance(data, list):
# #         return data
# #     elif isinstance(data, dict):
# #         return [data]
# #     else:
# #         raise ValueError(f"Expected list or dict, got {type(data)}")


# # def classify_payin(payin_str):
# #     """Convert Payin string to float and classify its range"""
# #     try:
# #         payin_clean = str(payin_str).replace('%', '').replace(' ', '')
# #         payin_value = float(payin_clean)
        
# #         if payin_value <= 20:
# #             category = "Payin Below 20%"
# #         elif 21 <= payin_value <= 30:
# #             category = "Payin 21% to 30%"
# #         elif 31 <= payin_value <= 50:
# #             category = "Payin 31% to 50%"
# #         else:
# #             category = "Payin Above 50%"
# #         return payin_value, category
# #     except (ValueError, TypeError):
# #         logger.warning(f"Could not parse payin value: {payin_str}")
# #         return 0.0, "Payin Below 20%"


# # def parse_chunk_with_retry(chunk_text: str, sheet_name: str, company_name: str, 
# #                            chunk_idx: int, total_chunks: int) -> List[Dict]:
# #     """Parse a chunk of data with retry logic"""
    
# #     parse_prompt = f"""
# #     Analyze this data chunk ({chunk_idx + 1} of {total_chunks}) from sheet {sheet_name}.
# #     Company: {company_name}

# #     Extract JSON records with these fields:
# #     - "Segment": LOB (TW, PVT CAR, CV, BUS, TAXI, MISD) + policy type (TP, COMP, SAOD, Package)
# #       For CV/PCV: preserve GVW/tonnage (e.g., "CV 0 T-3.5T", "CV upto 2.5 Tn")
# #       Priority: PCV/Commercial Vehicle → CV segment
# #     - "Location": region + state (e.g., "East: CG")
# #     - "Policy Type": New/Roll over/SATP/Package (use COMP/TP if not specified)
# #     - "Payin": normalize to percentage (0.625 → 62.5%, "-" → 0%)
# #     - "Doable District": RTO codes (e.g., "UP21, DL-NCR")
# #     - "Remarks": vehicle makes, age, transaction type, terms, declined RTOs

# #     CRITICAL: Process EVERY row in this chunk. Do not skip any rows.
# #     Return ONLY valid JSON array with one record per row.

# #     Data:
# #     {chunk_text}
# #     """
    
# #     for attempt in range(MAX_RETRIES):
# #         try:
# #             response = client.chat.completions.create(
# #                 model="gpt-4o-mini",
# #                 messages=[
# #                     {"role": "system", "content": "Extract ALL rows as JSON array. Normalize Payin to %. NEVER skip rows."},
# #                     {"role": "user", "content": parse_prompt}
# #                 ],
# #                 temperature=0.1
# #             )
            
# #             parsed_json = response.choices[0].message.content.strip()
# #             cleaned_json = clean_json_response(parsed_json)
# #             policy_data = json.loads(cleaned_json)
# #             policy_data = ensure_list_format(policy_data)
            
# #             # Validate we got expected number of records
# #             expected_rows = chunk_text.count('\n') + 1  # Rough estimate
# #             if len(policy_data) < expected_rows * 0.5:  # Allow 50% tolerance
# #                 logger.warning(f"Chunk {chunk_idx}: Got {len(policy_data)} records, expected ~{expected_rows}")
            
# #             return policy_data
            
# #         except (json.JSONDecodeError, Exception) as e:
# #             logger.error(f"Attempt {attempt + 1} failed for chunk {chunk_idx}: {str(e)}")
# #             if attempt < MAX_RETRIES - 1:
# #                 time.sleep(RETRY_DELAY)
# #             else:
# #                 # Return error placeholder
# #                 return [{
# #                     "Segment": "Error",
# #                     "Location": "N/A",
# #                     "Policy Type": "N/A",
# #                     "Payin": "0%",
# #                     "Doable District": "N/A",
# #                     "Remarks": f"Failed to parse chunk {chunk_idx} after {MAX_RETRIES} attempts"
# #                 }]


# # def parse_sheet_in_chunks(df: pd.DataFrame, sheet_name: str, company_name: str) -> List[Dict]:
# #     """Parse large DataFrame in chunks"""
# #     st.info(f"📋 Sheet '{sheet_name}' has {len(df)} rows. Processing in chunks of {CHUNK_SIZE}...")
    
# #     chunks = chunk_dataframe(df, CHUNK_SIZE)
# #     all_policy_data = []
    
# #     progress_bar = st.progress(0)
# #     status_text = st.empty()
    
# #     for idx, chunk_df in enumerate(chunks):
# #         status_text.text(f"Processing chunk {idx + 1}/{len(chunks)}...")
        
# #         # Convert chunk to string representation
# #         chunk_text = chunk_df.to_string(index=False)
        
# #         # Parse chunk
# #         chunk_policy_data = parse_chunk_with_retry(
# #             chunk_text, sheet_name, company_name, idx, len(chunks)
# #         )
        
# #         all_policy_data.extend(chunk_policy_data)
        
# #         # Update progress
# #         progress = (idx + 1) / len(chunks)
# #         progress_bar.progress(progress)
        
# #         # Small delay to avoid rate limits
# #         time.sleep(0.5)
    
# #     status_text.text(f"✅ Completed processing {len(chunks)} chunks → {len(all_policy_data)} records")
# #     progress_bar.empty()
    
# #     return all_policy_data


# # def apply_formula_directly(policy_data, company_name):
# #     """Apply formula rules"""
# #     calculated_data = []
    
# #     for record in policy_data:
# #         segment = record.get('Segment', '').upper()
# #         payin_value = record.get('Payin_Value', 0)
# #         payin_category = record.get('Payin_Category', '')
        
# #         if payin_value == 0:
# #             result_record = record.copy()
# #             result_record['Calculated Payout'] = "0%"
# #             result_record['Formula Used'] = "Payin is 0"
# #             result_record['Rule Explanation'] = f"Payin is 0; set Calculated Payout to 0%"
# #             calculated_data.append(result_record)
# #             continue
        
# #         # Determine LOB
# #         lob = ""
# #         if any(kw in segment for kw in ['TW', '2W', 'TWO WHEELER']):
# #             lob = "TW"
# #         elif any(kw in segment for kw in ['PVT CAR', 'PRIVATE CAR', 'CAR']):
# #             lob = "PVT CAR"
# #         elif any(kw in segment for kw in ['CV', 'COMMERCIAL', 'LCV', 'GVW', 'TN']):
# #             lob = "CV"
# #         elif 'BUS' in segment:
# #             lob = "BUS"
# #         elif 'TAXI' in segment:
# #             lob = "TAXI"
# #         elif any(kw in segment for kw in ['MISD', 'TRACTOR', 'MISC']):
# #             lob = "MISD"
# #         else:
# #             lob = "UNKNOWN"
        
# #         # Find matching formula rule
# #         matched_rule = None
# #         company_normalized = company_name.upper().replace('GENERAL', '').replace('INSURANCE', '').strip()
        
# #         for rule in FORMULA_DATA:
# #             if rule["LOB"] != lob:
# #                 continue
            
# #             rule_segment = rule["SEGMENT"].upper()
# #             segment_match = False
            
# #             # Segment matching logic
# #             if lob == "CV":
# #                 if "UPTO 2.5" in rule_segment:
# #                     if any(kw in segment for kw in ["UPTO 2.5", "2.5 TN", "2.5 GVW"]):
# #                         segment_match = True
# #                 elif "ALL GVW" in rule_segment:
# #                     if not any(kw in segment for kw in ["UPTO 2.5", "2.5 TN", "2.5"]):
# #                         segment_match = True
# #             elif lob == "BUS":
# #                 if ("SCHOOL" in rule_segment and "SCHOOL" in segment) or \
# #                    ("STAFF" in rule_segment and "STAFF" in segment):
# #                     segment_match = True
# #             elif lob == "PVT CAR":
# #                 if "COMP" in rule_segment and any(kw in segment for kw in ["COMP", "COMPREHENSIVE"]):
# #                     segment_match = True
# #                 elif "TP" in rule_segment and "TP" in segment and "COMP" not in segment:
# #                     segment_match = True
# #             elif lob == "TW":
# #                 if ("1+5" in rule_segment and "1+5" in segment) or \
# #                    ("SAOD + COMP" in rule_segment and any(kw in segment for kw in ["SAOD", "COMP"])) or \
# #                    ("TP" in rule_segment and "TP" in segment):
# #                     segment_match = True
# #             else:
# #                 segment_match = True
            
# #             if not segment_match:
# #                 continue
            
# #             # Company matching
# #             insurers = [ins.strip().upper() for ins in rule["INSURER"].split(',')]
# #             company_match = False
            
# #             if "ALL COMPANIES" in insurers:
# #                 company_match = True
# #             elif "REST OF COMPANIES" in insurers:
# #                 is_in_specific_list = False
# #                 for other_rule in FORMULA_DATA:
# #                     if (other_rule["LOB"] == rule["LOB"] and 
# #                         other_rule["SEGMENT"] == rule["SEGMENT"] and
# #                         "REST OF COMPANIES" not in other_rule["INSURER"] and
# #                         "ALL COMPANIES" not in other_rule["INSURER"]):
# #                         other_insurers = [ins.strip().upper() for ins in other_rule["INSURER"].split(',')]
# #                         if any(ck in company_normalized for ck in other_insurers):
# #                             is_in_specific_list = True
# #                             break
# #                 if not is_in_specific_list:
# #                     company_match = True
# #             else:
# #                 for insurer in insurers:
# #                     if insurer in company_normalized or company_normalized in insurer:
# #                         company_match = True
# #                         break
            
# #             if not company_match:
# #                 continue
            
# #             # Remarks matching
# #             remarks = rule.get("REMARKS", "")
            
# #             if remarks == "NIL" or "NIL" in remarks.upper():
# #                 matched_rule = rule
# #                 break
# #             elif any(pk in remarks for pk in ["Payin Below", "Payin 21%", "Payin 31%", "Payin Above"]):
# #                 if payin_category in remarks:
# #                     matched_rule = rule
# #                     break
# #             else:
# #                 matched_rule = rule
# #                 break
        
# #         # Calculate payout
# #         if matched_rule:
# #             po_formula = matched_rule["PO"]
            
# #             if "90% of Payin" in po_formula:
# #                 calculated_payout = payin_value * 0.9
# #             elif "88% of Payin" in po_formula:
# #                 calculated_payout = payin_value * 0.88
# #             elif "Less 2% of Payin" in po_formula or "-2%" in po_formula:
# #                 calculated_payout = payin_value - 2
# #             elif "-3%" in po_formula:
# #                 calculated_payout = payin_value - 3
# #             elif "-4%" in po_formula:
# #                 calculated_payout = payin_value - 4
# #             elif "-5%" in po_formula:
# #                 calculated_payout = payin_value - 5
# #             else:
# #                 calculated_payout = payin_value
            
# #             calculated_payout = max(0, calculated_payout)
# #             formula_used = po_formula
# #             rule_explanation = f"Matched: LOB={lob}, Segment={matched_rule['SEGMENT']}, Company={matched_rule['INSURER']}"
# #         else:
# #             calculated_payout = payin_value
# #             formula_used = "No matching rule"
# #             rule_explanation = f"No rule matched for LOB={lob}, Segment={segment}"
        
# #         result_record = record.copy()
# #         result_record['Calculated Payout'] = f"{int(calculated_payout)}%"
# #         result_record['Formula Used'] = formula_used
# #         result_record['Rule Explanation'] = rule_explanation
        
# #         calculated_data.append(result_record)
    
# #     return calculated_data


# # def process_files(policy_file_bytes, policy_filename, policy_content_type, company_name):
# #     """Main processing function with chunking support"""
# #     try:
# #         st.info("🔍 Extracting data from policy file...")
        
# #         extracted_data = extract_text_from_file(policy_file_bytes, policy_filename, policy_content_type)
        
# #         results = {
# #             "extracted_text": {},
# #             "parsed_data": {},
# #             "calculated_data": {},
# #             "excel_data": None,
# #             "df_calc": {}
# #         }
        
# #         output = BytesIO()
# #         with pd.ExcelWriter(output, engine='openpyxl') as writer:
# #             for sheet_name, sheet_data in extracted_data.items():
# #                 st.success(f"✅ Processing sheet: {sheet_name}")
                
# #                 # Handle DataFrames vs strings
# #                 if isinstance(sheet_data, pd.DataFrame):
# #                     df = sheet_data
# #                     # Use chunked parsing for large DataFrames
# #                     if len(df) > CHUNK_SIZE:
# #                         policy_data = parse_sheet_in_chunks(df, sheet_name, company_name)
# #                     else:
# #                         # Small sheet - process normally
# #                         chunk_text = df.to_string(index=False)
# #                         policy_data = parse_chunk_with_retry(chunk_text, sheet_name, company_name, 0, 1)
# #                 else:
# #                     # String data (from PDF/image/text)
# #                     extracted_text = sheet_data
# #                     results["extracted_text"][sheet_name] = extracted_text
# #                     policy_data = parse_chunk_with_retry(extracted_text, sheet_name, company_name, 0, 1)
                
# #                 st.success(f"✅ Parsed {len(policy_data)} records from {sheet_name}")
                
# #                 # Classify Payin
# #                 for record in policy_data:
# #                     try:
# #                         payin_val, payin_cat = classify_payin(record.get('Payin', '0%'))
# #                         record['Payin_Value'] = payin_val
# #                         record['Payin_Category'] = payin_cat
# #                     except Exception as e:
# #                         logger.warning(f"Error classifying payin: {e}")
# #                         record['Payin_Value'] = 0.0
# #                         record['Payin_Category'] = "Payin Below 20%"
                
# #                 # Apply formulas
# #                 st.info(f"🧮 Calculating payouts for {sheet_name}...")
# #                 calculated_data = apply_formula_directly(policy_data, company_name)
                
# #                 st.success(f"✅ Calculated {len(calculated_data)} records for {sheet_name}")
                
# #                 # Create DataFrame
# #                 df_calc = pd.DataFrame(calculated_data)
                
# #                 # Write to Excel
# #                 df_calc.to_excel(writer, sheet_name=sheet_name, startrow=2, index=False)
# #                 worksheet = writer.sheets[sheet_name]
# #                 headers = list(df_calc.columns)
# #                 for col_num, value in enumerate(headers, 1):
# #                     cell = worksheet.cell(row=3, column=col_num, value=value)
# #                     cell.font = cell.font.copy(bold=True)
                
# #                 company_cell = worksheet.cell(row=1, column=1, value=company_name)
# #                 worksheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
# #                 company_cell.font = company_cell.font.copy(bold=True, size=14)
# #                 company_cell.alignment = company_cell.alignment.copy(horizontal='center')
                
# #                 title_cell = worksheet.cell(row=2, column=1, value=f'Policy Data - {sheet_name}')
# #                 worksheet.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(headers))
# #                 title_cell.font = title_cell.font.copy(bold=True, size=12)
# #                 title_cell.alignment = title_cell.alignment.copy(horizontal='center')
                
# #                 results["parsed_data"][sheet_name] = policy_data
# #                 results["calculated_data"][sheet_name] = calculated_data
# #                 results["df_calc"][sheet_name] = df_calc
        
# #         output.seek(0)
# #         results["excel_data"] = output.read()
        
# #         return results
    
# #     except Exception as e:
# #         logger.error(f"Unexpected error: {str(e)}", exc_info=True)
# #         raise Exception(f"An error occurred: {str(e)}")


# # def main():
# #     st.set_page_config(
# #         page_title="Insurance Policy Processor", 
# #         page_icon="📋", 
# #         layout="wide"
# #     )
    
# #     st.title("🏢 Insurance Policy Processing System (Chunked)")
# #     st.markdown("---")
    
# #     # Sidebar
# #     with st.sidebar:
# #         st.header("📁 File Upload")
        
# #         company_name = st.text_input(
# #             "Company Name", 
# #             value="Unknown Company",
# #             help="Enter the insurance company name"
# #         )
        
# #         policy_file = st.file_uploader(
# #             "📄 Upload Policy File",
# #             type=['pdf', 'png', 'jpg', 'jpeg', 'txt', 'csv', 'xlsx', 'xls'],
# #             help="Upload your insurance policy document"
# #         )
        
# #         st.info(f"⚙️ Chunk Size: {CHUNK_SIZE} rows")
# #         st.info(f"🔄 Max Retries: {MAX_RETRIES}")
        
# #         process_button = st.button(
# #             "🚀 Process Policy File", 
# #             type="primary",
# #             disabled=not policy_file
# #         )
    
# #     # Main content
# #     if not policy_file:
# #         st.info("👆 Please upload a policy file to begin processing.")
# #         st.markdown("""
# #         ### 📋 Instructions:
# #         1. **Company Name**: Enter the insurance company name
# #         2. **Policy File**: Upload the document (PDF, Image, Excel, CSV, etc.)
# #         3. **Process**: Click the process button
        
# #         ### ✨ Key Features:
# #         - **Chunked Processing**: Large Excel files (150+ rows) are processed in chunks of 25 rows
# #         - **Retry Logic**: Failed chunks are automatically retried up to 3 times
# #         - **Progress Tracking**: Real-time progress bar shows processing status
# #         - **No Data Loss**: Every row is processed - no skipping!
# #         - **Multi-format Support**: PDF, Images, Excel, CSV, Text files
# #         - **Smart Formula Application**: Embedded formula rules for accurate calculations
        
# #         ### 🔧 Technical Details:
# #         - **Chunk Size**: 25 rows per chunk (configurable)
# #         - **Rate Limiting**: 0.5s delay between chunks to avoid API limits
# #         - **Error Handling**: Robust retry mechanism with exponential backoff
# #         - **Validation**: Checks that expected number of records are extracted
        
# #         ### 📊 Supported LOBs:
# #         - TW (Two Wheeler)
# #         - PVT CAR (Private Car)
# #         - CV (Commercial Vehicle)
# #         - BUS (School/Staff Bus)
# #         - TAXI
# #         - MISD (Miscellaneous/Tractor)
# #         """)
# #         return
    
# #     if process_button:
# #         try:
# #             policy_file_bytes = policy_file.read()
            
# #             with st.spinner("Processing policy file... This may take a few moments."):
# #                 results = process_files(
# #                     policy_file_bytes, policy_file.name, policy_file.type,
# #                     company_name
# #                 )
            
# #             st.success("🎉 Processing completed successfully!")
            
# #             # Display results in tabs
# #             tab1, tab2, tab3, tab4, tab5 = st.tabs([
# #                 "📊 Final Results", 
# #                 "📝 Extracted Text", 
# #                 "🧾 Parsed Data", 
# #                 "🧮 Calculated Data",
# #                 "📥 Download"
# #             ])
            
# #             with tab1:
# #                 st.subheader("📊 Final Processed Data")
# #                 for sheet_name, df_calc in results["df_calc"].items():
# #                     st.write(f"### {sheet_name}")
# #                     st.dataframe(df_calc, use_container_width=True)
                    
# #                     col1, col2, col3, col4 = st.columns(4)
# #                     with col1:
# #                         st.metric(f"Total Records ({sheet_name})", len(results["calculated_data"][sheet_name]))
# #                     with col2:
# #                         avg_payin = sum([r.get('Payin_Value', 0) for r in results["calculated_data"][sheet_name]]) / len(results["calculated_data"][sheet_name]) if results["calculated_data"][sheet_name] else 0
# #                         st.metric(f"Avg Payin ({sheet_name})", f"{avg_payin:.1f}%")
# #                     with col3:
# #                         segments = set([r.get('Segment', 'N/A') for r in results["calculated_data"][sheet_name]])
# #                         st.metric(f"Unique Segments ({sheet_name})", len(segments))
# #                     with col4:
# #                         st.metric("Company", company_name)
                    
# #                     st.write(f"**Formula Rules Applied for {sheet_name}**")
# #                     formula_summary = {}
# #                     for record in results["calculated_data"][sheet_name]:
# #                         formula = record.get('Formula Used', 'Unknown')
# #                         if formula not in formula_summary:
# #                             formula_summary[formula] = 0
# #                         formula_summary[formula] += 1
# #                     for formula, count in formula_summary.items():
# #                         st.write(f"• **{formula}**: Applied to {count} record(s)")
            
# #             with tab2:
# #                 st.subheader("📝 Extracted Text from Policy File")
# #                 if results["extracted_text"]:
# #                     for sheet_name, text in results["extracted_text"].items():
# #                         st.write(f"### {sheet_name}")
# #                         st.text_area(
# #                             f"Policy Text - {sheet_name}", 
# #                             text, 
# #                             height=400,
# #                             key=f"policy_text_{sheet_name}"
# #                         )
# #                 else:
# #                     st.info("Excel files are processed directly as DataFrames (no text extraction needed)")
                
# #                 st.subheader("📊 Embedded Formula Rules")
# #                 df_formula = pd.DataFrame(FORMULA_DATA)
# #                 st.dataframe(df_formula, use_container_width=True)
            
# #             with tab3:
# #                 st.subheader("🧾 Parsed Policy Data")
# #                 for sheet_name, data in results["parsed_data"].items():
# #                     st.write(f"### {sheet_name}")
# #                     with st.expander(f"View JSON ({len(data)} records)"):
# #                         st.json(data)
            
# #             with tab4:
# #                 st.subheader("🧮 Calculated Data with Formulas")
# #                 for sheet_name, data in results["calculated_data"].items():
# #                     st.write(f"### {sheet_name}")
# #                     with st.expander(f"View JSON ({len(data)} records)"):
# #                         st.json(data)
                    
# #                     st.write(f"**Rule Explanations for {sheet_name}**")
# #                     for i, record in enumerate(data[:10]):  # Show first 10
# #                         with st.expander(f"Record {i+1}: {record.get('Segment', 'Unknown')}"):
# #                             st.write(f"**Payin**: {record.get('Payin', 'N/A')}")
# #                             st.write(f"**Calculated Payout**: {record.get('Calculated Payout', 'N/A')}")
# #                             st.write(f"**Formula Used**: {record.get('Formula Used', 'N/A')}")
# #                             st.write(f"**Rule Explanation**: {record.get('Rule Explanation', 'N/A')}")
# #                     if len(data) > 10:
# #                         st.info(f"Showing first 10 of {len(data)} records")
            
# #             with tab5:
# #                 st.subheader("📥 Download Results")
                
# #                 st.download_button(
# #                     label="📊 Download Consolidated Excel File",
# #                     data=results["excel_data"],
# #                     file_name=f"{company_name}_processed_policies.xlsx",
# #                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
# #                 )
                
# #                 for sheet_name, data in results["calculated_data"].items():
# #                     st.write(f"### {sheet_name}")
# #                     json_data = json.dumps(data, indent=2)
# #                     st.download_button(
# #                         label=f"📄 Download JSON Data ({sheet_name})",
# #                         data=json_data,
# #                         file_name=f"{company_name}_processed_data_{sheet_name}.json",
# #                         mime="application/json",
# #                         key=f"json_{sheet_name}"
# #                     )
# #                     csv_data = results["df_calc"][sheet_name].to_csv(index=False)
# #                     st.download_button(
# #                         label=f"📋 Download CSV File ({sheet_name})",
# #                         data=csv_data,
# #                         file_name=f"{company_name}_processed_policies_{sheet_name}.csv",
# #                         mime="text/csv",
# #                         key=f"csv_{sheet_name}"
# #                     )
                
# #                 st.info("💡 The Excel file contains all processed data with proper formatting.")

# #         except Exception as e:
# #             st.error(f"❌ Error processing files: {str(e)}")
# #             st.exception(e)


# # if __name__ == "__main__":
# #     main()

# import streamlit as st
# import pandas as pd
# from io import BytesIO
# import base64
# import json
# import os
# from dotenv import load_dotenv
# import logging
# import re
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from functools import partial

# # Optional libs
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

# # Embedded Formula Data (unchanged)
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

# # ---------- Helper utilities for chunking & OpenAI batching ----------
# BATCH_SIZE = int(os.getenv('SOMPO_BATCH_SIZE', 20))  # lines per batch
# MAX_WORKERS = int(os.getenv('SOMPO_MAX_WORKERS', 4))  # parallel batches
# REQUEST_DELAY = float(os.getenv('SOMPO_REQUEST_DELAY', 0.3))  # seconds between requests per worker to avoid rate limits


# def chunk_lines(text: str, size: int = BATCH_SIZE) -> list:
#     """Split large text into chunks by non-empty lines."""
#     lines = [l for l in text.splitlines() if l.strip()]
#     if not lines:
#         return []
#     chunks = ["\n".join(lines[i:i+size]) for i in range(0, len(lines), size)]
#     return chunks


# def call_openai_for_batch(batch_text: str, sheet_name: str, company_name: str, batch_index: int):
#     """Send a single batch to OpenAI and return parsed JSON (list)."""
#     parse_prompt = f"""
#     Analyze the following text (partial sheet: batch {batch_index}) belonging to sheet {sheet_name}.
#     Company Name: {company_name}

#     IMPORTANT: Always return a valid JSON array (list) of records. If there is no data in the batch return an empty array [].

#     The fields and extraction rules are identical to the full-file parser used by the app:
#     - Segment, Location, Policy Type, Payin (normalized to percentage), Doable District, Remarks
#     - Preserve GVW/tonnage labels, handle 'Package', 'PCV' -> CV precedence, ignore "discount" sections
    

#     Text to analyze:
#     {batch_text}

#     Return ONLY a valid JSON array.
#     """

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "Extract policy data as JSON array. Always rename 'Payout' → 'Payin'. Normalize all Payin values to percentage format. Use intelligence to identify segments accurately. Return ONLY valid JSON array."},
#                 {"role": "user", "content": parse_prompt}
#             ],
#             temperature=0.1
#         )

#         parsed_json = response.choices[0].message.content.strip()
#         token_usage = getattr(response, 'usage', None)
#         if token_usage:
#             logger.info(f"Batch {batch_index} tokens: prompt={token_usage.prompt_tokens}, completion={token_usage.completion_tokens}, total={token_usage.total_tokens}")

#         cleaned = clean_json_response(parsed_json)
#         try:
#             parsed = json.loads(cleaned)
#             parsed = ensure_list_format(parsed)
#         except Exception as e:
#             logger.error(f"JSON parsing error for batch {batch_index}: {e}. Raw cleaned text: {cleaned[:400]}")
#             parsed = []

#         return parsed
#     except Exception as e:
#         logger.error(f"OpenAI request failed for batch {batch_index}: {e}")
#         return []


# # ---------- Existing helper functions (kept unchanged) ----------

# def extract_text_from_pdf_file(pdf_bytes: bytes) -> str:
#     """Extract text from PDF using multiple methods"""
#     extracted_text = ""

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

#     logger.warning("All PDF text extraction methods failed")
#     return ""


# def extract_text_from_file(file_bytes: bytes, filename: str, content_type: str) -> dict:
#     """Extract text from uploaded file, returning a dictionary with sheet-wise data for Excel files"""
#     file_extension = filename.split('.')[-1].lower() if '.' in filename else ''
#     file_type = content_type if content_type else file_extension

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
#         token_usage = response.usage
#         st.info(f"Tokens used for image extraction: Prompt: {token_usage.prompt_tokens}, Completion: {token_usage.completion_tokens}, Total: {token_usage.total_tokens}")
#         return {"Sheet1": response.choices[0].message.content.strip()}

#     if file_extension == 'pdf':
#         pdf_text = extract_text_from_pdf_file(file_bytes)

#         if pdf_text and len(pdf_text.strip()) > 50:
#             logger.info("Using direct PDF text extraction")
#             return {"Sheet1": pdf_text}
#         else:
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
#             token_usage = response.usage
#             st.info(f"Tokens used for image extraction: Prompt: {token_usage.prompt_tokens}, Completion: {token_usage.completion_tokens}, Total: {token_usage.total_tokens}")
#             return {"Sheet1": response.choices[0].message.content.strip()}

#     if file_extension == 'txt':
#         return {"Sheet1": file_bytes.decode('utf-8', errors='ignore')}

#     if file_extension == 'csv':
#         df = pd.read_csv(BytesIO(file_bytes), encoding='utf-8', errors='ignore')
#         return {"Sheet1": df.to_string()}

#     if file_extension in ['xlsx', 'xls']:
#         all_sheets = pd.read_excel(BytesIO(file_bytes), sheet_name=None)
#         sheet_data = {}
#         for sheet_name, df_sheet in all_sheets.items():
#             sheet_data[sheet_name] = df_sheet.to_string(index=False)
#         return sheet_data

#     raise ValueError(f"Unsupported file type for {filename}")


# def clean_json_response(response_text: str) -> str:
#     """Clean and extract JSON from OpenAI response"""
#     cleaned = re.sub(r'```json\s*', '', response_text)
#     cleaned = re.sub(r'```\s*$', '', cleaned)

#     json_start = -1
#     for i, char in enumerate(cleaned):
#         if char in '[{':
#             json_start = i
#             break

#     if json_start != -1:
#         cleaned = cleaned[json_start:]

#     json_end = -1
#     for i in range(len(cleaned) - 1, -1, -1):
#         if cleaned[i] in ']}' :
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
#         return [data]
#     else:
#         raise ValueError(f"Expected list or dict, got {type(data)}")


# def classify_payin(payin_str):
#     """
#     Converts Payin string (e.g., '50%') to float and classifies its range.
#     """
#     try:
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


# # apply_formula_directly is identical to previous implementation (keeps same matching logic)
# # For brevity I'm including the full function here (unchanged from your original file)

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

#         # Check if Payin_Value is 0; set Calculated Payout to 0% and skip formula application
#         if payin_value == 0:
#             result_record = record.copy()
#             result_record['Calculated Payout'] = "0%"
#             result_record['Formula Used'] = "Payin is 0"
#             result_record['Rule Explanation'] = f"Payin is 0 for Segment={segment}, Company={company_name}; set Calculated Payout to 0%"
#             calculated_data.append(result_record)
#             continue

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
#                 if "UPTO 2.5" in rule_segment:
#                     if any(keyword in segment.upper() for keyword in ["UPTO 2.5", "2.5 TN", "2.5 GVW", "2.5TN", "2.5GVW", "UPTO2.5"]):
#                         segment_match = True
#                 elif "ALL GVW" in rule_segment:
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
#                 segment_match = True

#             if not segment_match:
#                 continue

#             # Check company matching
#             insurers = [ins.strip().upper() for ins in rule["INSURER"].split(',')]
#             company_match = False

#             if "ALL COMPANIES" in insurers:
#                 company_match = True
#             elif "REST OF COMPANIES" in insurers:
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
#                 for insurer in insurers:
#                     if insurer in company_normalized or company_normalized in insurer:
#                         company_match = True
#                         break

#             if not company_match:
#                 continue

#             # Check if remarks require payin category matching
#             remarks = rule.get("REMARKS", "")

#             if remarks == "NIL" or "NIL" in remarks.upper():
#                 matched_rule = rule
#                 rule_explanation = f"Direct match: LOB={lob}, Segment={rule_segment}, Company={rule['INSURER']}, No payin category check (NIL remarks)"
#                 break
#             elif any(payin_keyword in remarks for payin_keyword in ["Payin Below", "Payin 21%", "Payin 31%", "Payin Above"]):
#                 if payin_category in remarks:
#                     matched_rule = rule
#                     rule_explanation = f"Payin category match: LOB={lob}, Segment={rule_segment}, Company={rule['INSURER']}, Payin={payin_category}"
#                     break
#             else:
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
#                 calculated_payout = payin_value

#             calculated_payout = max(0, calculated_payout)

#             formula_used = po_formula
#         else:
#             calculated_payout = payin_value
#             formula_used = "No matching rule found"
#             rule_explanation = f"No formula rule matched for LOB={lob}, Segment={segment}, Company={company_name}"

#         result_record = record.copy()
#         result_record['Calculated Payout'] = f"{int(calculated_payout)}%"
#         result_record['Formula Used'] = formula_used
#         result_record['Rule Explanation'] = rule_explanation

#         calculated_data.append(result_record)

#     return calculated_data


# # ---------- New process_files with chunking + parallel batch processing ----------

# def process_files(policy_file_bytes, policy_filename, policy_content_type, company_name):
#     """Main processing function with chunking and optional parallel batch requests."""
#     try:
#         st.info("🔍 Extracting text from policy file...")

#         extracted_data = extract_text_from_file(policy_file_bytes, policy_filename, policy_content_type)
#         logger.info(f"Extracted data: {list(extracted_data.keys())} sheets")

#         results = {
#             "extracted_text": {},
#             "parsed_data": {},
#             "calculated_data": {},
#             "excel_data": None,
#             "df_calc": {}
#         }

#         output = BytesIO()
#         with pd.ExcelWriter(output, engine='openpyxl') as writer:
#             for sheet_name, extracted_text in extracted_data.items():
#                 st.success(f"✅ Text extracted successfully for {sheet_name}! Length: {len(extracted_text)} chars")

#                 # Decide chunking strategy: if extracted_text is short, process as single batch
#                 batches = chunk_lines(extracted_text, size=BATCH_SIZE)
#                 if not batches:
#                     # No line-based chunks (maybe short free-text) → treat whole text as single batch
#                     batches = [extracted_text]

#                 st.info(f"Processing sheet '{sheet_name}' in {len(batches)} batch(es) (batch_size={BATCH_SIZE})")

#                 parsed_results = []

#                 # Process batches in parallel using ThreadPoolExecutor but respecting rate limits by small sleep inside
#                 with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, max(1, len(batches)))) as executor:
#                     future_to_index = {}
#                     for idx, batch_text in enumerate(batches, start=1):
#                         # submit with a small wrapper to rate-limit slightly
#                         future = executor.submit(call_openai_for_batch, batch_text, sheet_name, company_name, idx)
#                         future_to_index[future] = idx
#                         time.sleep(REQUEST_DELAY)  # small stagger to reduce burst pressure

#                     for future in as_completed(future_to_index):
#                         idx = future_to_index[future]
#                         try:
#                             parsed = future.result()
#                             if parsed:
#                                 parsed_results.extend(parsed)
#                         except Exception as e:
#                             logger.error(f"Batch {idx} failed with exception: {e}")

#                 # If no parsed results returned by OpenAI, create a default record
#                 if not parsed_results:
#                     parsed_results = [{
#                         "Segment": "Unknown",
#                         "Location": "N/A",
#                         "Policy Type": "N/A",
#                         "Payin": "0%",
#                         "Doable District": "N/A",
#                         "Remarks": "No parsed data from OpenAI for this sheet/batches"
#                     }]

#                 st.success(f"✅ Parsed {len(parsed_results)} records for sheet '{sheet_name}'")

#                 # Pre-classify Payin values
#                 for record in parsed_results:
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
#                 calculated_data = apply_formula_directly(parsed_results, company_name)

#                 st.success(f"✅ Successfully calculated {len(calculated_data)} records for {sheet_name}")

#                 # Create DataFrame and write to Excel
#                 df_calc = pd.DataFrame(calculated_data)
#                 df_calc.to_excel(writer, sheet_name=sheet_name, startrow=2, index=False)
#                 worksheet = writer.sheets[sheet_name]
#                 headers = list(df_calc.columns)
#                 for col_num, value in enumerate(headers, 1):
#                     worksheet.cell(row=3, column=col_num, value=value)
#                     worksheet.cell(row=3, column=col_num).font = worksheet.cell(row=3, column=col_num).font.copy(bold=True)
#                 company_cell = worksheet.cell(row=1, column=1, value=company_name)
#                 worksheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max(1, len(headers)))
#                 company_cell.font = company_cell.font.copy(bold=True, size=14)
#                 company_cell.alignment = company_cell.alignment.copy(horizontal='center')
#                 title_cell = worksheet.cell(row=2, column=1, value=f'Policy Data with Payin and Calculated Payouts - {sheet_name}')
#                 worksheet.merge_cells(start_row=2, start_column=1, end_row=2, end_column=max(1, len(headers)))
#                 title_cell.font = title_cell.font.copy(bold=True, size=12)
#                 title_cell.alignment = title_cell.alignment.copy(horizontal='center')

#                 # Store results
#                 results["extracted_text"][sheet_name] = extracted_text
#                 results["parsed_data"][sheet_name] = parsed_results
#                 results["calculated_data"][sheet_name] = calculated_data
#                 results["df_calc"][sheet_name] = df_calc

#         output.seek(0)
#         results["excel_data"] = output.read()
#         return results

#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}", exc_info=True)
#         raise Exception(f"An error occurred: {str(e)}")


# # ---------- Streamlit App (unchanged, small adjustments to mention chunking) ----------

# def main():
#     st.set_page_config(
#         page_title="Insurance Policy Processing (Chunked)", 
#         page_icon="📋", 
#         layout="wide"
#     )

#     st.title("🏢 Insurance Policy Processing System — Chunked Mode")
#     st.markdown("---")

#     with st.sidebar:
#         st.header("📁 File Upload")
#         company_name = st.text_input("Company Name", value="Unknown Company", help="Enter the insurance company name")
#         policy_file = st.file_uploader("📄 Upload Policy File", type=['pdf', 'png', 'jpg', 'jpeg', 'txt', 'csv', 'xlsx', 'xls'], help="Upload your insurance policy document")
#         st.info("📊 Formula rules are embedded in the system and will be automatically applied.")
#         st.write( f"\nCurrent batch size: **{BATCH_SIZE}** lines per batch. Max parallel workers: **{MAX_WORKERS}**.")
#         process_button = st.button("🚀 Process Policy File", type="primary", disabled=not policy_file)

#     if not policy_file:
#         st.info("👆 Please upload a policy file to begin processing.")
#         return

#     if process_button:
#         try:
#             policy_file_bytes = policy_file.read()
#             with st.spinner("Processing policy file with chunking..."):
#                 results = process_files(policy_file_bytes, policy_file.name, policy_file.type, company_name)

#             st.success("🎉 Processing completed successfully!")

#             tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Final Results", "📝 Extracted Text", "🧾 Parsed Data", "🧮 Calculated Data","📥 Download"]) 

#             with tab1:
#                 st.subheader("📊 Final Processed Data")
#                 for sheet_name, df_calc in results["df_calc"].items():
#                     st.write(f"### {sheet_name}")
#                     st.dataframe(df_calc, use_container_width=True)
#                     col1, col2, col3, col4 = st.columns(4)
#                     with col1:
#                         st.metric(f"Total Records ({sheet_name})", len(results["calculated_data"][sheet_name]))
#                     with col2:
#                         avg_payin = sum([r.get('Payin_Value', 0) for r in results["calculated_data"][sheet_name]]) / len(results["calculated_data"][sheet_name]) if results["calculated_data"][sheet_name] else 0
#                         st.metric(f"Avg Payin ({sheet_name})", f"{avg_payin:.1f}%")
#                     with col3:
#                         segments = set([r.get('Segment', 'N/A') for r in results["calculated_data"][sheet_name]])
#                         st.metric(f"Unique Segments ({sheet_name})", len(segments))
#                     with col4:
#                         st.metric("Company", company_name)

#             with tab2:
#                 st.subheader("📝 Extracted Text from Policy File")
#                 for sheet_name, text in results["extracted_text"].items():
#                     st.write(f"### {sheet_name}")
#                     st.text_area(f"Policy Text - {sheet_name}", text, height=400, key=f"policy_text_{sheet_name}")

#             with tab3:
#                 st.subheader("🧾 Parsed Policy Data")
#                 for sheet_name, data in results["parsed_data"].items():
#                     st.write(f"### {sheet_name}")
#                     st.json(data)

#             with tab4:
#                 st.subheader("🧮 Calculated Data with Formulas")
#                 for sheet_name, data in results["calculated_data"].items():
#                     st.write(f"### {sheet_name}")
#                     st.json(data)

#             with tab5:
#                 st.subheader("📥 Download Results")
#                 st.download_button(label="📊 Download Consolidated Excel File", data=results["excel_data"], file_name=f"{company_name}_processed_policies.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

# Optional libs
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

# Embedded Formula Data (unchanged)
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

# ---------- Config ----------
BATCH_SIZE = int(os.getenv('SOMPO_BATCH_SIZE', 20))  # rows per batch for DataFrame
MAX_WORKERS = int(os.getenv('SOMPO_MAX_WORKERS', 4))  # parallel batches
REQUEST_DELAY = float(os.getenv('SOMPO_REQUEST_DELAY', 0.2))  # seconds between submitting futures

# ---------- Helpers ----------

def chunk_lines(text: str, size: int = BATCH_SIZE) -> list:
    """Split large text into chunks by non-empty lines."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return []
    chunks = ["\n".join(lines[i:i+size]) for i in range(0, len(lines), size)]
    return chunks


def chunk_dataframe_rows(df: pd.DataFrame, size: int = BATCH_SIZE) -> list:
    """Split a DataFrame into list of CSV snippets, each containing `size` rows and header."""
    if df is None or df.shape[0] == 0:
        return []
    chunks = []
    for i in range(0, df.shape[0], size):
        slice_df = df.iloc[i:i+size]
        # Convert to CSV snippet (without index). This keeps column headers for the model.
        csv_snippet = slice_df.to_csv(index=False)
        chunks.append(csv_snippet)
    return chunks


def call_openai_for_batch(batch_text: str, sheet_name: str, company_name: str, batch_index: int):
    """Send a single batch to OpenAI and return parsed JSON (list)."""
    # The parse_prompt is intentionally detailed (keeps your domain rules). The batch_text
    # may be raw text or CSV snippet—the prompt asks the model to handle both.
    parse_prompt = f"""
    Analyze the following text (partial sheet: batch {batch_index}) belonging to sheet {sheet_name}.
    Company Name: {company_name}

    IMPORTANT: Always return a valid JSON array (list) of records. If there is no data in the batch return an empty array [].

    The fields and extraction rules are identical to the full-file parser used by the app:
    - Segment, Location, Policy Type, Payin (normalized to percentage), Doable District, Remarks
    - Preserve GVW/tonnage labels, handle 'Package', 'PCV' -> Comes under Taxi or Bus, ignore "discount" sections

    Domain-specific rules:
    - PCV is Passenger Carrying Vehicle. It falls under BUS or TAXI categories:
        * If a seater count is mentioned and fits the Taxi range, classify under TAXI.
        * If seater fits under Bus (e.g., Staff Bus or School Bus), classify under BUS.
        * For PCV, searching through policy type does not matter.

    - PVT CAR:
        * If 'package' is mentioned, it means COMP. So LOB is PVT CAR and segment is PVT CAR COMP+SAOD.
        * If 'TP' is mentioned, then classify as PVT CAR TP.
        * If 'more than 5 years old (without addon) on net' is mentioned, it means COMP. So segment is PVT CAR COMP+SAOD.

    - TW (Two Wheeler):
        * If 'SAOD' is mentioned, then classify as TW SAOD+COMP.
        * If 'TP' is mentioned, then classify as TW TP.
        * If 'COMP' is mentioned, then classify as TW SAOD+COMP.

    Text to analyze (the text may be raw or a CSV table with headers):
    {batch_text}

    Return ONLY a valid JSON array.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract policy data as JSON array. Always rename 'Payout' → 'Payin'. Normalize all Payin values to percentage format. Use intelligence to identify segments accurately. Return ONLY valid JSON array."},
                {"role": "user", "content": parse_prompt}
            ],
            temperature=0.1
        )

        parsed_json = response.choices[0].message.content.strip()
        token_usage = getattr(response, 'usage', None)
        if token_usage:
            logger.info(f"Batch {batch_index} tokens: prompt={token_usage.prompt_tokens}, completion={token_usage.completion_tokens}, total={token_usage.total_tokens}")

        cleaned = clean_json_response(parsed_json)
        try:
            parsed = json.loads(cleaned)
            parsed = ensure_list_format(parsed)
        except Exception as e:
            logger.error(f"JSON parsing error for batch {batch_index}: {e}. Raw cleaned text: {cleaned[:800]}")
            parsed = []

        return parsed
    except Exception as e:
        logger.error(f"OpenAI request failed for batch {batch_index}: {e}")
        return []


# ---------- Existing helper functions (kept, with small changes) ----------

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


def extract_text_from_file(file_bytes: bytes, filename: str, content_type: str) -> dict:
    """Extract text from uploaded file, returning sheet-wise data. For Excel sheets we return DataFrames so we can chunk row-wise."""
    file_extension = filename.split('.')[-1].lower() if '.' in filename else ''
    file_type = content_type if content_type else file_extension

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

    if file_extension == 'pdf':
        pdf_text = extract_text_from_pdf_file(file_bytes)

        if pdf_text and len(pdf_text.strip()) > 50:
            logger.info("Using direct PDF text extraction")
            return {"Sheet1": pdf_text}
        else:
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

    if file_extension == 'txt':
        return {"Sheet1": file_bytes.decode('utf-8', errors='ignore')}

    if file_extension == 'csv':
        df = pd.read_csv(BytesIO(file_bytes), encoding='utf-8', errors='ignore')
        return {"Sheet1": df}

    if file_extension in ['xlsx', 'xls']:
        all_sheets = pd.read_excel(BytesIO(file_bytes), sheet_name=None)
        sheet_data = {}
        for sheet_name, df_sheet in all_sheets.items():
            # Return the DataFrame directly so we can chunk rows preserving columns
            sheet_data[sheet_name] = df_sheet
        return sheet_data

    raise ValueError(f"Unsupported file type for {filename}")


def clean_json_response(response_text: str) -> str:
    """Clean and extract JSON from OpenAI response"""
    cleaned = re.sub(r'```json\s*', '', response_text)
    cleaned = re.sub(r'```\s*$', '', cleaned)

    # Find first { or [ and last } or ]
    json_start = -1
    for i, char in enumerate(cleaned):
        if char in '[{':
            json_start = i
            break

    if json_start != -1:
        cleaned = cleaned[json_start:]

    json_end = -1
    for i in range(len(cleaned) - 1, -1, -1):
        if cleaned[i] in ']}' :
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
        return [data]
    else:
        raise ValueError(f"Expected list or dict, got {type(data)}")


def classify_payin(payin_str):
    """
    Converts Payin string (e.g., '50%') to float and classifies its range.
    """
    try:
        payin_clean = str(payin_str).replace('%', '').replace(' ', '')
        if payin_clean in ['NIL', '', 'N/A', 'NA', 'nan', 'None']:
            return 0.0, "Payin Below 20%"
        payin_value = float(re.sub(r'[^0-9\.]', '', payin_clean))

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
#     Apply formula rules directly. This function now contains more flexible LOB/segment matching
#     and fallback strategies to avoid 'UNKNOWN' when the model classifies segments slightly differently.
#     """
#     calculated_data = []

#     for record in policy_data:
#         # Skip explicit Discount rows if model produced them
#         seg_raw = str(record.get('Segment', '')).strip()
#         if not seg_raw or 'DISCOUNT' in seg_raw.upper():
#             # skip discount rows entirely
#             continue

#         segment = seg_raw.upper()
#         payin_value = record.get('Payin_Value', 0)
#         payin_category = record.get('Payin_Category', '')

#         # If Payin_Value is 0; keep it zero
#         if payin_value == 0:
#             result_record = record.copy()
#             result_record['Calculated Payout'] = "0%"
#             result_record['Formula Used'] = "Payin is 0"
#             result_record['Rule Explanation'] = f"Payin is 0 for Segment={segment}, Company={company_name}; set Calculated Payout to 0%"
#             calculated_data.append(result_record)
#             continue

#         # Normalize segment text for pattern matching
#         segment_clean = re.sub(r"[^A-Z0-9 %&()\\-]", ' ', segment)
#         segment_clean = re.sub(r"\s+", ' ', segment_clean).strip()

#         # Build a list of candidate LOB guesses (try sensible fallbacks)
#         candidates = []
#         if any(k in segment_clean for k in ['TW', '2W', 'TWO WHEELER']):
#             candidates = ['TW']
#         elif any(k in segment_clean for k in ['PVT CAR', 'PRIVATE CAR', 'CAR', 'PACKAGE', 'COMP', 'SAOD', 'TP']):
#             candidates = ['PVT CAR']
#         elif 'PCV' in segment_clean:
#             # PCV can be classified by the model as BUS or TAXI, but our formula grid for PCV historically matched CV rules.
#             # Attempt CV first, then BUS, then TAXI as fallback.
#             candidates = ['CV', 'BUS', 'TAXI']
#         elif any(k in segment_clean for k in ['CV', 'COMMERCIAL', 'LCV', 'GVW', 'TN']):
#             candidates = ['CV']
#         elif 'BUS' in segment_clean:
#             candidates = ['BUS']
#         elif 'TAXI' in segment_clean:
#             candidates = ['TAXI']
#         else:
#             candidates = ['UNKNOWN']

#         matched_rule = None
#         rule_explanation = ''

#         # Try candidates in order
#         for cand in candidates:
#             for rule in FORMULA_DATA:
#                 rule_lob = rule.get('LOB', '').upper()
#                 if rule_lob != cand:
#                     continue

#                 # Segment matching heuristics: flexible substring matching
#                 rule_segment = rule.get('SEGMENT', '').upper()
#                 segment_match = False

#                 # Exact or substring matches
#                 if rule_segment and (rule_segment in segment_clean or segment_clean in rule_segment):
#                     segment_match = True

#                 # CV-specific checks
#                 if not segment_match and cand == 'CV':
#                     if 'PCV' in segment_clean and 'PCV' in rule_segment:
#                         segment_match = True
#                     # Handle GVW/upto 2.5 logic
#                     if 'UPTO 2.5' in rule_segment and any(k in segment_clean for k in ['UPTO 2.5', '2.5 TN', '2.5GVW', '2.5']):
#                         segment_match = True
#                     if 'ALL GVW' in rule_segment and not any(k in segment_clean for k in ['UPTO 2.5', '2.5 TN', '2.5GVW', '2.5']):
#                         segment_match = True

#                 # BUS specifics: allow PCV/SATP to match generic BUS rules if rule is school/staff
#                 if not segment_match and cand == 'BUS':
#                     if 'PCV' in segment_clean and any(k in rule_segment for k in ['SCHOOL', 'STAFF', 'BUS']):
#                         segment_match = True

#                 if not segment_match:
#                     continue

#                 # Company matching (same as before)
#                 insurers = [ins.strip().upper() for ins in rule.get('INSURER', '').split(',')]
#                 company_match = False
#                 company_normalized = company_name.upper().replace('GENERAL', '').replace('INSURANCE', '').strip()

#                 if 'ALL COMPANIES' in insurers:
#                     company_match = True
#                 elif 'REST OF COMPANIES' in insurers:
#                     # ensure this company is not explicitly listed elsewhere
#                     is_in_specific_list = False
#                     for other_rule in FORMULA_DATA:
#                         if other_rule.get('LOB') == rule.get('LOB') and other_rule.get('SEGMENT') == rule.get('SEGMENT') and 'REST OF COMPANIES' not in other_rule.get('INSURER', '') and 'ALL COMPANIES' not in other_rule.get('INSURER', ''):
#                             other_insurers = [ins.strip().upper() for ins in other_rule.get('INSURER', '').split(',')]
#                             if any(company_key in company_normalized for company_key in other_insurers):
#                                 is_in_specific_list = True
#                                 break
#                     if not is_in_specific_list:
#                         company_match = True
#                 else:
#                     for insurer in insurers:
#                         if insurer and (insurer in company_normalized or company_normalized in insurer):
#                             company_match = True
#                             break

#                 if not company_match:
#                     continue

#                 # Remarks / payin category checks
#                 remarks = rule.get('REMARKS', '')
#                 if remarks == 'NIL' or 'NIL' in str(remarks).upper():
#                     matched_rule = rule
#                     rule_explanation = f"Direct match: LOB={cand}, Segment={rule_segment}, Company={rule.get('INSURER')}, No payin category check (NIL remarks)"
#                     break
#                 elif any(payin_keyword in str(remarks) for payin_keyword in ['Payin Below', 'Payin 21%', 'Payin 31%', 'Payin Above']):
#                     if payin_category in remarks:
#                         matched_rule = rule
#                         rule_explanation = f"Payin category match: LOB={cand}, Segment={rule_segment}, Company={rule.get('INSURER')}, Payin={payin_category}"
#                         break
#                 else:
#                     matched_rule = rule
#                     rule_explanation = f"Other remarks match: LOB={cand}, Segment={rule_segment}, Company={rule.get('INSURER')}, Remarks={remarks}"
#                     break

#             if matched_rule:
#                 break

#         # If rule matched, compute payout
#         if matched_rule:
#             po_formula = matched_rule.get('PO', '')
#             calculated_payout = 0
#             if '90% of Payin' in po_formula:
#                 calculated_payout = payin_value * 0.9
#             elif '88% of Payin' in po_formula:
#                 calculated_payout = payin_value * 0.88
#             elif 'Less 2% of Payin' in po_formula or '-2%' in po_formula:
#                 calculated_payout = payin_value - 2
#             elif '-3%' in po_formula:
#                 calculated_payout = payin_value - 3
#             elif '-4%' in po_formula:
#                 calculated_payout = payin_value - 4
#             elif '-5%' in po_formula:
#                 calculated_payout = payin_value - 5
#             else:
#                 calculated_payout = payin_value

#             calculated_payout = max(0, calculated_payout)
#             formula_used = po_formula
#         else:
#             calculated_payout = payin_value
#             formula_used = 'No matching rule found'
#             rule_explanation = f"No formula rule matched for LOBs tried={candidates}, Segment={segment_clean}, Company={company_name}"

#         result_record = record.copy()
#         result_record['Calculated Payout'] = f"{int(calculated_payout)}%"
#         result_record['Formula Used'] = formula_used
#         result_record['Rule Explanation'] = rule_explanation

#         calculated_data.append(result_record)

#     return calculated_data

def apply_formula_directly(policy_data, company_name):
    """
    Apply formula rules directly. Normalizes both incoming segment text and the
    rule SEGMENT text so variants like 'TW SAOD + COMP' and 'TW SAOD COMP' match.
    """
    calculated_data = []

    def normalize_segment_text(s: str) -> str:
        if s is None:
            return ""
        s = str(s).upper()
        # unify plus, slash, ampersand into spaces
        s = s.replace('+', ' ').replace('/', ' ').replace('&', ' ')
        # remove punctuation except alphanumerics and spaces
        s = re.sub(r'[^A-Z0-9 ]', ' ', s)
        # collapse multiple spaces
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    for record in policy_data:
        seg_raw = str(record.get('Segment', '')).strip()
        if not seg_raw or 'DISCOUNT' in seg_raw.upper():
            continue  # skip discount rows

        segment = seg_raw.upper()
        payin_value = record.get('Payin_Value', 0)
        payin_category = record.get('Payin_Category', '')

        # If Payin_Value is 0; shortcut
        if payin_value == 0:
            result_record = record.copy()
            result_record['Calculated Payout'] = "0%"
            result_record['Formula Used'] = "Payin is 0"
            result_record['Rule Explanation'] = f"Payin is 0 for Segment={segment}, Company={company_name}; set Calculated Payout to 0%"
            calculated_data.append(result_record)
            continue

        segment_clean = normalize_segment_text(segment)

        # Candidate LOB guesses
        candidates = []
        if any(k in segment_clean for k in ['TW', '2W', 'TWO WHEELER', 'SAOD', 'TP']) and 'CAR' not in segment_clean:
            candidates = ['TW']
        elif any(k in segment_clean for k in ['PVT CAR', 'PRIVATE CAR', 'CAR', 'PACKAGE', 'COMP', 'SAOD', 'TP']):
            candidates = ['PVT CAR']
        elif 'PCV' in segment_clean:
            candidates = ['CV', 'BUS', 'TAXI']
        elif any(k in segment_clean for k in ['CV', 'COMMERCIAL', 'LCV', 'GVW', 'TN']):
            candidates = ['CV']
        elif 'BUS' in segment_clean:
            candidates = ['BUS']
        elif 'TAXI' in segment_clean:
            candidates = ['TAXI']
        else:
            candidates = ['UNKNOWN']

        matched_rule = None
        rule_explanation = ''

        for cand in candidates:
            for rule in FORMULA_DATA:
                rule_lob = rule.get('LOB', '').upper()
                if rule_lob != cand:
                    continue

                rule_segment = str(rule.get('SEGMENT', '')).upper()
                rule_segment_clean = normalize_segment_text(rule_segment)

                # flexible substring matching
                segment_match = False
                if rule_segment_clean and (rule_segment_clean in segment_clean or segment_clean in rule_segment_clean):
                    segment_match = True

                # Heuristics for TW
                if not segment_match and cand == 'TW':
                    if ('SAOD' in segment_clean or 'COMP' in segment_clean) and ('SAOD' in rule_segment_clean or 'COMP' in rule_segment_clean):
                        segment_match = True
                    if 'TP' in segment_clean and 'TP' in rule_segment_clean:
                        segment_match = True

                # CV specifics
                if not segment_match and cand == 'CV':
                    if 'UPTO 2.5' in rule_segment_clean and any(k in segment_clean for k in ['UPTO 2.5', '2.5TN', '2.5 GVW']):
                        segment_match = True
                    if 'ALL GVW' in rule_segment_clean and not any(k in segment_clean for k in ['UPTO 2.5', '2.5TN', '2.5 GVW']):
                        segment_match = True
                    if 'PCV' in segment_clean and 'PCV' in rule_segment_clean:
                        segment_match = True

                # BUS specifics
                if not segment_match and cand == 'BUS':
                    if 'PCV' in segment_clean and any(k in rule_segment_clean for k in ['SCHOOL', 'STAFF', 'BUS']):
                        segment_match = True

                if not segment_match:
                    continue

                # Company matching
                insurers = [ins.strip().upper() for ins in rule.get('INSURER', '').split(',')]
                company_match = False
                company_normalized = company_name.upper().replace('GENERAL', '').replace('INSURANCE', '').strip()

                if 'ALL COMPANIES' in insurers:
                    company_match = True
                elif 'REST OF COMPANIES' in insurers:
                    is_in_specific_list = False
                    for other_rule in FORMULA_DATA:
                        if other_rule.get('LOB') == rule.get('LOB') and other_rule.get('SEGMENT') == rule.get('SEGMENT') and \
                           'REST OF COMPANIES' not in other_rule.get('INSURER', '') and 'ALL COMPANIES' not in other_rule.get('INSURER', ''):
                            other_insurers = [ins.strip().upper() for ins in other_rule.get('INSURER', '').split(',')]
                            if any(company_key in company_normalized for company_key in other_insurers):
                                is_in_specific_list = True
                                break
                    if not is_in_specific_list:
                        company_match = True
                else:
                    for insurer in insurers:
                        if insurer and (insurer in company_normalized or company_normalized in insurer):
                            company_match = True
                            break

                if not company_match:
                    continue

                remarks = rule.get('REMARKS', '')
                if remarks == 'NIL' or 'NIL' in str(remarks).upper():
                    matched_rule = rule
                    rule_explanation = f"Direct match: LOB={cand}, Segment={rule_segment}, Company={rule.get('INSURER')}, No payin category check"
                    break
                elif any(payin_keyword in str(remarks) for payin_keyword in ['Payin Below', 'Payin 21%', 'Payin 31%', 'Payin Above']):
                    if payin_category in remarks:
                        matched_rule = rule
                        rule_explanation = f"Payin category match: LOB={cand}, Segment={rule_segment}, Company={rule.get('INSURER')}, Payin={payin_category}"
                        break
                else:
                    matched_rule = rule
                    rule_explanation = f"Other remarks match: LOB={cand}, Segment={rule_segment}, Company={rule.get('INSURER')}, Remarks={remarks}"
                    break

            if matched_rule:
                break

        if matched_rule:
            po_formula = matched_rule.get('PO', '')
            calculated_payout = 0
            if '90% of PAYIN' in po_formula.upper():
                calculated_payout = payin_value * 0.9
            elif '88% of PAYIN' in po_formula.upper():
                calculated_payout = payin_value * 0.88
            elif 'LESS 2% OF PAYIN' in po_formula.upper() or '-2%' in po_formula:
                calculated_payout = payin_value - 2
            elif '-3%' in po_formula:
                calculated_payout = payin_value - 3
            elif '-4%' in po_formula:
                calculated_payout = payin_value - 4
            elif '-5%' in po_formula:
                calculated_payout = payin_value - 5
            else:
                calculated_payout = payin_value

            calculated_payout = max(0, calculated_payout)
            formula_used = po_formula
        else:
            calculated_payout = payin_value
            formula_used = 'No matching rule found'
            rule_explanation = f"No formula rule matched for LOBs tried={candidates}, Segment={segment_clean}, Company={company_name}"

        result_record = record.copy()
        result_record['Calculated Payout'] = f"{int(calculated_payout)}%"
        result_record['Formula Used'] = formula_used
        result_record['Rule Explanation'] = rule_explanation

        calculated_data.append(result_record)

    return calculated_data


# ---------- New process_files with DataFrame row-wise chunking + parallel batches ----------

def process_files(policy_file_bytes, policy_filename, policy_content_type, company_name):
    """Main processing function that preserves Excel structure, chunks DataFrame rows when possible,
    and applies improved formula matching."""
    try:
        st.info("🔍 Extracting text from policy file...")

        extracted_data = extract_text_from_file(policy_file_bytes, policy_filename, policy_content_type)
        logger.info(f"Extracted data: {list(extracted_data.keys())} sheets")

        results = {
            "extracted_text": {},
            "parsed_data": {},
            "calculated_data": {},
            "excel_data": None,
            "df_calc": {},
            "raw_parsed_batches": {}
        }

        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for sheet_name, sheet_content in extracted_data.items():
                # If sheet_content is DataFrame, do row-wise chunking preserving headers
                if isinstance(sheet_content, pd.DataFrame):
                    df_sheet = sheet_content
                    st.success(f"✅ Loaded DataFrame for sheet '{sheet_name}' with {len(df_sheet)} rows")
                    # Create CSV batches (each batch includes header)
                    batches = chunk_dataframe_rows(df_sheet, size=BATCH_SIZE)
                    # For transparency, store original DataFrame as extracted_text (CSV)
                    results['extracted_text'][sheet_name] = df_sheet.to_csv(index=False)
                else:
                    # sheet_content is plain text
                    st.success(f"✅ Text extracted successfully for {sheet_name}! Length: {len(sheet_content)} chars")
                    batches = chunk_lines(sheet_content, size=BATCH_SIZE)
                    results['extracted_text'][sheet_name] = sheet_content

                if not batches:
                    batches = [results['extracted_text'][sheet_name] or '']

                st.info(f"Processing sheet '{sheet_name}' in {len(batches)} batch(es) (batch_size={BATCH_SIZE})")

                parsed_results = []
                raw_batches = []

                # Submit batches in parallel
                with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, max(1, len(batches)))) as executor:
                    future_to_index = {}
                    for idx, batch_text in enumerate(batches, start=1):
                        # If this is CSV content, prefix it so we can tell the model it's a table (not mandatory)
                        # but we keep content unchanged mostly so the prompt applies as-is.
                        future = executor.submit(call_openai_for_batch, batch_text, sheet_name, company_name, idx)
                        future_to_index[future] = idx
                        time.sleep(REQUEST_DELAY)

                    for future in as_completed(future_to_index):
                        idx = future_to_index[future]
                        try:
                            parsed = future.result()
                            if parsed:
                                parsed_results.extend(parsed)
                            raw_batches.append((idx, parsed))
                        except Exception as e:
                            logger.error(f"Batch {idx} failed with exception: {e}")

                # Save raw parsed batches for debugging
                results['raw_parsed_batches'][sheet_name] = raw_batches

                # If nothing parsed, create a placeholder
                if not parsed_results:
                    parsed_results = [{
                        "Segment": "Unknown",
                        "Location": "N/A",
                        "Policy Type": "N/A",
                        "Payin": "0%",
                        "Doable District": "N/A",
                        "Remarks": "No parsed data from OpenAI for this sheet/batches"
                    }]

                st.success(f"✅ Parsed {len(parsed_results)} records for sheet '{sheet_name}'")

                # Filter out rows that are clearly discounts or header duplicates
                filtered_results = []
                for rec in parsed_results:
                    seg = str(rec.get('Segment', '')).strip().upper()
                    if 'DISCOUNT' in seg:
                        continue
                    filtered_results.append(rec)
                if not filtered_results:
                    filtered_results = parsed_results

                # Pre-classify Payin values
                for record in filtered_results:
                    try:
                        payin_val, payin_cat = classify_payin(record.get('Payin', '0%'))
                        record['Payin_Value'] = payin_val
                        record['Payin_Category'] = payin_cat
                    except Exception as e:
                        logger.warning(f"Error classifying payin for record in {sheet_name}: {record}, error: {str(e)}")
                        record['Payin_Value'] = 0.0
                        record['Payin_Category'] = "Payin Below 20%"

                # Apply formulas
                st.info(f"🧮 Applying formulas and calculating payouts for {sheet_name}...")
                calculated_data = apply_formula_directly(filtered_results, company_name)

                st.success(f"✅ Successfully calculated {len(calculated_data)} records for {sheet_name}")

                # Create DataFrame and write to Excel
                df_calc = pd.DataFrame(calculated_data)
                # Ensure we always write a non-empty DataFrame
                if df_calc.empty:
                    df_calc = pd.DataFrame([{
                        'Segment': 'N/A', 'Location': 'N/A', 'Policy Type': 'N/A', 'Payin': '0%',
                        'Payin_Value': 0, 'Payin_Category': 'Payin Below 20%', 'Calculated Payout': '0%'
                    }])

                df_calc.to_excel(writer, sheet_name=sheet_name[:31], startrow=2, index=False)
                worksheet = writer.sheets[sheet_name[:31]]
                headers = list(df_calc.columns)
                for col_num, value in enumerate(headers, 1):
                    worksheet.cell(row=3, column=col_num, value=value)
                    worksheet.cell(row=3, column=col_num).font = worksheet.cell(row=3, column=col_num).font.copy(bold=True)
                company_cell = worksheet.cell(row=1, column=1, value=company_name)
                worksheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max(1, len(headers)))
                company_cell.font = company_cell.font.copy(bold=True, size=14)
                company_cell.alignment = company_cell.alignment.copy(horizontal='center')
                title_cell = worksheet.cell(row=2, column=1, value=f'Policy Data with Payin and Calculated Payouts - {sheet_name}')
                worksheet.merge_cells(start_row=2, start_column=1, end_row=2, end_column=max(1, len(headers)))
                title_cell.font = title_cell.font.copy(bold=True, size=12)
                title_cell.alignment = title_cell.alignment.copy(horizontal='center')

                # Store results
                results["parsed_data"][sheet_name] = filtered_results
                results["calculated_data"][sheet_name] = calculated_data
                results["df_calc"][sheet_name] = df_calc

        output.seek(0)
        results["excel_data"] = output.read()
        return results

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise Exception(f"An error occurred: {str(e)}")


# ---------- Streamlit App ----------

def main():
    st.set_page_config(
        page_title="Insurance Policy Processing (Chunked)", 
        page_icon="📋", 
        layout="wide"
    )

    st.title("🏢 Insurance Policy Processing System — Chunked Mode")
    st.markdown("---")

    with st.sidebar:
        st.header("📁 File Upload")
        company_name = st.text_input("Company Name", value="Unknown Company", help="Enter the insurance company name")
        policy_file = st.file_uploader("📄 Upload Policy File", type=['pdf', 'png', 'jpg', 'jpeg', 'txt', 'csv', 'xlsx', 'xls'], help="Upload your insurance policy document")
        st.info("📊 Formula rules are embedded in the system and will be automatically applied.")
        st.write( f"\nCurrent batch size (rows per batch): **{BATCH_SIZE}**. Max parallel workers: **{MAX_WORKERS}**.")
        process_button = st.button("🚀 Process Policy File", type="primary", disabled=not policy_file)

    if not policy_file:
        st.info("👆 Please upload a policy file to begin processing.")
        return

    if process_button:
        try:
            policy_file_bytes = policy_file.read()
            with st.spinner("Processing policy file with chunking..."):
                results = process_files(policy_file_bytes, policy_file.name, policy_file.type, company_name)

            st.success("🎉 Processing completed successfully!")

            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Final Results", "📝 Extracted Text", "🧾 Parsed Data", "🧮 Calculated Data","📥 Download"]) 

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

            with tab2:
                st.subheader("📝 Extracted Text from Policy File")
                for sheet_name, text in results["extracted_text"].items():
                    st.write(f"### {sheet_name}")
                    st.text_area(f"Policy Text - {sheet_name}", text, height=400, key=f"policy_text_{sheet_name}")

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

            with tab5:
                st.subheader("📥 Download Results")
                st.download_button(label="📊 Download Consolidated Excel File", data=results["excel_data"], file_name=f"{company_name}_processed_policies.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            st.error(f"❌ Error processing files: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()
