# # # import streamlit as st
# # # import pandas as pd
# # # from io import BytesIO
# # # import base64
# # # import json
# # # import os
# # # from dotenv import load_dotenv
# # # import logging
# # # import re

# # # # Check if required packages are available
# # # try:
# # #     from openai import OpenAI
# # # except ImportError:
# # #     st.error("OpenAI package not found. Please install it using 'pip install openai'")
# # #     st.stop()

# # # try:
# # #     import PyPDF2
# # # except ImportError:
# # #     st.warning("PyPDF2 not found. PDF text extraction will use OpenAI vision only.")
# # #     PyPDF2 = None

# # # try:
# # #     import pdfplumber
# # # except ImportError:
# # #     st.warning("pdfplumber not found. PDF text extraction will use PyPDF2 or OpenAI vision only.")
# # #     pdfplumber = None

# # # # Load environment variables
# # # load_dotenv()

# # # # Configure logging
# # # logging.basicConfig(level=logging.INFO)
# # # logger = logging.getLogger(__name__)

# # # # Load OpenAI API key
# OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")# # # if not OPENAI_API_KEY:
# # #     st.error("OPENAI_API_KEY environment variable not set. Please set it in your .env file or environment variables.")
# # #     st.stop()

# # # # Initialize OpenAI client
# # # try:
# # #     client = OpenAI(api_key=OPENAI_API_KEY)
# # # except Exception as e:
# # #     st.error(f"Failed to initialize OpenAI client: {str(e)}")
# # #     st.stop()


# # # def extract_text_from_pdf_file(pdf_bytes: bytes) -> str:
# # #     """Extract text from PDF using multiple methods"""
# # #     extracted_text = ""
    
# # #     # Method 1: Try pdfplumber first (most accurate)
# # #     if pdfplumber:
# # #         try:
# # #             with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
# # #                 for page in pdf.pages:
# # #                     page_text = page.extract_text()
# # #                     if page_text:
# # #                         extracted_text += page_text + "\n"
# # #             if extracted_text.strip():
# # #                 logger.info("PDF text extracted using pdfplumber")
# # #                 return extracted_text.strip()
# # #         except Exception as e:
# # #             logger.warning(f"pdfplumber extraction failed: {str(e)}")
    
# # #     # Method 2: Try PyPDF2 as fallback
# # #     if PyPDF2:
# # #         try:
# # #             pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
# # #             for page in pdf_reader.pages:
# # #                 page_text = page.extract_text()
# # #                 if page_text:
# # #                     extracted_text += page_text + "\n"
# # #             if extracted_text.strip():
# # #                 logger.info("PDF text extracted using PyPDF2")
# # #                 return extracted_text.strip()
# # #         except Exception as e:
# # #             logger.warning(f"PyPDF2 extraction failed: {str(e)}")
    
# # #     # Method 3: Return empty string if all methods fail
# # #     logger.warning("All PDF text extraction methods failed")
# # #     return ""


# # # def extract_text_from_file(file_bytes: bytes, filename: str, content_type: str, is_formula: bool = False) -> str:
# # #     """Extract text from uploaded file"""
# # #     file_extension = filename.split('.')[-1].lower() if '.' in filename else ''
# # #     file_type = content_type if content_type else file_extension

# # #     # Image-based extraction with enhanced OCR
# # #     image_extensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']
# # #     if file_extension in image_extensions or file_type.startswith('image/'):
# # #         image_base64 = base64.b64encode(file_bytes).decode('utf-8')
        
# # #         if is_formula:
# # #             prompt = """Extract all formulas, rules and grid information accurately from this image. 
# # #             Focus on identifying:
# # #             - LOB segments (TW, PVT CAR, CV, BUS, TAXI, MISD)
# # #             - Policy types (TP, COMP, SAOD, etc.)
# # #             - Company names and their specific rules
# # #             - Percentage deductions and formulas
# # #             - Payin categories and their corresponding rates
# # #             Extract everything exactly as shown in the grid/table format."""
# # #         else:
# # #             prompt = """Extract all insurance policy text accurately from this image using OCR.
# # #             Pay special attention to identifying:
# # #             - Segment information (like TW, PVT CAR, CV, BUS, TAXI, MISD)
# # #             - Company names
# # #             - Policy types
# # #             - Payin/Payout percentages
# # #             - Any numerical values
# # #             - Location information
# # #             Extract all text exactly as it appears."""
            
# # #         response = client.chat.completions.create(
# # #             model="gpt-4o",
# # #             messages=[{
# # #                 "role": "user",
# # #                 "content": [
# # #                     {"type": "text", "text": prompt},
# # #                     {"type": "image_url", "image_url": {"url": f"data:image/{file_extension};base64,{image_base64}"}}
# # #                 ]
# # #             }],
# # #             temperature=0.1
# # #         )
# # #         return response.choices[0].message.content.strip()

# # #     # Enhanced PDF extraction
# # #     if file_extension == 'pdf':
# # #         # First try to extract text directly from PDF
# # #         pdf_text = extract_text_from_pdf_file(file_bytes)
        
# # #         if pdf_text and len(pdf_text.strip()) > 50:
# # #             # If we got good text extraction, use it
# # #             logger.info("Using direct PDF text extraction")
# # #             return pdf_text
# # #         else:
# # #             # Fallback to OpenAI vision for PDF
# # #             logger.info("Using OpenAI vision for PDF")
# # #             pdf_base64 = base64.b64encode(file_bytes).decode("utf-8")
            
# # #             if is_formula:
# # #                 prompt = """Extract formulas, rules, and grid information from this PDF.
# # #                 Look for LOB segments, company-specific rules, percentage deductions, and payin categories."""
# # #             else:
# # #                 prompt = """Extract insurance policy details from this PDF.
# # #                 Focus on identifying segments, company names, policy types, and payout percentages."""
                
# # #             response = client.chat.completions.create(
# # #                 model="gpt-4o",
# # #                 messages=[{
# # #                     "role": "user",
# # #                     "content": [
# # #                         {"type": "text", "text": prompt},
# # #                         {"type": "image_url", "image_url": {"url": f"data:application/pdf;base64,{pdf_base64}"}}
# # #                     ]
# # #                 }],
# # #                 temperature=0.1
# # #             )
# # #             return response.choices[0].message.content.strip()

# # #     # Text files
# # #     if file_extension == 'txt':
# # #         return file_bytes.decode('utf-8', errors='ignore')

# # #     # CSV files
# # #     if file_extension == 'csv':
# # #         df = pd.read_csv(BytesIO(file_bytes), encoding='utf-8', errors='ignore')
# # #         return df.to_string()

# # #     # Excel files
# # #     if file_extension in ['xlsx', 'xls']:
# # #         all_sheets = pd.read_excel(BytesIO(file_bytes), sheet_name=None)
# # #         dfs = []
# # #         for sheet_name, df_sheet in all_sheets.items():
# # #             df_sheet["Source_Sheet"] = sheet_name
# # #             dfs.append(df_sheet)
# # #         df = pd.concat(dfs, ignore_index=True, join="outer")
# # #         return df.to_string(index=False)

# # #     raise ValueError(f"Unsupported file type for {filename}")


# # # def clean_json_response(response_text: str) -> str:
# # #     """Clean and extract JSON from OpenAI response"""
# # #     # Remove markdown code blocks
# # #     cleaned = re.sub(r'```json\s*', '', response_text)
# # #     cleaned = re.sub(r'```\s*$', '', cleaned)
    
# # #     # Remove any text before the first [ or {
# # #     json_start = -1
# # #     for i, char in enumerate(cleaned):
# # #         if char in '[{':
# # #             json_start = i
# # #             break
    
# # #     if json_start != -1:
# # #         cleaned = cleaned[json_start:]
    
# # #     # Remove any text after the last ] or }
# # #     json_end = -1
# # #     for i in range(len(cleaned) - 1, -1, -1):
# # #         if cleaned[i] in ']}':
# # #             json_end = i + 1
# # #             break
    
# # #     if json_end != -1:
# # #         cleaned = cleaned[:json_end]
    
# # #     return cleaned.strip()


# # # def ensure_list_format(data) -> list:
# # #     """Ensure data is in list format"""
# # #     if isinstance(data, list):
# # #         return data
# # #     elif isinstance(data, dict):
# # #         return [data]  # Convert single object to list
# # #     else:
# # #         raise ValueError(f"Expected list or dict, got {type(data)}")


# # # def classify_payin(payin_str):
# # #     """
# # #     Converts Payin string (e.g., '50%') to float and classifies its range.
# # #     """
# # #     try:
# # #         # Handle various formats
# # #         payin_clean = str(payin_str).replace('%', '').replace(' ', '')
# # #         payin_value = float(payin_clean)
        
# # #         if payin_value <= 20:
# # #             category = "Payin Below 20%"
# # #         elif 21 <= payin_value <= 30:
# # #             category = "Payin 21% to 30%"
# # #         elif 31 <= payin_value <= 50:
# # #             category = "Payin 31% to 50%"
# # #         else:
# # #             category = "Payin from 51%"
# # #         return payin_value, category
# # #     except (ValueError, TypeError):
# # #         logger.warning(f"Could not parse payin value: {payin_str}")
# # #         return 0.0, "Payin Below 20%"


# # # def process_files(policy_file_bytes, policy_filename, policy_content_type,
# # #                  formula_file_bytes, formula_filename, formula_content_type, 
# # #                  company_name):
# # #     """Main processing function"""
# # #     try:
# # #         st.info("🔍 Extracting text from files...")
        
# # #         # Extract text with enhanced intelligence
# # #         extracted_text = extract_text_from_file(policy_file_bytes, policy_filename, policy_content_type, is_formula=False)
# # #         extracted_formula_text = extract_text_from_file(formula_file_bytes, formula_filename, formula_content_type, is_formula=True)

# # #         logger.info(f"Extracted text length: {len(extracted_text)}")
# # #         logger.info(f"Extracted formula text length: {len(extracted_formula_text)}")

# # #         st.success(f"✅ Text extracted successfully! Policy: {len(extracted_text)} chars, Formula: {len(extracted_formula_text)} chars")

# # #         # Parse policy data with enhanced segment identification
# # #         st.info("🧠 Parsing policy data with AI...")
        
# # #         parse_prompt = f"""
# # #         Analyze the following text, which contains insurance policy details.
# # #         Use your intelligence to identify and extract the data accurately.

# # #         Company Name: {company_name}

# # #         IMPORTANT: Always return a valid JSON array (list) of records, even if there's only one record.

# # #         Extract into JSON records with these exact fields:
# # #         - "Segment": identify LOB like TW, PVT CAR, CV, BUS, TAXI, MISD and policy type like TP, COMP, SAOD
# # #         - "Location": location information
# # #         - "Policy Type": policy type information
# # #         - "Payin": convert payout values to percentage format, e.g. 0.5 → 50%, 34 → 34%
# # #         - "Doable District": district information
# # #         - "Remarks": any additional information

# # #         Be intelligent in identifying segments even if the format varies.
# # #         Look for patterns like:
# # #         - TW (Two Wheeler) related segments
# # #         - PVT CAR (Private Car) segments  
# # #         - CV (Commercial Vehicle) segments
# # #         - BUS segments
# # #         - TAXI segments
# # #         - MISD (Miscellaneous) segments

# # #         If you cannot find specific information for any field, use "N/A" or reasonable defaults.
# # #         Note : If Policy is not defined or told , then consider it as COMP/TP
# # #         Text to analyze:
# # #         {extracted_text}
        
# # #         Return ONLY a valid JSON array, no other text.
# # #         """
        
# # #         response = client.chat.completions.create(
# # #             model="gpt-4o-mini",
# # #             messages=[
# # #                 {"role": "system", "content": "Extract policy data as JSON array. Always rename 'Payout' → 'Payin'. Normalize all Payin values to percentage format. Use intelligence to identify segments accurately. Return ONLY valid JSON array."},
# # #                 {"role": "user", "content": parse_prompt}
# # #             ],
# # #             temperature=0.1
# # #         )
        
# # #         parsed_json = response.choices[0].message.content.strip()
# # #         logger.info(f"Raw OpenAI response: {parsed_json[:500]}...")
        
# # #         # Clean the JSON response
# # #         cleaned_json = clean_json_response(parsed_json)
# # #         logger.info(f"Cleaned JSON: {cleaned_json[:500]}...")
        
# # #         try:
# # #             policy_data = json.loads(cleaned_json)
# # #             policy_data = ensure_list_format(policy_data)
# # #         except json.JSONDecodeError as e:
# # #             logger.error(f"JSON decode error: {str(e)}")
# # #             logger.error(f"Problematic JSON: {cleaned_json}")
            
# # #             # Fallback: create a basic structure
# # #             policy_data = [{
# # #                 "Segment": "Unknown",
# # #                 "Location": "N/A",
# # #                 "Policy Type": "N/A", 
# # #                 "Payin": "0%",
# # #                 "Doable District": "N/A",
# # #                 "Remarks": f"Error parsing data: {str(e)}"
# # #             }]

# # #         st.success(f"✅ Successfully parsed {len(policy_data)} policy records")

# # #         # Pre-classify Payin values
# # #         for record in policy_data:
# # #             try:
# # #                 payin_val, payin_cat = classify_payin(record.get('Payin', '0%'))
# # #                 record['Payin_Value'] = payin_val
# # #                 record['Payin_Category'] = payin_cat
# # #             except Exception as e:
# # #                 logger.warning(f"Error classifying payin for record: {record}, error: {str(e)}")
# # #                 record['Payin_Value'] = 0.0
# # #                 record['Payin_Category'] = "Payin Below 20%"

# # #         # Apply formulas with enhanced intelligence
# # #         st.info("🧮 Applying formulas and calculating payouts...")
        
# # #         formula_prompt = f"""
# # #         You are an intelligent insurance formula calculator with access to:
        
# # #         Company Name: {company_name}
# # #         Policy Data: {json.dumps(policy_data, indent=2)}
# # #         Formula Rules/Grid: {extracted_formula_text}

# # #         IMPORTANT GRID STRUCTURE RULES:
# # #         - BUS LOB has two segments: "SCHOOL BUS" and "STAFF BUS" - treat them separately
# # #         - TAXI LOB has one segment: "TAXI" with multiple payin categories
# # #         - PVT CAR has: "PVT CAR COMP + SAOD" and "PVT CAR TP"
# # #         - CV has: "Upto 2.5 GVW" and "All GVW & PCV 3W, GCV 3W"
# # #         - Each segment can have different company rules and payin category rules

# # #         TASK: Use your intelligence to:
# # #         1. Analyze each policy record's Segment and identify the correct LOB (TW, PVT CAR, CV, BUS, TAXI, MISD)
# # #         2. Within the LOB, identify the specific segment (e.g., for BUS: "SCHOOL BUS" vs "STAFF BUS")
# # #         3. Match the company name ({company_name}) with the formula grid:
# # #            - Look for exact matches first (DIGIT, BAJAJ, ICICI, SBI, TATA, RELIANCE)
# # #            - Handle variations (e.g., "Bajaj Allianz" → "BAJAJ", "SBI General" → "SBI")
# # #            - If no specific company match, use "Rest of Companies" or "All Companies"
# # #         4. Match the Payin category correctly:
# # #            - "Payin Below 20%" for payin ≤ 20%
# # #            - "Payin 21% to 30%" for payin 21-30%
# # #            - "Payin 31% to 50%" for payin 31-50%
# # #            - "Payin Above 50%" for payin > 50%
# # #         5. Apply the correct formula based on the grid:
# # #         These are the examples i am giving you:
# # #            - "90% of Payin" → multiply Payin_Value by 0.9
# # #            - "88% of Payin" → multiply Payin_Value by 0.88
# # #            - "-2%" → subtract 2 from Payin_Value
# # #            - "-3%" → subtract 3 from Payin_Value
# # #            - "-4%" → subtract 4 from Payin_Value
# # #            - "-5%" → subtract 5 from Payin_Value
# # #            - "Less 2 of Payin" → subtract 2 from Payin_Value
# # #         6. Calculate: Calculated Payout = Apply the formula to Payin_Value
# # #         7. Round down Calculated Payout to the nearest whole number
# # #         8. Format as percentage (e.g., 19 → "19%")

# # #         SPECIAL ATTENTION:
# # #         - For BUS policies, clearly distinguish between SCHOOL BUS and STAFF BUS
# # #         - For TAXI, all companies follow the same payin-category-based rules
# # #         - Pay attention to company-specific rules vs "All Companies" rules

# # #         Return ONLY a valid JSON array with all original fields plus:
# # #         - 'Calculated Payout' (as percentage string)
# # #         - 'Formula Used' (the exact rule applied)
# # #         - 'Rule Explanation' (detailed explanation: LOB → Segment → Company → Payin Category → Rule)
# # #         Note : Please don't make any calculation mistake while applying formula
# # #         """
        
# # #         formula_response = client.chat.completions.create(
# # #             model="gpt-4o-mini",
# # #             messages=[
# # #                 {"role": "system", "content": "You are an intelligent financial calculator. Use your intelligence to match policy data with formula grid rules accurately. Output ONLY valid JSON array."},
# # #                 {"role": "user", "content": formula_prompt}
# # #             ],
# # #             temperature=0.0
# # #         )
        
# # #         calc_json = formula_response.choices[0].message.content.strip()
# # #         logger.info(f"Raw formula response: {calc_json[:500]}...")
        
# # #         # Clean and parse formula response
# # #         calc_json_cleaned = clean_json_response(calc_json)
        
# # #         try:
# # #             calculated_data = json.loads(calc_json_cleaned)
# # #             calculated_data = ensure_list_format(calculated_data)
# # #         except json.JSONDecodeError as e:
# # #             logger.error(f"Formula JSON decode error: {str(e)}")
# # #             logger.error(f"Problematic formula JSON: {calc_json_cleaned}")
            
# # #             # Fallback: use original data with default calculated values
# # #             calculated_data = []
# # #             for record in policy_data:
# # #                 record_copy = record.copy()
# # #                 record_copy['Calculated Payout'] = f"{int(record.get('Payin_Value', 0))}%"
# # #                 record_copy['Formula Used'] = "Default (no formula applied)"
# # #                 record_copy['Rule Explanation'] = "Formula parsing failed, using original payin value"
# # #                 calculated_data.append(record_copy)

# # #         st.success(f"✅ Successfully calculated {len(calculated_data)} records with payouts")

# # #         # Create Excel file
# # #         st.info("📊 Creating Excel file...")
        
# # #         df_calc = pd.DataFrame(calculated_data)
# # #         output = BytesIO()
# # #         with pd.ExcelWriter(output, engine='openpyxl') as writer:
# # #             df_calc.to_excel(writer, sheet_name='Policy Data', startrow=2, index=False)
# # #             worksheet = writer.sheets['Policy Data']

# # #             headers = list(df_calc.columns)
# # #             for col_num, value in enumerate(headers, 1):
# # #                 worksheet.cell(row=3, column=col_num, value=value)
# # #                 worksheet.cell(row=3, column=col_num).font = worksheet.cell(row=3, column=col_num).font.copy(bold=True)

# # #             company_cell = worksheet.cell(row=1, column=1, value=company_name)
# # #             worksheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
# # #             company_cell.font = company_cell.font.copy(bold=True, size=14)
# # #             company_cell.alignment = company_cell.alignment.copy(horizontal='center')

# # #             title_cell = worksheet.cell(row=2, column=1, value='Policy Data with Payin and Calculated Payouts')
# # #             worksheet.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(headers))
# # #             title_cell.font = title_cell.font.copy(bold=True, size=12)
# # #             title_cell.alignment = title_cell.alignment.copy(horizontal='center')

# # #         output.seek(0)
# # #         excel_data = output.read()

# # #         return {
# # #             "extracted_text": extracted_text,
# # #             "extracted_formula_text": extracted_formula_text,
# # #             "parsed_data": policy_data,
# # #             "calculated_data": calculated_data,
# # #             "excel_data": excel_data,
# # #             "df_calc": df_calc
# # #         }

# # #     except Exception as e:
# # #         logger.error(f"Unexpected error: {str(e)}", exc_info=True)
# # #         raise Exception(f"An error occurred: {str(e)}")


# # # # Streamlit App
# # # def main():
# # #     st.set_page_config(
# # #         page_title="Insurance Policy Processing", 
# # #         page_icon="📋", 
# # #         layout="wide"
# # #     )
    
# # #     st.title("🏢 Insurance Policy Processing System")
# # #     st.markdown("---")
    
# # #     # Sidebar for inputs
# # #     with st.sidebar:
# # #         st.header("📁 File Upload")
        
# # #         # Company name input
# # #         company_name = st.text_input(
# # #             "Company Name", 
# # #             value="Unknown Company",
# # #             help="Enter the insurance company name"
# # #         )
        
# # #         # Policy file upload
# # #         policy_file = st.file_uploader(
# # #             "📄 Upload Policy File",
# # #             type=['pdf', 'png', 'jpg', 'jpeg', 'txt', 'csv', 'xlsx', 'xls'],
# # #             help="Upload your insurance policy document"
# # #         )
        
# # #         # Formula file upload
# # #         formula_file = st.file_uploader(
# # #             "📊 Upload Formula/Rules File",
# # #             type=['pdf', 'png', 'jpg', 'jpeg', 'txt', 'csv', 'xlsx', 'xls'],
# # #             help="Upload your formula rules or grid document"
# # #         )
        
# # #         # Process button
# # #         process_button = st.button(
# # #             "🚀 Process Files", 
# # #             type="primary",
# # #             disabled=not (policy_file and formula_file)
# # #         )
    
# # #     # Main content area
# # #     if not policy_file or not formula_file:
# # #         st.info("👆 Please upload both policy and formula files to begin processing.")
# # #         st.markdown("""
# # #         ### 📋 Instructions:
# # #         1. **Company Name**: Enter the insurance company name
# # #         2. **Policy File**: Upload the document containing policy data (PDF, Image, Excel, CSV, etc.)
# # #         3. **Formula File**: Upload the document containing formula rules and grids
# # #         4. **Process**: Click the process button to extract data and calculate payouts
        
# # #         ### 🎯 Features:
# # #         - **Multi-format Support**: PDF, Images, Excel, CSV, Text files
# # #         - **AI-Powered Extraction**: Uses GPT-4 for intelligent text extraction
# # #         - **Smart Formula Application**: Automatically applies rules based on segments and companies
# # #         - **Excel Export**: Download processed data as formatted Excel file
# # #         """)
# # #         return
    
# # #     if process_button:
# # #         try:
# # #             # Read file contents
# # #             policy_file_bytes = policy_file.read()
# # #             formula_file_bytes = formula_file.read()
            
# # #             # Process files
# # #             with st.spinner("Processing files... This may take a few moments."):
# # #                 results = process_files(
# # #                     policy_file_bytes, policy_file.name, policy_file.type,
# # #                     formula_file_bytes, formula_file.name, formula_file.type,
# # #                     company_name
# # #                 )
            
# # #             st.success("🎉 Processing completed successfully!")
            
# # #             # Display results in tabs
# # #             tab1, tab2, tab3, tab4, tab5 = st.tabs([
# # #                 "📊 Final Results", 
# # #                 "📝 Extracted Text", 
# # #                 "🧾 Parsed Data", 
# # #                 "🧮 Calculated Data",
# # #                 "📥 Download"
# # #             ])
            
# # #             with tab1:
# # #                 st.subheader("📊 Final Processed Data")
# # #                 st.dataframe(results["df_calc"], use_container_width=True)
                
# # #                 # Summary statistics
# # #                 col1, col2, col3, col4 = st.columns(4)
# # #                 with col1:
# # #                     st.metric("Total Records", len(results["calculated_data"]))
# # #                 with col2:
# # #                     avg_payin = sum([r.get('Payin_Value', 0) for r in results["calculated_data"]]) / len(results["calculated_data"])
# # #                     st.metric("Avg Payin", f"{avg_payin:.1f}%")
# # #                 with col3:
# # #                     segments = set([r.get('Segment', 'N/A') for r in results["calculated_data"]])
# # #                     st.metric("Unique Segments", len(segments))
# # #                 with col4:
# # #                     st.metric("Company", company_name)
            
# # #             with tab2:
# # #                 st.subheader("📝 Extracted Text")
                
# # #                 col1, col2 = st.columns(2)
# # #                 with col1:
# # #                     st.write("**Policy File Text:**")
# # #                     st.text_area(
# # #                         "Policy Text", 
# # #                         results["extracted_text"], 
# # #                         height=300,
# # #                         key="policy_text"
# # #                     )
                
# # #                 with col2:
# # #                     st.write("**Formula File Text:**")
# # #                     st.text_area(
# # #                         "Formula Text", 
# # #                         results["extracted_formula_text"], 
# # #                         height=300,
# # #                         key="formula_text"
# # #                     )
            
# # #             with tab3:
# # #                 st.subheader("🧾 Parsed Policy Data")
# # #                 st.json(results["parsed_data"])
            
# # #             with tab4:
# # #                 st.subheader("🧮 Calculated Data with Formulas")
# # #                 st.json(results["calculated_data"])
            
# # #             with tab5:
# # #                 st.subheader("📥 Download Results")
                
# # #                 # Excel download
# # #                 st.download_button(
# # #                     label="📊 Download Excel File",
# # #                     data=results["excel_data"],
# # #                     file_name=f"{company_name}_processed_policies.xlsx",
# # #                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
# # #                 )
                
# # #                 # JSON download
# # #                 json_data = json.dumps(results["calculated_data"], indent=2)
# # #                 st.download_button(
# # #                     label="📄 Download JSON Data",
# # #                     data=json_data,
# # #                     file_name=f"{company_name}_processed_data.json",
# # #                     mime="application/json"
# # #                 )
                
# # #                 st.info("💡 The Excel file contains formatted data with company header and calculated payouts.")
                
# # #         except Exception as e:
# # #             st.error(f"❌ Error processing files: {str(e)}")
# # #             st.exception(e)


# # # if __name__ == "__main__":
# # #     main()

# # import streamlit as st
# # import pandas as pd
# # from io import BytesIO
# # import base64
# # import json
# # import os
# # from dotenv import load_dotenv
# # import logging
# # import re

# # # Check if required packages are available
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
# #     st.error("OPENAI_API_KEY environment variable not set. Please set it in your .env file or environment variables.")
# #     st.stop()

# # # Initialize OpenAI client
# # try:
# #     client = OpenAI(api_key=OPENAI_API_KEY)
# # except Exception as e:
# #     st.error(f"Failed to initialize OpenAI client: {str(e)}")
# #     st.stop()


# # def extract_text_from_pdf_file(pdf_bytes: bytes) -> str:
# #     """Extract text from PDF using multiple methods"""
# #     extracted_text = ""
    
# #     # Method 1: Try pdfplumber first (most accurate)
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
    
# #     # Method 2: Try PyPDF2 as fallback
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
    
# #     # Method 3: Return empty string if all methods fail
# #     logger.warning("All PDF text extraction methods failed")
# #     return ""


# # def extract_text_from_file(file_bytes: bytes, filename: str, content_type: str, is_formula: bool = False) -> str:
# #     """Extract text from uploaded file"""
# #     file_extension = filename.split('.')[-1].lower() if '.' in filename else ''
# #     file_type = content_type if content_type else file_extension

# #     # Image-based extraction with enhanced OCR
# #     image_extensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']
# #     if file_extension in image_extensions or file_type.startswith('image/'):
# #         image_base64 = base64.b64encode(file_bytes).decode('utf-8')
        
# #         if is_formula:
# #             prompt = """Extract all formulas, rules and grid information accurately from this image. 
# #             Focus on identifying:
# #             - LOB segments (TW, PVT CAR, CV, BUS, TAXI, MISD)
# #             - Policy types (TP, COMP, SAOD, etc.)
# #             - Company names and their specific rules
# #             - Percentage deductions and formulas
# #             - Payin categories and their corresponding rates
# #             Extract everything exactly as shown in the grid/table format."""
# #         else:
# #             prompt = """Extract all insurance policy text accurately from this image using OCR.
# #             Pay special attention to identifying:
# #             - Segment information (like TW, PVT CAR, CV, BUS, TAXI, MISD)
# #             - Company names
# #             - Policy types
# #             - Payin/Payout percentages
# #             - Any numerical values
# #             - Location information
# #             Extract all text exactly as it appears."""
            
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
# #         return response.choices[0].message.content.strip()

# #     # Enhanced PDF extraction
# #     if file_extension == 'pdf':
# #         # First try to extract text directly from PDF
# #         pdf_text = extract_text_from_pdf_file(file_bytes)
        
# #         if pdf_text and len(pdf_text.strip()) > 50:
# #             # If we got good text extraction, use it
# #             logger.info("Using direct PDF text extraction")
# #             return pdf_text
# #         else:
# #             # Fallback to OpenAI vision for PDF
# #             logger.info("Using OpenAI vision for PDF")
# #             pdf_base64 = base64.b64encode(file_bytes).decode("utf-8")
            
# #             if is_formula:
# #                 prompt = """Extract formulas, rules, and grid information from this PDF.
# #                 Look for LOB segments, company-specific rules, percentage deductions, and payin categories."""
# #             else:
# #                 prompt = """Extract insurance policy details from this PDF.
# #                 Focus on identifying segments, company names, policy types, and payout percentages."""
                
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
# #             return response.choices[0].message.content.strip()

# #     # Text files
# #     if file_extension == 'txt':
# #         return file_bytes.decode('utf-8', errors='ignore')

# #     # CSV files
# #     if file_extension == 'csv':
# #         df = pd.read_csv(BytesIO(file_bytes), encoding='utf-8', errors='ignore')
# #         return df.to_string()

# #     # Excel files
# #     if file_extension in ['xlsx', 'xls']:
# #         all_sheets = pd.read_excel(BytesIO(file_bytes), sheet_name=None)
# #         dfs = []
# #         for sheet_name, df_sheet in all_sheets.items():
# #             df_sheet["Source_Sheet"] = sheet_name
# #             dfs.append(df_sheet)
# #         df = pd.concat(dfs, ignore_index=True, join="outer")
# #         return df.to_string(index=False)

# #     raise ValueError(f"Unsupported file type for {filename}")


# # def clean_json_response(response_text: str) -> str:
# #     """Clean and extract JSON from OpenAI response"""
# #     # Remove markdown code blocks
# #     cleaned = re.sub(r'```json\s*', '', response_text)
# #     cleaned = re.sub(r'```\s*$', '', cleaned)
    
# #     # Remove any text before the first [ or {
# #     json_start = -1
# #     for i, char in enumerate(cleaned):
# #         if char in '[{':
# #             json_start = i
# #             break
    
# #     if json_start != -1:
# #         cleaned = cleaned[json_start:]
    
# #     # Remove any text after the last ] or }
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
# #         return [data]  # Convert single object to list
# #     else:
# #         raise ValueError(f"Expected list or dict, got {type(data)}")


# # def classify_payin(payin_str):
# #     """
# #     Converts Payin string (e.g., '50%') to float and classifies its range.
# #     """
# #     try:
# #         # Handle various formats
# #         payin_clean = str(payin_str).replace('%', '').replace(' ', '')
# #         payin_value = float(payin_clean)
        
# #         if payin_value <= 20:
# #             category = "Payin Below 20%"
# #         elif 21 <= payin_value <= 30:
# #             category = "Payin 21% to 30%"
# #         elif 31 <= payin_value <= 50:
# #             category = "Payin 31% to 50%"
# #         else:
# #             category = "Payin from 51%"
# #         return payin_value, category
# #     except (ValueError, TypeError):
# #         logger.warning(f"Could not parse payin value: {payin_str}")
# #         return 0.0, "Payin Below 20%"


# # def process_files(policy_file_bytes, policy_filename, policy_content_type,
# #                  formula_file_bytes, formula_filename, formula_content_type, 
# #                  company_name):
# #     """Main processing function"""
# #     try:
# #         st.info("🔍 Extracting text from files...")
        
# #         # Extract text with enhanced intelligence
# #         extracted_text = extract_text_from_file(policy_file_bytes, policy_filename, policy_content_type, is_formula=False)
# #         extracted_formula_text = extract_text_from_file(formula_file_bytes, formula_filename, formula_content_type, is_formula=True)

# #         logger.info(f"Extracted text length: {len(extracted_text)}")
# #         logger.info(f"Extracted formula text length: {len(extracted_formula_text)}")

# #         st.success(f"✅ Text extracted successfully! Policy: {len(extracted_text)} chars, Formula: {len(extracted_formula_text)} chars")

# #         # Parse policy data with enhanced segment identification
# #         st.info("🧠 Parsing policy data with AI...")
        
# #         parse_prompt = f"""
# #         Analyze the following text, which contains insurance policy details.
# #         Use your intelligence to identify and extract the data accurately.

# #         Company Name: {company_name}

# #         IMPORTANT: Always return a valid JSON array (list) of records, even if there's only one record.

# #         Extract into JSON records with these exact fields:
# #         - "Segment": identify LOB like TW, PVT CAR, CV, BUS, TAXI, MISD and policy type like TP, COMP, SAOD
# #         - "Location": location information
# #         - "Policy Type": policy type information
# #         - "Payin": convert payout values to percentage format, e.g. 0.5 → 50%, 34 → 34%
# #         - "Doable District": district information
# #         - "Remarks": any additional information INCLUDING vehicle make information

# #         REMARKS FIELD INSTRUCTIONS:
# #         - Extract ALL additional information that doesn't fit in other specific fields
# #         - Include any notes, conditions, special instructions, exclusions, etc.
# #         - ALSO look for vehicle make information in the text (e.g., "Make – Tata, Maruti" or similar patterns)
# #         - Vehicle makes include brands like: TATA, MARUTI, SUZUKI, HYUNDAI, HONDA, TOYOTA, MAHINDRA, BAJAJ, HERO, TVS, YAMAHA, etc.
# #         - If vehicle makes are found, include them as part of the remarks: "Vehicle Makes: [make1, make2, ...]; [other remarks]"
# #         - If no other remarks exist, just include: "Vehicle Makes: [make1, make2, ...]"
# #         - Combine all relevant additional information into the remarks field

# #         Be intelligent in identifying segments even if the format varies.
# #         Look for patterns like:
# #         - TW (Two Wheeler) related segments
# #         - PVT CAR (Private Car) segments  
# #         - CV (Commercial Vehicle) segments
# #         - BUS segments
# #         - TAXI segments
# #         - MISD (Miscellaneous) segments

# #         If you cannot find specific information for any field, use "N/A" or reasonable defaults.
# #         Note : If Policy is not defined or told anything from Policy types (TP, COMP, SAOD, etc.), then consider it as COMP/TP and add in that
# #         Text to analyze:
# #         {extracted_text}
        
# #         Return ONLY a valid JSON array, no other text.
# #         """
        
# #         response = client.chat.completions.create(
# #             model="gpt-4o-mini",
# #             messages=[
# #                 {"role": "system", "content": "Extract policy data as JSON array. Always rename 'Payout' → 'Payin'. Normalize all Payin values to percentage format. Use intelligence to identify segments accurately. Extract ALL additional information for remarks including vehicle makes, notes, conditions, exclusions, etc. Return ONLY valid JSON array."},
# #                 {"role": "user", "content": parse_prompt}
# #             ],
# #             temperature=0.1
# #         )
        
# #         parsed_json = response.choices[0].message.content.strip()
# #         logger.info(f"Raw OpenAI response: {parsed_json[:500]}...")
        
# #         # Clean the JSON response
# #         cleaned_json = clean_json_response(parsed_json)
# #         logger.info(f"Cleaned JSON: {cleaned_json[:500]}...")
        
# #         try:
# #             policy_data = json.loads(cleaned_json)
# #             policy_data = ensure_list_format(policy_data)
# #         except json.JSONDecodeError as e:
# #             logger.error(f"JSON decode error: {str(e)}")
# #             logger.error(f"Problematic JSON: {cleaned_json}")
            
# #             # Fallback: create a basic structure
# #             policy_data = [{
# #                 "Segment": "Unknown",
# #                 "Location": "N/A",
# #                 "Policy Type": "N/A", 
# #                 "Payin": "0%",
# #                 "Doable District": "N/A",
# #                 "Remarks": f"Error parsing data: {str(e)}"
# #             }]

# #         st.success(f"✅ Successfully parsed {len(policy_data)} policy records")

# #         # Pre-classify Payin values
# #         for record in policy_data:
# #             try:
# #                 payin_val, payin_cat = classify_payin(record.get('Payin', '0%'))
# #                 record['Payin_Value'] = payin_val
# #                 record['Payin_Category'] = payin_cat
# #             except Exception as e:
# #                 logger.warning(f"Error classifying payin for record: {record}, error: {str(e)}")
# #                 record['Payin_Value'] = 0.0
# #                 record['Payin_Category'] = "Payin Below 20%"

# #         # Apply formulas with enhanced intelligence
# #         st.info("🧮 Applying formulas and calculating payouts...")
        
# #         formula_prompt = f"""
# #         You are an intelligent insurance formula calculator with access to:
        
# #         Company Name: {company_name}
# #         Policy Data: {json.dumps(policy_data, indent=2)}
# #         Formula Rules/Grid: {extracted_formula_text}

# #         IMPORTANT GRID STRUCTURE RULES:
# #         - BUS LOB has two segments: "SCHOOL BUS" and "STAFF BUS" - treat them separately
# #         - TAXI LOB has one segment: "TAXI" with multiple payin categories
# #         - PVT CAR has: "PVT CAR COMP + SAOD" and "PVT CAR TP"
# #         - CV has: "Upto 2.5 GVW" and "All GVW & PCV 3W, GCV 3W"
# #         - Each segment can have different company rules and payin category rules

# #         TASK: Use your intelligence to:
# #         1. Analyze each policy record's Segment and identify the correct LOB (TW, PVT CAR, CV, BUS, TAXI, MISD)
# #         2. Within the LOB, identify the specific segment (e.g., for BUS: "SCHOOL BUS" vs "STAFF BUS")
# #         3. Match the company name ({company_name}) with the formula grid:
# #            - Look for exact matches first (DIGIT, BAJAJ, ICICI, SBI, TATA, RELIANCE)
# #            - Handle variations (e.g., "Bajaj Allianz" → "BAJAJ", "SBI General" → "SBI")
# #            - If no specific company match, use "Rest of Companies" or "All Companies"
# #         4. Match the Payin category correctly:
# #            - "Payin Below 20%" for payin ≤ 20%
# #            - "Payin 21% to 30%" for payin 21-30%
# #            - "Payin 31% to 50%" for payin 31-50%
# #            - "Payin Above 50%" for payin > 50%
# #         5. Apply the correct formula based on the grid:
# #         These are the examples i am giving you:
# #            - "90% of Payin" → multiply Payin_Value by 0.9
# #            - "88% of Payin" → multiply Payin_Value by 0.88
# #            - "-2%" → subtract 2 from Payin_Value
# #            - "-3%" → subtract 3 from Payin_Value
# #            - "-4%" → subtract 4 from Payin_Value
# #            - "-5%" → subtract 5 from Payin_Value
# #            - "Less 2 of Payin" → subtract 2 from Payin_Value
# #         6. Calculate: Calculated Payout = Apply the formula to Payin_Value
# #         7. Round down Calculated Payout to the nearest whole number
# #         8. Format as percentage (e.g., 19 → "19%")

# #         SPECIAL ATTENTION:
# #         - For BUS policies, clearly distinguish between SCHOOL BUS and STAFF BUS
# #         - For TAXI, all companies follow the same payin-category-based rules
# #         - Pay attention to company-specific rules vs "All Companies" rules

# #         Return ONLY a valid JSON array with all original fields plus:
# #         - 'Calculated Payout' (as percentage string)
# #         - 'Formula Used' (the exact rule applied)
# #         - 'Rule Explanation' (detailed explanation: LOB → Segment → Company → Payin Category → Rule)
# #         Note : Please don't make any calculation mistake while applying formula
# #         """
        
# #         formula_response = client.chat.completions.create(
# #             model="gpt-4o-mini",
# #             messages=[
# #                 {"role": "system", "content": "You are an intelligent financial calculator. Use your intelligence to match policy data with formula grid rules accurately. Output ONLY valid JSON array."},
# #                 {"role": "user", "content": formula_prompt}
# #             ],
# #             temperature=0.0
# #         )
        
# #         calc_json = formula_response.choices[0].message.content.strip()
# #         logger.info(f"Raw formula response: {calc_json[:500]}...")
        
# #         # Clean and parse formula response
# #         calc_json_cleaned = clean_json_response(calc_json)
        
# #         try:
# #             calculated_data = json.loads(calc_json_cleaned)
# #             calculated_data = ensure_list_format(calculated_data)
# #         except json.JSONDecodeError as e:
# #             logger.error(f"Formula JSON decode error: {str(e)}")
# #             logger.error(f"Problematic formula JSON: {calc_json_cleaned}")
            
# #             # Fallback: use original data with default calculated values
# #             calculated_data = []
# #             for record in policy_data:
# #                 record_copy = record.copy()
# #                 record_copy['Calculated Payout'] = f"{int(record.get('Payin_Value', 0))}%"
# #                 record_copy['Formula Used'] = "Default (no formula applied)"
# #                 record_copy['Rule Explanation'] = "Formula parsing failed, using original payin value"
# #                 calculated_data.append(record_copy)

# #         st.success(f"✅ Successfully calculated {len(calculated_data)} records with payouts")

# #         # Create Excel file
# #         st.info("📊 Creating Excel file...")
        
# #         df_calc = pd.DataFrame(calculated_data)
# #         output = BytesIO()
# #         with pd.ExcelWriter(output, engine='openpyxl') as writer:
# #             df_calc.to_excel(writer, sheet_name='Policy Data', startrow=2, index=False)
# #             worksheet = writer.sheets['Policy Data']

# #             headers = list(df_calc.columns)
# #             for col_num, value in enumerate(headers, 1):
# #                 worksheet.cell(row=3, column=col_num, value=value)
# #                 worksheet.cell(row=3, column=col_num).font = worksheet.cell(row=3, column=col_num).font.copy(bold=True)

# #             company_cell = worksheet.cell(row=1, column=1, value=company_name)
# #             worksheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
# #             company_cell.font = company_cell.font.copy(bold=True, size=14)
# #             company_cell.alignment = company_cell.alignment.copy(horizontal='center')

# #             title_cell = worksheet.cell(row=2, column=1, value='Policy Data with Payin and Calculated Payouts')
# #             worksheet.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(headers))
# #             title_cell.font = title_cell.font.copy(bold=True, size=12)
# #             title_cell.alignment = title_cell.alignment.copy(horizontal='center')

# #         output.seek(0)
# #         excel_data = output.read()

# #         return {
# #             "extracted_text": extracted_text,
# #             "extracted_formula_text": extracted_formula_text,
# #             "parsed_data": policy_data,
# #             "calculated_data": calculated_data,
# #             "excel_data": excel_data,
# #             "df_calc": df_calc
# #         }

# #     except Exception as e:
# #         logger.error(f"Unexpected error: {str(e)}", exc_info=True)
# #         raise Exception(f"An error occurred: {str(e)}")


# # # Streamlit App
# # def main():
# #     st.set_page_config(
# #         page_title="Insurance Policy Processing", 
# #         page_icon="📋", 
# #         layout="wide"
# #     )
    
# #     st.title("🏢 Insurance Policy Processing System")
# #     st.markdown("---")
    
# #     # Sidebar for inputs
# #     with st.sidebar:
# #         st.header("📁 File Upload")
        
# #         # Company name input
# #         company_name = st.text_input(
# #             "Company Name", 
# #             value="Unknown Company",
# #             help="Enter the insurance company name"
# #         )
        
# #         # Policy file upload
# #         policy_file = st.file_uploader(
# #             "📄 Upload Policy File",
# #             type=['pdf', 'png', 'jpg', 'jpeg', 'txt', 'csv', 'xlsx', 'xls'],
# #             help="Upload your insurance policy document"
# #         )
        
# #         # Formula file upload
# #         formula_file = st.file_uploader(
# #             "📊 Upload Formula/Rules File",
# #             type=['pdf', 'png', 'jpg', 'jpeg', 'txt', 'csv', 'xlsx', 'xls'],
# #             help="Upload your formula rules or grid document"
# #         )
        
# #         # Process button
# #         process_button = st.button(
# #             "🚀 Process Files", 
# #             type="primary",
# #             disabled=not (policy_file and formula_file)
# #         )
    
# #     # Main content area
# #     if not policy_file or not formula_file:
# #         st.info("👆 Please upload both policy and formula files to begin processing.")
# #         st.markdown("""
# #         ### 📋 Instructions:
# #         1. **Company Name**: Enter the insurance company name
# #         2. **Policy File**: Upload the document containing policy data (PDF, Image, Excel, CSV, etc.)
# #         3. **Formula File**: Upload the document containing formula rules and grids
# #         4. **Process**: Click the process button to extract data and calculate payouts
        
# #         ### 🎯 Features:
# #         - **Multi-format Support**: PDF, Images, Excel, CSV, Text files
# #         - **AI-Powered Extraction**: Uses GPT-4 for intelligent text extraction
# #         - **Vehicle Make Detection**: Automatically detects and extracts vehicle make information
# #         - **Smart Formula Application**: Automatically applies rules based on segments and companies
# #         - **Excel Export**: Download processed data as formatted Excel file
# #         """)
# #         return
    
# #     if process_button:
# #         try:
# #             # Read file contents
# #             policy_file_bytes = policy_file.read()
# #             formula_file_bytes = formula_file.read()
            
# #             # Process files
# #             with st.spinner("Processing files... This may take a few moments."):
# #                 results = process_files(
# #                     policy_file_bytes, policy_file.name, policy_file.type,
# #                     formula_file_bytes, formula_file.name, formula_file.type,
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
# #                 st.dataframe(results["df_calc"], use_container_width=True)
                
# #                 # Summary statistics
# #                 col1, col2, col3, col4 = st.columns(4)
# #                 with col1:
# #                     st.metric("Total Records", len(results["calculated_data"]))
# #                 with col2:
# #                     avg_payin = sum([r.get('Payin_Value', 0) for r in results["calculated_data"]]) / len(results["calculated_data"])
# #                     st.metric("Avg Payin", f"{avg_payin:.1f}%")
# #                 with col3:
# #                     segments = set([r.get('Segment', 'N/A') for r in results["calculated_data"]])
# #                     st.metric("Unique Segments", len(segments))
# #                 with col4:
# #                     st.metric("Company", company_name)
            
# #             with tab2:
# #                 st.subheader("📝 Extracted Text")
                
# #                 col1, col2 = st.columns(2)
# #                 with col1:
# #                     st.write("**Policy File Text:**")
# #                     st.text_area(
# #                         "Policy Text", 
# #                         results["extracted_text"], 
# #                         height=300,
# #                         key="policy_text"
# #                     )
                
# #                 with col2:
# #                     st.write("**Formula File Text:**")
# #                     st.text_area(
# #                         "Formula Text", 
# #                         results["extracted_formula_text"], 
# #                         height=300,
# #                         key="formula_text"
# #                     )
            
# #             with tab3:
# #                 st.subheader("🧾 Parsed Policy Data")
# #                 st.json(results["parsed_data"])
            
# #             with tab4:
# #                 st.subheader("🧮 Calculated Data with Formulas")
# #                 st.json(results["calculated_data"])
            
# #             with tab5:
# #                 st.subheader("📥 Download Results")
                
# #                 # Excel download
# #                 st.download_button(
# #                     label="📊 Download Excel File",
# #                     data=results["excel_data"],
# #                     file_name=f"{company_name}_processed_policies.xlsx",
# #                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
# #                 )
                
# #                 # JSON download
# #                 json_data = json.dumps(results["calculated_data"], indent=2)
# #                 st.download_button(
# #                     label="📄 Download JSON Data",
# #                     data=json_data,
# #                     file_name=f"{company_name}_processed_data.json",
# #                     mime="application/json"
# #                 )
                
# #                 st.info("💡 The Excel file contains formatted data with company header and calculated payouts.")
                
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


# def extract_text_from_file(file_bytes: bytes, filename: str, content_type: str, is_formula: bool = False) -> str:
#     """Extract text from uploaded file"""
#     file_extension = filename.split('.')[-1].lower() if '.' in filename else ''
#     file_type = content_type if content_type else file_extension

#     # Image-based extraction with enhanced OCR
#     image_extensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']
#     if file_extension in image_extensions or file_type.startswith('image/'):
#         image_base64 = base64.b64encode(file_bytes).decode('utf-8')
        
#         if is_formula:
#             prompt = """Extract all formulas, rules and grid information accurately from this image. 
#             Focus on identifying:
#             - LOB segments (TW, PVT CAR, CV, BUS, TAXI, MISD)
#             - Policy types (TP, COMP, SAOD, etc.)
#             - Company names and their specific rules
#             - Percentage deductions and formulas
#             - Payin categories and their corresponding rates
#             Extract everything exactly as shown in the grid/table format."""
#         else:
#             prompt = """Extract all insurance policy text accurately from this image using OCR.
#             Pay special attention to identifying:
#             - Segment information (like TW, PVT CAR, CV, BUS, TAXI, MISD)
#             - Company names
#             - Policy types
#             - Payin/Payout percentages
#             - Any numerical values
#             - Location information
#             Extract all text exactly as it appears."""
            
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
            
#             if is_formula:
#                 prompt = """Extract formulas, rules, and grid information from this PDF.
#                 Look for LOB segments, company-specific rules, percentage deductions, and payin categories."""
#             else:
#                 prompt = """Extract insurance policy details from this PDF.
#                 Focus on identifying segments, company names, policy types, and payout percentages."""
                
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
#             category = "Payin from 51%"
#         return payin_value, category
#     except (ValueError, TypeError):
#         logger.warning(f"Could not parse payin value: {payin_str}")
#         return 0.0, "Payin Below 20%"


# def process_files(policy_file_bytes, policy_filename, policy_content_type,
#                  formula_file_bytes, formula_filename, formula_content_type, 
#                  company_name):
#     """Main processing function"""
#     try:
#         st.info("🔍 Extracting text from files...")
        
#         # Extract text with enhanced intelligence
#         extracted_text = extract_text_from_file(policy_file_bytes, policy_filename, policy_content_type, is_formula=False)
#         extracted_formula_text = extract_text_from_file(formula_file_bytes, formula_filename, formula_content_type, is_formula=True)

#         logger.info(f"Extracted text length: {len(extracted_text)}")
#         logger.info(f"Extracted formula text length: {len(extracted_formula_text)}")

#         st.success(f"✅ Text extracted successfully! Policy: {len(extracted_text)} chars, Formula: {len(extracted_formula_text)} chars")

#         # Parse policy data with enhanced segment identification
#         st.info("🧠 Parsing policy data with AI...")
        
#         parse_prompt = f"""
#         Analyze the following text, which contains insurance policy details.
#         Use your intelligence to identify and extract the data accurately.

#         Company Name: {company_name}

#         IMPORTANT: Always return a valid JSON array (list) of records, even if there's only one record.

#         Extract into JSON records with these exact fields:
#         - "Segment": identify LOB like TW, PVT CAR, CV, BUS, TAXI, MISD and policy type like TP, COMP, SAOD
#         - "Location": location information
#         - "Policy Type": policy type information
#         - "Payin": convert payout values to percentage format, e.g. 0.5 → 50%, 34 → 34%
#         - "Doable District": district information
#         - "Remarks": any additional information INCLUDING vehicle make information

#         ENHANCED REMARKS FIELD INSTRUCTIONS:
#         - Extract ALL additional information that doesn't fit in other specific fields
#         - Include any notes, conditions, special instructions, exclusions, etc.
#         - ALSO look for vehicle make information in the text (e.g., "Make – Tata, Maruti" or similar patterns)
#         - Vehicle makes include brands like: TATA, MARUTI, SUZUKI, HYUNDAI, HONDA, TOYOTA, MAHINDRA, BAJAJ, HERO, TVS, YAMAHA, etc.
#         - ALSO look for AGE information (e.g., ">5 years", "Age: 3 years", "Below 5 years", etc.)
#         - ALSO look for TRANSACTION information (e.g., "New", "Old", "Renewal", "Fresh", "Transaction: New/Old", etc.)
#         - If vehicle makes are found, include them: "Vehicle Makes: [make1, make2, ...]"
#         - If age information is found, include it: "Age: [age_info]"
#         - If transaction type is found, include it: "Transaction: [New/Old/Renewal]" and if its not found then dont add in the remarks column
#         - Combine all relevant additional information into the remarks field, separated by semicolons
#         - Example format: "Vehicle Makes: Tata, Maruti; Age: >5 years; Transaction: Old; [other remarks]"
#         - If no other remarks exist, just include the available information

#         Be intelligent in identifying segments even if the format varies.
#         Look for patterns like:
#         - TW (Two Wheeler) related segments
#         - PVT CAR (Private Car) segments  
#         - CV (Commercial Vehicle) segments
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
#                 {"role": "system", "content": "Extract policy data as JSON array. Always rename 'Payout' → 'Payin'. Normalize all Payin values to percentage format. Use intelligence to identify segments accurately. Extract ALL additional information for remarks including vehicle makes, age information, transaction type (New/Old/Renewal), notes, conditions, exclusions, etc. Return ONLY valid JSON array."},
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

#         # Apply formulas with enhanced intelligence
#         st.info("🧮 Applying formulas and calculating payouts...")
        
#         formula_prompt = f"""
#         You are an intelligent insurance formula calculator with access to:
        
#         Company Name: {company_name}
#         Policy Data: {json.dumps(policy_data, indent=2)}
#         Formula Rules/Grid: {extracted_formula_text}

#         IMPORTANT GRID STRUCTURE RULES:
#         - BUS LOB has two segments: "SCHOOL BUS" and "STAFF BUS" - treat them separately
#         - TAXI LOB has one segment: "TAXI" with multiple payin categories
#         - PVT CAR has: "PVT CAR COMP + SAOD" and "PVT CAR TP"
#         - CV has: "Upto 2.5 GVW" and "All GVW & PCV 3W, GCV 3W"
#         - Each segment can have different company rules and payin category rules

#         TASK: Use your intelligence to:
#         1. Analyze each policy record's Segment and identify the correct LOB (TW, PVT CAR, CV, BUS, TAXI, MISD)
#         2. Within the LOB, identify the specific segment (e.g., for BUS: "SCHOOL BUS" vs "STAFF BUS")
#         3. Match the company name ({company_name}) with the formula grid:
#            - Look for exact matches first (DIGIT, BAJAJ, ICICI, SBI, TATA, RELIANCE)
#            - Handle variations (e.g., "Bajaj Allianz" → "BAJAJ", "SBI General" → "SBI")
#            - If no specific company match, use "Rest of Companies" or "All Companies"
#         4. Match the Payin category correctly:
#            - "Payin Below 20%" for payin ≤ 20%
#            - "Payin 21% to 30%" for payin 21-30%
#            - "Payin 31% to 50%" for payin 31-50%
#            - "Payin Above 50%" for payin > 50%
#         5. Apply the correct formula based on the grid:
#         These are the examples i am giving you:
#            - "90% of Payin" → multiply Payin_Value by 0.9
#            - "88% of Payin" → multiply Payin_Value by 0.88
#            - "-2%" → subtract 2 from Payin_Value
#            - "-3%" → subtract 3 from Payin_Value
#            - "-4%" → subtract 4 from Payin_Value
#            - "-5%" → subtract 5 from Payin_Value
#            - "Less 2 of Payin" → subtract 2 from Payin_Value
#         6. Calculate: Calculated Payout = Apply the formula to Payin_Value
#         7. Round down Calculated Payout to the nearest whole number
#         8. Format as percentage (e.g., 19 → "19%")

#         SPECIAL ATTENTION:
#         - For BUS policies, clearly distinguish between SCHOOL BUS and STAFF BUS
#         - For TAXI, all companies follow the same payin-category-based rules
#         - Pay attention to company-specific rules vs "All Companies" rules

#         Return ONLY a valid JSON array with all original fields plus:
#         - 'Calculated Payout' (as percentage string)
#         - 'Formula Used' (the exact rule applied)
#         - 'Rule Explanation' (detailed explanation: LOB → Segment → Company → Payin Category → Rule)
#         Note : Please don't make any calculation mistake while applying formula
#         """
        
#         formula_response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You are an intelligent financial calculator. Use your intelligence to match policy data with formula grid rules accurately. Output ONLY valid JSON array."},
#                 {"role": "user", "content": formula_prompt}
#             ],
#             temperature=0.0
#         )
        
#         calc_json = formula_response.choices[0].message.content.strip()
#         logger.info(f"Raw formula response: {calc_json[:500]}...")
        
#         # Clean and parse formula response
#         calc_json_cleaned = clean_json_response(calc_json)
        
#         try:
#             calculated_data = json.loads(calc_json_cleaned)
#             calculated_data = ensure_list_format(calculated_data)
#         except json.JSONDecodeError as e:
#             logger.error(f"Formula JSON decode error: {str(e)}")
#             logger.error(f"Problematic formula JSON: {calc_json_cleaned}")
            
#             # Fallback: use original data with default calculated values
#             calculated_data = []
#             for record in policy_data:
#                 record_copy = record.copy()
#                 record_copy['Calculated Payout'] = f"{int(record.get('Payin_Value', 0))}%"
#                 record_copy['Formula Used'] = "Default (no formula applied)"
#                 record_copy['Rule Explanation'] = "Formula parsing failed, using original payin value"
#                 calculated_data.append(record_copy)

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
#             "extracted_formula_text": extracted_formula_text,
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
        
#         # Policy file upload
#         policy_file = st.file_uploader(
#             "📄 Upload Policy File",
#             type=['pdf', 'png', 'jpg', 'jpeg', 'txt', 'csv', 'xlsx', 'xls'],
#             help="Upload your insurance policy document"
#         )
        
#         # Formula file upload
#         formula_file = st.file_uploader(
#             "📊 Upload Formula/Rules File",
#             type=['pdf', 'png', 'jpg', 'jpeg', 'txt', 'csv', 'xlsx', 'xls'],
#             help="Upload your formula rules or grid document"
#         )
        
#         # Process button
#         process_button = st.button(
#             "🚀 Process Files", 
#             type="primary",
#             disabled=not (policy_file and formula_file)
#         )
    
#     # Main content area
#     if not policy_file or not formula_file:
#         st.info("👆 Please upload both policy and formula files to begin processing.")
#         st.markdown("""
#         ### 📋 Instructions:
#         1. **Company Name**: Enter the insurance company name
#         2. **Policy File**: Upload the document containing policy data (PDF, Image, Excel, CSV, etc.)
#         3. **Formula File**: Upload the document containing formula rules and grids
#         4. **Process**: Click the process button to extract data and calculate payouts
        
#         ### 🎯 Features:
#         - **Multi-format Support**: PDF, Images, Excel, CSV, Text files
#         - **AI-Powered Extraction**: Uses GPT-4 for intelligent text extraction
#         - **Enhanced Remarks Extraction**: Automatically detects and extracts:
#           - Vehicle make information (Tata, Maruti, etc.)
#           - Age information (>5 years, etc.)
#           - Transaction type (New/Old/Renewal)
#         - **Smart Formula Application**: Automatically applies rules based on segments and companies
#         - **Excel Export**: Download processed data as formatted Excel file
#         """)
#         return
    
#     if process_button:
#         try:
#             # Read file contents
#             policy_file_bytes = policy_file.read()
#             formula_file_bytes = formula_file.read()
            
#             # Process files
#             with st.spinner("Processing files... This may take a few moments."):
#                 results = process_files(
#                     policy_file_bytes, policy_file.name, policy_file.type,
#                     formula_file_bytes, formula_file.name, formula_file.type,
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
            
#             with tab2:
#                 st.subheader("📝 Extracted Text")
                
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.write("**Policy File Text:**")
#                     st.text_area(
#                         "Policy Text", 
#                         results["extracted_text"], 
#                         height=300,
#                         key="policy_text"
#                     )
                
#                 with col2:
#                     st.write("**Formula File Text:**")
#                     st.text_area(
#                         "Formula Text", 
#                         results["extracted_formula_text"], 
#                         height=300,
#                         key="formula_text"
#                     )
            
#             with tab3:
#                 st.subheader("🧾 Parsed Policy Data")
#                 st.json(results["parsed_data"])
            
#             with tab4:
#                 st.subheader("🧮 Calculated Data with Formulas")
#                 st.json(results["calculated_data"])
            
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


def extract_text_from_file(file_bytes: bytes, filename: str, content_type: str, is_formula: bool = False) -> str:
    """Extract text from uploaded file"""
    file_extension = filename.split('.')[-1].lower() if '.' in filename else ''
    file_type = content_type if content_type else file_extension

    # Image-based extraction with enhanced OCR
    image_extensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']
    if file_extension in image_extensions or file_type.startswith('image/'):
        image_base64 = base64.b64encode(file_bytes).decode('utf-8')
        
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
            category = "Payin from 51%"
        return payin_value, category
    except (ValueError, TypeError):
        logger.warning(f"Could not parse payin value: {payin_str}")
        return 0.0, "Payin Below 20%"


def process_files(policy_file_bytes, policy_filename, policy_content_type,
                 formula_file_bytes, formula_filename, formula_content_type, 
                 company_name):
    """Main processing function"""
    try:
        st.info("🔍 Extracting text from files...")
        
        # Extract text with enhanced intelligence
        extracted_text = extract_text_from_file(policy_file_bytes, policy_filename, policy_content_type, is_formula=False)
        extracted_formula_text = extract_text_from_file(formula_file_bytes, formula_filename, formula_content_type, is_formula=True)

        logger.info(f"Extracted text length: {len(extracted_text)}")
        logger.info(f"Extracted formula text length: {len(extracted_formula_text)}")

        st.success(f"✅ Text extracted successfully! Policy: {len(extracted_text)} chars, Formula: {len(extracted_formula_text)} chars")

        # Parse policy data with enhanced segment identification
        st.info("🧠 Parsing policy data with AI...")
        
        parse_prompt = f"""
        Analyze the following text, which contains insurance policy details.
        Use your intelligence to identify and extract the data accurately.

        Company Name: {company_name}

        IMPORTANT: Always return a valid JSON array (list) of records, even if there's only one record.

        Extract into JSON records with these exact fields:
        - "Segment": identify LOB like TW, PVT CAR, CV, BUS, TAXI, MISD and policy type like TP, COMP, SAOD
        - "Location": location information
        - "Policy Type": policy type information
        - "Payin": convert payout values to percentage format, e.g. 0.5 → 50%, 34 → 34%
        - "Doable District": district information
        - "Remarks": any additional information INCLUDING vehicle make information

        ENHANCED REMARKS FIELD INSTRUCTIONS:
        - Extract ALL additional information that doesn't fit in other specific fields
        - Include any notes, conditions, special instructions, exclusions, etc.
        - ALSO look for vehicle make information in the text (e.g., "Make – Tata, Maruti" or similar patterns)
        - Vehicle makes include brands like: TATA, MARUTI, SUZUKI, HYUNDAI, HONDA, TOYOTA, MAHINDRA, BAJAJ, HERO, TVS, YAMAHA, etc.
        - ALSO look for AGE information (e.g., ">5 years", "Age: 3 years", "Below 5 years", etc.)
        - ALSO look for TRANSACTION information (e.g., "New", "Old", "Renewal", "Fresh", "Transaction: New/Old", etc.)
        - If vehicle makes are found, include them: "Vehicle Makes: [make1, make2, ...]"
        - If age information is found, include it: "Age: [age_info]"
        - If transaction type is found, include it: "Transaction: [New/Old/Renewal]"
        - Combine all relevant additional information into the remarks field, separated by semicolons
        - Example format: "Vehicle Makes: Tata, Maruti; Age: >5 years; Transaction: Old; [other remarks]"
        - If no other remarks exist, just include the available information

        Be intelligent in identifying segments even if the format varies.
        Look for patterns like:
        - TW (Two Wheeler) related segments
        - PVT CAR (Private Car) segments  
        - CV (Commercial Vehicle) segments
        - BUS segments
        - TAXI segments
        - MISD (Miscellaneous) segments

        If you cannot find specific information for any field, use "N/A" or reasonable defaults.
        Note : If Policy is not defined or told , then consider it as COMP/TP
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

        # Apply formulas with enhanced intelligence
        st.info("🧮 Applying formulas and calculating payouts...")
        
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
        These are the examples i am giving you:
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

        Return ONLY a valid JSON array with all original fields plus:
        - 'Calculated Payout' (as percentage string)
        - 'Formula Used' (the exact rule applied)
        - 'Rule Explanation' (detailed explanation: LOB → Segment → Company → Payin Category → Rule)
        Note : Please don't make any calculation mistake while applying formula
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
            "extracted_formula_text": extracted_formula_text,
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
        
        # Policy file upload
        policy_file = st.file_uploader(
            "📄 Upload Policy File",
            type=['pdf', 'png', 'jpg', 'jpeg', 'txt', 'csv', 'xlsx', 'xls'],
            help="Upload your insurance policy document"
        )
        
        # Formula file upload
        formula_file = st.file_uploader(
            "📊 Upload Formula/Rules File",
            type=['pdf', 'png', 'jpg', 'jpeg', 'txt', 'csv', 'xlsx', 'xls'],
            help="Upload your formula rules or grid document"
        )
        
        # Process button
        process_button = st.button(
            "🚀 Process Files", 
            type="primary",
            disabled=not (policy_file and formula_file)
        )
    
    # Main content area
    if not policy_file or not formula_file:
        st.info("👆 Please upload both policy and formula files to begin processing.")
        st.markdown("""
        ### 📋 Instructions:
        1. **Company Name**: Enter the insurance company name
        2. **Policy File**: Upload the document containing policy data (PDF, Image, Excel, CSV, etc.)
        3. **Formula File**: Upload the document containing formula rules and grids
        4. **Process**: Click the process button to extract data and calculate payouts
        
        ### 🎯 Features:
        - **Multi-format Support**: PDF, Images, Excel, CSV, Text files
        - **AI-Powered Extraction**: Uses GPT-4 for intelligent text extraction
        - **Enhanced Remarks Extraction**: Automatically detects and extracts:
          - Vehicle make information (Tata, Maruti, etc.)
          - Age information (>5 years, etc.)
          - Transaction type (New/Old/Renewal)
        - **Smart Formula Application**: Automatically applies rules based on segments and companies
        - **Excel Export**: Download processed data as formatted Excel file
        """)
        return
    
    if process_button:
        try:
            # Read file contents
            policy_file_bytes = policy_file.read()
            formula_file_bytes = formula_file.read()
            
            # Process files
            with st.spinner("Processing files... This may take a few moments."):
                results = process_files(
                    policy_file_bytes, policy_file.name, policy_file.type,
                    formula_file_bytes, formula_file.name, formula_file.type,
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
            
            with tab2:
                st.subheader("📝 Extracted Text")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Policy File Text:**")
                    st.text_area(
                        "Policy Text", 
                        results["extracted_text"], 
                        height=300,
                        key="policy_text"
                    )
                
                with col2:
                    st.write("**Formula File Text:**")
                    st.text_area(
                        "Formula Text", 
                        results["extracted_formula_text"], 
                        height=300,
                        key="formula_text"
                    )
            
            with tab3:
                st.subheader("🧾 Parsed Policy Data")
                st.json(results["parsed_data"])
            
            with tab4:
                st.subheader("🧮 Calculated Data with Formulas")
                st.json(results["calculated_data"])
            
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
                
                st.info("💡 The Excel file contains formatted data with company header and calculated payouts.")
                
        except Exception as e:
            st.error(f"❌ Error processing files: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()
