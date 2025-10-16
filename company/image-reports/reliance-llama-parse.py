# import streamlit as st
# from io import BytesIO
# import base64
# import json
# import os
# from dotenv import load_dotenv
# import logging
# import re
# import pandas as pd
# import img2pdf
# from llama_parse import LlamaParse
# from llama_index.core import Document

# # Check if required packages are available
# try:
#     from openai import OpenAI
# except ImportError:
#     st.error("OpenAI package not found. Please install it using 'pip install openai'")
#     st.stop()

# # Load environment variables
# load_dotenv()

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load API keys
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
# if not OPENAI_API_KEY:
#     st.error("OPENAI_API_KEY environment variable not set. Please set it in your .env file or environment variables.")
#     st.stop()
# if not LLAMA_CLOUD_API_KEY:
#     st.error("LLAMA_CLOUD_API_KEY environment variable not set. Please set it in your .env file or environment variables.")
#     st.stop()

# # Initialize OpenAI client
# try:
#     client = OpenAI(api_key=OPENAI_API_KEY)
# except Exception as e:
#     st.error(f"Failed to initialize OpenAI client: {str(e)}")
#     st.stop()

# # Initialize LlamaParse
# try:
#     parser = LlamaParse(
#         api_key=LLAMA_CLOUD_API_KEY,
#         result_type="text",  # Use "text" for raw text extraction; "markdown" or "json" for structured output
#         num_workers=2,
#         verbose=True,
#         parsing_instruction="Extract ALL text from the image, including tables, headers, and remarks, with extreme accuracy. Preserve table structure and all numerical values."
#     )
# except Exception as e:
#     st.error(f"Failed to initialize LlamaParse: {str(e)}")
#     st.stop()

# # Embedded Formula Data (unchanged)
# FORMULA_DATA = [
#     # ... (keep your existing FORMULA_DATA array as is)
#     {"LOB": "PVT CAR", "SEGMENT": "PVT CAR COMP + SAOD", "INSURER": "All Companies", "PO": "90% of Payin", "REMARKS": "All Fuel"},
#     # ... (rest of the array)
# ]

# def extract_text_from_file(file_bytes: bytes, filename: str, content_type: str) -> str:
#     """Extract text from uploaded image file using LlamaParse"""
#     file_extension = filename.split('.')[-1].lower() if '.' in filename else ''
#     file_type = content_type if content_type else file_extension

#     # Image-based extraction with LlamaParse
#     image_extensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']
#     if file_extension in image_extensions or file_type.startswith('image/'):
#         try:
#             # Convert image to PDF for LlamaParse (required for image input)
#             pdf_bytes = BytesIO(img2pdf.convert(BytesIO(file_bytes), format='pdf'))
            
#             # Parse the PDF with LlamaParse
#             with st.spinner("Extracting text with LlamaParse..."):
#                 documents = parser.load_data(pdf_bytes)
            
#             if not documents or not documents[0].text.strip():
#                 logger.error("LlamaParse returned empty or invalid text")
#                 return ""
            
#             extracted_text = documents[0].text.strip()
#             logger.info(f"Extracted text length: {len(extracted_text)}")
            
#             if len(extracted_text) < 10:
#                 logger.error("LlamaParse returned very short text")
#                 return ""
            
#             return extracted_text
            
#         except Exception as e:
#             logger.error(f"Error in LlamaParse extraction: {str(e)}")
#             raise ValueError(f"Failed to extract text from image with LlamaParse: {str(e)}")

#     raise ValueError(f"Unsupported file type for {filename}. Only images are supported.")

# # Rest of the functions remain largely unchanged (clean_json_response, ensure_list_format, classify_payin, apply_formula_directly, process_files, main)
# # Only minor adjustments are needed in process_files to handle LlamaParse output

# def process_files(policy_file_bytes, policy_filename, policy_content_type, company_name):
#     """Main processing function with enhanced error handling"""
#     try:
#         st.info("🔍 Extracting text from policy image with LlamaParse...")
        
#         extracted_text = extract_text_from_file(policy_file_bytes, policy_filename, policy_content_type)
#         logger.info(f"Extracted text length: {len(extracted_text)}")

#         if not extracted_text.strip():
#             logger.error("No text extracted from the image")
#             st.error("❌ No text could be extracted. Please ensure the image is clear and contains readable text.")
#             return {
#                 "extracted_text": "",
#                 "parsed_data": [],
#                 "calculated_data": [],
#                 "excel_data": None,
#                 "df_calc": pd.DataFrame()
#             }

#         st.success(f"✅ Text extracted successfully with LlamaParse! Length: {len(extracted_text)} chars")

#         st.info("🧠 Parsing policy data with AI...")
        
#         parse_prompt = f"""Analyze this insurance policy text and extract structured data.

# Company Name: {company_name}

# CRITICAL INSTRUCTIONS:
# 1. ALWAYS return a valid JSON ARRAY (list) of objects, even if there's only one record
# 2. Each object must have these EXACT field names:
#    - "Segment": LOB + policy type (e.g., "TW TP", "PVT CAR COMP", "CV upto 2.5 Tn")
#    - "Location": location/region information (use "N/A" if not found)
#    - "Policy Type": policy type details (use "COMP/TP" if not specified)
#    - "Payin": percentage value (convert decimals: 0.625 → 62.5%, or keep as is: 34%)
#    - "Doable District": district info (use "N/A" if not found)
#    - "Remarks": additional info including vehicle makes, age, transaction type, validity

# 3. For Segment field:
#    - Identify LOB: TW, PVT CAR, CV, BUS, TAXI, MISD
#    - Add policy type: TP, COMP, SAOD, etc.
#    - For CV: preserve tonnage (e.g., "CV upto 2.5 Tn")

# 4. For Payin field:
#    - If you see decimals like 0.625, convert to 62.5%
#    - If you see whole numbers like 34, add % to make 34%
#    - If you see percentages, keep them as is
#    - Use the value from the "PO" column or any column that indicates payout/payin
#    - Do not use values from "Discount" column for Payin

# 5. For Remarks field - extract ALL additional info:
#    - Vehicle makes (Tata, Maruti, etc.) → "Vehicle Makes: Tata, Maruti"
#    - Age info (>5 years, etc.) → "Age: >5 years"
#    - Transaction type (New/Old/Renewal) → "Transaction: New"
#    - Validity dates → "Validity till: [date]"
#    - Decline RTO information (e.g., "Decline RTO: Dhar, Jhabua")
#    - Combine with semicolons: "Vehicle Makes: Tata; Age: >5 years; Transaction: New"

# IMPORTANT: 
# - If a field is not found, use "N/A"
# - Return ONLY the JSON array, no other text
# - Ensure the JSON is valid and parseable
# - Do not extract or include the "Discount" column or its values in any field. Ignore it completely.
# - The "PO" column contains the Payin values - use that for the "Payin" field.
# - The table may have a "Discount" column - IGNORE it completely. Do not include its values anywhere, not even in remarks.
# IGNORE these columns completely - DO NOT extract them:
#    - Discount
#    - CD1
#    - Any column containing "discount" or "cd1" 
#    - These are not needed for our analysis

# NOTE:
# - Taxi PCV comes under the category of Taxi
# - Multiple columns are there which has payouts based on either policy type or fuel type , so consider that as payin
# - PCV < 6 STR comes under Taxi
# - PC means Private Car and STP = TP
# - Kali Pilli or Kaali Pilli means Taxi and it comes under Taxi
# - If in SEGMENT of Private Car, SAOD mentioned then it comes into PVT CAR COMP + SAOD segment, also same for COMP
# Here is the training Data:
# I am training you

# ### Overview of How the Data is Given
# [Truncated for brevity - keep your existing overview and image analysis sections]

# Text to analyze:
# {extracted_text}
# """
        
#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {
#                         "role": "system", 
#                         "content": "You are a data extraction expert. Extract policy data as a JSON array. Convert all Payin values to percentage format. Always return valid JSON array with complete field names. Extract all additional information for remarks."
#                     },
#                     {"role": "user", "content": parse_prompt}
#                 ],
#                 temperature=0.0,
#                 max_tokens=4000
#             )
            
#             parsed_json = response.choices[0].message.content.strip()
#             logger.info(f"Raw parsing response: {parsed_json[:500]}...")
            
#             cleaned_json = clean_json_response(parsed_json)
#             logger.info(f"Cleaned JSON: {cleaned_json[:500]}...")
            
#             try:
#                 policy_data = json.loads(cleaned_json)
#                 policy_data = ensure_list_format(policy_data)
                
#                 if not policy_data or len(policy_data) == 0:
#                     raise ValueError("Parsed data is empty")
                    
#             except json.JSONDecodeError as e:
#                 logger.error(f"JSON decode error: {str(e)} with cleaned JSON: {cleaned_json}")
#                 st.warning("⚠️ AI response was not valid JSON. Creating fallback structure...")
#                 policy_data = [{
#                     "Segment": "Unknown",
#                     "Location": "N/A",
#                     "Policy Type": "N/A", 
#                     "Payin": "0%",
#                     "Doable District": "N/A",
#                     "Remarks": f"Failed to parse - please check image quality. Extract manually from: {extracted_text[:200]}"
#                 }]
        
#         except Exception as e:
#             logger.error(f"Error in AI parsing: {str(e)} with raw response: {parsed_json[:500]}...")
#             st.warning("⚠️ AI parsing failed. Creating fallback structure...")
#             policy_data = [{
#                 "Segment": "Unknown",
#                 "Location": "N/A",
#                 "Policy Type": "N/A",
#                 "Payin": "0%",
#                 "Doable District": "N/A",
#                 "Remarks": f"Parsing error: {str(e)}"
#             }]

#         st.success(f"✅ Successfully parsed {len(policy_data)} policy records")

#         # Rest of the process_files function remains the same (classify_payin, apply_formula_directly, Excel generation)
#         for record in policy_data:
#             try:
#                 if 'Discount' in record:
#                     del record['Discount']
#                 payin_val, payin_cat = classify_payin(record.get('Payin', '0%'))
#                 record['Payin_Value'] = payin_val
#                 record['Payin_Category'] = payin_cat
#             except Exception as e:
#                 logger.warning(f"Error classifying payin: {e}")
#                 record['Payin_Value'] = 0.0
#                 record['Payin_Category'] = "Payin Below 20%"

#         st.info("🧮 Applying formulas and calculating payouts...")
#         calculated_data = apply_formula_directly(policy_data, company_name)
        
#         if not calculated_data or len(calculated_data) == 0:
#             st.error("❌ No data after formula application")
#             return {
#                 "extracted_text": extracted_text,
#                 "parsed_data": policy_data,
#                 "calculated_data": [],
#                 "excel_data": None,
#                 "df_calc": pd.DataFrame()
#             }

#         st.success(f"✅ Successfully calculated {len(calculated_data)} records")

#         st.info("📊 Creating Excel file...")
        
#         df_calc = pd.DataFrame(calculated_data)
        
#         if df_calc.empty:
#             st.error("❌ DataFrame is empty")
#             return {
#                 "extracted_text": extracted_text,
#                 "parsed_data": policy_data,
#                 "calculated_data": calculated_data,
#                 "excel_data": None,
#                 "df_calc": df_calc
#             }

#         output = BytesIO()
#         try:
#             with pd.ExcelWriter(output, engine='openpyxl') as writer:
#                 df_calc.to_excel(writer, sheet_name='Policy Data', startrow=2, index=False)
#                 worksheet = writer.sheets['Policy Data']

#                 headers = list(df_calc.columns)
#                 for col_num, value in enumerate(headers, 1):
#                     cell = worksheet.cell(row=3, column=col_num, value=value)
#                     cell.font = cell.font.copy(bold=True)

#                 if len(headers) > 1:
#                     company_cell = worksheet.cell(row=1, column=1, value=company_name)
#                     worksheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
#                     company_cell.font = company_cell.font.copy(bold=True, size=14)
#                     company_cell.alignment = company_cell.alignment.copy(horizontal='center')

#                     title_cell = worksheet.cell(row=2, column=1, value='Policy Data with Payin and Calculated Payouts')
#                     worksheet.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(headers))
#                     title_cell.font = title_cell.font.copy(bold=True, size=12)
#                     title_cell.alignment = title_cell.alignment.copy(horizontal='center')
#                 else:
#                     worksheet.cell(row=1, column=1, value=company_name)
#                     worksheet.cell(row=2, column=1, value='Policy Data with Payin and Calculated Payouts')

#         except Exception as e:
#             logger.error(f"Error creating Excel file: {str(e)}")
#             st.error(f"❌ Error creating Excel: {str(e)}")
#             return {
#                 "extracted_text": extracted_text,
#                 "parsed_data": policy_data,
#                 "calculated_data": calculated_data,
#                 "excel_data": None,
#                 "df_calc": df_calc
#             }

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
#         logger.error(f"Unexpected error in process_files: {str(e)}", exc_info=True)
#         st.error(f"❌ Processing error: {str(e)}")
#         return {
#             "extracted_text": "",
#             "parsed_data": [],
#             "calculated_data": [],
#             "excel_data": None,
#             "df_calc": pd.DataFrame()
#         }

# # Main function remains unchanged
# def main():
#     st.set_page_config(
#         page_title="Insurance Policy Processing", 
#         page_icon="📋", 
#         layout="wide"
#     )
    
#     st.title("🏢 Insurance Policy Processing System")
#     st.markdown("---")
    
#     with st.sidebar:
#         st.header("📁 File Upload")
        
#         company_name = st.text_input(
#             "Company Name", 
#             value="Unknown Company",
#             help="Enter the insurance company name"
#         )
        
#         policy_file = st.file_uploader(
#             "📄 Upload Policy Image",
#             type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'],
#             help="Upload your insurance policy image"
#         )
        
#         st.info("📊 Formula rules are embedded in the system and will be automatically applied.")
        
#         process_button = st.button(
#             "🚀 Process Policy File", 
#             type="primary",
#             disabled=not policy_file
#         )
    
#     if not policy_file:
#         st.info("👆 Please upload a policy image to begin processing.")
#         st.markdown("""
#         ### 📋 Instructions:
#         1. **Company Name**: Enter the insurance company name
#         2. **Policy Image**: Upload the image containing policy data (PNG, JPG, etc.)
#         3. **Process**: Click the process button to extract data and calculate payouts
        
#         ### 🎯 Features:
#         - **Image Support**: PNG, JPG, JPEG, GIF, BMP, TIFF
#         - **AI-Powered Extraction**: Uses LlamaParse for intelligent text extraction from images
#         - **Enhanced Remarks Extraction**: Automatically detects and extracts:
#           - Vehicle make information (Tata, Maruti, etc.)
#           - Age information (>5 years, etc.)
#           - Transaction type (New/Old/Renewal)
#           - Validity dates
#         - **Smart Formula Application**: Uses embedded formula rules for accurate calculations
#         - **Excel Export**: Download processed data as formatted Excel file
#         """)
#         return
    
#     if process_button:
#         try:
#             policy_file_bytes = policy_file.read()
            
#             if len(policy_file_bytes) == 0:
#                 st.error("❌ Uploaded file is empty. Please upload a valid image.")
#                 return
            
#             with st.spinner("Processing policy image... This may take a few moments."):
#                 results = process_files(
#                     policy_file_bytes, 
#                     policy_file.name, 
#                     policy_file.type,
#                     company_name
#                 )
            
#             if not results["calculated_data"] or len(results["calculated_data"]) == 0:
#                 st.error("❌ No data was extracted or processed. Please check the image quality and try again.")
#                 if results["extracted_text"]:
#                     with st.expander("📝 View Extracted Text"):
#                         st.text_area("Extracted Text", results["extracted_text"], height=300)
#                 if results["parsed_data"]:
#                     with st.expander("🧾 View Parsed Data"):
#                         st.json(results["parsed_data"])
#                 return
            
#             st.success("🎉 Processing completed successfully!")
            
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
                
#                 col1, col2, col3, col4 = st.columns(4)
#                 with col1:
#                     st.metric("Total Records", len(results["calculated_data"]))
#                 with col2:
#                     if len(results["calculated_data"]) > 0:
#                         avg_payin = sum([r.get('Payin_Value', 0) for r in results["calculated_data"]]) / len(results["calculated_data"])
#                         st.metric("Avg Payin", f"{avg_payin:.1f}%")
#                     else:
#                         st.metric("Avg Payin", "0.0%")
#                 with col3:
#                     segments = set([r.get('Segment', 'N/A') for r in results["calculated_data"]])
#                     st.metric("Unique Segments", len(segments))
#                 with col4:
#                     st.metric("Company", company_name)
                
#                 st.subheader("🔧 Formula Rules Applied")
#                 formula_summary = {}
#                 for record in results["calculated_data"]:
#                     formula = record.get('Formula Used', 'Unknown')
#                     formula_summary[formula] = formula_summary.get(formula, 0) + 1
                
#                 for formula, count in formula_summary.items():
#                     st.write(f"• **{formula}**: Applied to {count} record(s)")

#             with tab2:
#                 st.subheader("📝 Extracted Text from Policy Image")
#                 st.text_area("Policy Text", results["extracted_text"], height=400, key="policy_text")
                
#                 st.subheader("📊 Embedded Formula Rules")
#                 df_formula = pd.DataFrame(FORMULA_DATA)
#                 st.dataframe(df_formula, use_container_width=True)
            
#             with tab3:
#                 st.subheader("🧾 Parsed Policy Data")
#                 st.json(results["parsed_data"])
            
#             with tab4:
#                 st.subheader("🧮 Calculated Data with Formulas")
#                 st.json(results["calculated_data"])
                
#                 st.subheader("🔍 Rule Explanations")
#                 for i, record in enumerate(results["calculated_data"]):
#                     with st.expander(f"Record {i+1}: {record.get('Segment', 'Unknown')}"):
#                         st.write(f"**Payin**: {record.get('Payin', 'N/A')}")
#                         st.write(f"**Calculated Payout**: {record.get('Calculated Payout', 'N/A')}")
#                         st.write(f"**Formula Used**: {record.get('Formula Used', 'N/A')}")
#                         st.write(f"**Rule Explanation**: {record.get('Rule Explanation', 'N/A')}")
            
#             with tab5:
#                 st.subheader("📥 Download Results")
                
#                 if results["excel_data"]:
#                     st.download_button(
#                         label="📊 Download Excel File",
#                         data=results["excel_data"],
#                         file_name=f"{company_name}_processed_policies.xlsx",
#                         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#                     )
#                 else:
#                     st.warning("⚠️ Excel file could not be generated")
                
#                 json_data = json.dumps(results["calculated_data"], indent=2)
#                 st.download_button(
#                     label="📄 Download JSON Data",
#                     data=json_data,
#                     file_name=f"{company_name}_processed_data.json",
#                     mime="application/json"
#                 )
                
#                 if not results["df_calc"].empty:
#                     csv_data = results["df_calc"].to_csv(index=False)
#                     st.download_button(
#                         label="📋 Download CSV File",
#                         data=csv_data,
#                         file_name=f"{company_name}_processed_policies.csv",
#                         mime="text/csv"
#                     )
#                 else:
#                     st.warning("⚠️ CSV file could not be generated")
                
#                 st.info("💡 The Excel file contains formatted data with company header and calculated payouts")
                
#         except Exception as e:
#             st.error(f"❌ Error processing files: {str(e)}")
#             logger.error(f"Main processing error: {str(e)}", exc_info=True)
#             st.exception(e)


# if __name__ == "__main__":
#     main()


    
import streamlit as st
from io import BytesIO
import base64
import json
import os
from dotenv import load_dotenv
import logging
import re
import pandas as pd
from llama_parse import LlamaParse
import tempfile

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

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY environment variable not set. Please set it in your .env file or environment variables.")
    st.stop()
if not LLAMA_CLOUD_API_KEY:
    st.error("LLAMA_CLOUD_API_KEY environment variable not set. Please set it in your .env file or environment variables.")
    st.stop()

# Initialize OpenAI client
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize OpenAI client: {str(e)}")
    st.stop()

# Initialize LlamaParse with better instructions
try:
    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",  # Changed to markdown for better table structure
        num_workers=4,
        verbose=True,
        language="en",
        parsing_instruction="""Extract ALL text from this insurance document with extreme accuracy.

CRITICAL REQUIREMENTS:
1. Preserve ALL table structures with headers and data
2. Extract EVERY column in multi-column tables separately
3. Maintain exact numerical values and percentages
4. Capture all headers, sub-headers, and row labels
5. Include all text outside tables (notes, conditions, bonuses)
6. Preserve formatting like bold, highlights, and emphasis

For tables:
- Identify column headers clearly
- Extract each cell value
- Maintain row-column relationships
- Note any merged cells or spanning headers

Return complete structured text with clear table delineation."""
    )
except Exception as e:
    st.error(f"Failed to initialize LlamaParse: {str(e)}")
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


def extract_text_from_file(file_bytes: bytes, filename: str, content_type: str) -> str:
    """Extract text from uploaded image file using LlamaParse"""
    file_extension = filename.split('.')[-1].lower() if '.' in filename else ''
    file_type = content_type if content_type else file_extension

    # Image-based extraction with LlamaParse
    image_extensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']
    if file_extension in image_extensions or file_type.startswith('image/'):
        try:
            # Save to temporary file for LlamaParse
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                tmp_file.write(file_bytes)
                tmp_file_path = tmp_file.name
            
            # Parse with LlamaParse
            with st.spinner("Extracting text with LlamaParse..."):
                documents = parser.load_data(tmp_file_path)
            
            # Clean up temp file
            os.unlink(tmp_file_path)
            
            if not documents or not documents[0].text.strip():
                logger.error("LlamaParse returned empty or invalid text")
                return ""
            
            extracted_text = documents[0].text.strip()
            logger.info(f"Extracted text length: {len(extracted_text)}")
            
            if len(extracted_text) < 10:
                logger.error("LlamaParse returned very short text")
                return ""
            
            return extracted_text
            
        except Exception as e:
            logger.error(f"Error in LlamaParse extraction: {str(e)}")
            raise ValueError(f"Failed to extract text from image with LlamaParse: {str(e)}")

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
            
            if any(tw_keyword in segment_upper for tw_keyword in ['TW', '2W', 'TWO WHEELER', 'TWO-WHEELER']):
                lob = "TW"
            elif any(car_keyword in segment_upper for car_keyword in ['PVT CAR', 'PRIVATE CAR', 'CAR', 'PVTCAR']):
                lob = "PVT CAR"
            elif any(cv_keyword in segment_upper for cv_keyword in ['CV', 'COMMERCIAL', 'LCV', 'GVW', 'TN', 'GCW']):
                lob = "CV"
            elif 'BUS' in segment_upper:
                lob = "BUS"
            elif 'TAXI' in segment_upper:
                lob = "TAXI"
            elif any(misd_keyword in segment_upper for misd_keyword in ['MISD', 'TRACTOR', 'MISC']):
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
                        if not any(keyword in segment_upper for keyword in ["UPTO 2.5", "2.5 TN", "2.5 GVW", "2.5TN", "2.5GVW", "UPTO2.5", "2.5"]):
                            segment_match = True
                elif lob == "BUS":
                    if matched_segment == rule_segment:
                        segment_match = True
                elif lob == "PVT CAR":
                    if "COMP" in rule_segment and any(keyword in segment for keyword in ["COMP", "COMPREHENSIVE"]):
                        segment_match = True
                    elif "TP" in rule_segment and "TP" in segment and "COMP" not in segment:
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
        st.info("🔍 Extracting text from policy image with LlamaParse...")
        
        extracted_text = extract_text_from_file(policy_file_bytes, policy_filename, policy_content_type)
        logger.info(f"Extracted text length: {len(extracted_text)}")

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

        st.success(f"✅ Text extracted successfully with LlamaParse! Length: {len(extracted_text)} chars")

        st.info("🧠 Parsing policy data with AI...")
        
        # IMPROVED PARSE PROMPT - Exact structure from reliance.py
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
   - If you see percentages, keep them as is (e.g., "50%", "45%", "-72%")
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

SPECIAL NOTES:
- SATP means TP, so whenever Private Car SATP is mentioned then it means PVT CAR TP
- Sometimes two columns are there in the input which are in values, like diesel or petrol, means fuel type also, so extract that too and parse that too
- Sometimes MCY means motorcycle, so consider them in TW LOB, be it Motorcycles, scooters
- Handle various formats: line-separated grids (e.g., tonnage with locations and percentages), tables with State/RTO/Fuel, emails with forwarded messages, incentives for YTD targets.
- Extract month/year (e.g., "June 25", "July 25", "Aug 25", "FY 25-26") in Remarks.
- For grids, create one object per row or entry (e.g., each state/RTO as a record).
- If text is ambiguous, infer LOB/Segment from context (e.g., "School bus >14 seater" to "SCHOOL BUS").
- Also if in the input % is mentioned in "-", this sign is not subtraction but in fact it is hyphen symbol
- Sometimes the values given in input are for example -68%, so in that case also consider it as 68% only and ignore the - sign
- Sometimes segment is also mentioned as Product in the input, and remember PCI means Private Car
- Sometimes two columns are there in the input which are in values, like diesel or petrol, means fuel type also, so extract that too and parse that too
- Taxi PCV comes under the category of Taxi
- Multiple columns are there which has payouts based on either policy type or fuel type, so consider that as payin
- PCV < 6 STR comes under Taxi
- PC means Private Car and STP = TP
- Kali Pilli or Kaali Pilli means Taxi and it comes under Taxi
- If in SEGMENT of Private Car, SAOD mentioned then it comes into PVT CAR COMP + SAOD segment, also same for COMP

MULTI-COLUMN TABLE EXTRACTION - CRITICAL:
When you see a table with multiple payout columns (e.g., SAOD Petrol, SAOD Diesel, Comp Petrol), you MUST:
1. Identify ALL column headers in the table
2. For each row (e.g., Zone: East, North, West, South), create a SEPARATE JSON object for EACH column
3. Each object should have:
   - Segment: Based on the column header (e.g., "PVT CAR SAOD" for SAOD Petrol column, "PVT CAR COMP + SAOD" for Comp Petrol)
   - Location: The row label (e.g., "East", "North", "West", "South")
   - Policy Type: The exact column header (e.g., "SAOD Petrol", "SAOD Diesel", "Comp Petrol")
   - Payin: The specific cell value at the intersection of that row and column
   - Remarks: Include all context notes that apply to this specific combination

Example: If table has 4 rows (East, North, West, South) and 3 payout columns (SAOD Petrol, SAOD Diesel, Comp Petrol), you MUST create 12 JSON objects (4 × 3 = 12).

DO NOT extract only one column - extract ALL columns as separate records!

Text to analyze:
{extracted_text}
"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a data extraction expert specializing in insurance policy data. Extract policy data as a JSON array. For tables with multiple payout columns, create a separate JSON object for each cell (row × column combination). Convert all Payin values to percentage format. Always return valid JSON array with complete field names. Extract all additional information for remarks. CRITICAL: When you see multi-column tables, iterate through ALL columns and create separate records for each column's data."
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
            logger.error(f"Error in AI parsing: {str(e)}")
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

        # Classify Payin values
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
        page_title="Insurance Policy Processing with LlamaParse", 
        page_icon="📋", 
        layout="wide"
    )
    
    st.title("🏢 Insurance Policy Processing System (LlamaParse)")
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
        - **LlamaParse Integration**: Advanced document parsing for better table extraction
        - **Multi-Column Table Support**: Extracts ALL columns from tables separately
        - **Enhanced Remarks Extraction**: Automatically detects and extracts:
          - Vehicle make information (Tata, Maruti, etc.)
          - Age information (>5 years, etc.)
          - Transaction type (New/Old/Renewal)
          - Validity dates
          - Fuel type variations (Petrol/Diesel)
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
            
            with st.spinner("Processing policy image with LlamaParse... This may take a few moments."):
                results = process_files(
                    policy_file_bytes, 
                    policy_file.name, 
                    policy_file.type,
                    company_name
                )
            
            if not results["calculated_data"] or len(results["calculated_data"]) == 0:
                st.error("❌ No data was extracted or processed. Please check the image quality and try again.")
                if results["extracted_text"]:
                    with st.expander("🔍 View Extracted Text"):
                        st.text_area("Extracted Text", results["extracted_text"], height=300)
                if results["parsed_data"]:
                    with st.expander("🧾 View Parsed Data"):
                        st.json(results["parsed_data"])
                return
            
            st.success("🎉 Processing completed successfully!")
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Final Results", 
                "🔍 Extracted Text", 
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
                st.subheader("🔍 Extracted Text from Policy Image (LlamaParse)")
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
