# # import streamlit as st
# # import pandas as pd
# # import logging 
# # from io import BytesIO 

# # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# # logger=logging.getLogger(__name__)

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

# # SEGMENT_MAPPING = {
# #     "PCV 3W (Non diesel)": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "Non diesel"},
# #     "School Bus": {"LOB": "BUS", "Normalized Segment": "SCHOOL BUS", "Remarks": ""},
# #     "PCV TAXI <6 Str ( NND)": {"LOB": "TAXI", "Normalized Segment": "TAXI", "Remarks": "NND"},
# #     "GCV < 2.5 K TATA / Maruti & Mahindra - Jeeto, Supro & Maxximo": {"LOB": "CV", "Normalized Segment": "Upto 2.5 GVW", "Remarks": "TATA / Maruti & Mahindra - Jeeto, Supro & Maxximo"},
# #     "GCV < 2.5 K Others": {"LOB": "CV", "Normalized Segment": "Upto 2.5 GVW", "Remarks": "Others"},
# #     "GCV 2.5K - 3.5K": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": ""},
# #     "GCV 3.5K-7.5K": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": ""},
# #     "GCV 7.5K-12K": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": ""},
# #     "GCV 12K-20K Tata / AL/ Eicher Make": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "Tata / AL/ Eicher Make"},
# #     "GCV 12K-20K All Make": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "All Make"},
# #     "GCV 20K-40K Tata / AL/ Eicher Make": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "Tata / AL/ Eicher Make"},
# #     "GCV 20K-40K All Make": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "All Make"},
# #     "GCV 40K - 50K ( All Age Band ) Tata / AL/ Eicher Make": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "All Age Band, Tata / AL/ Eicher Make"},
# #     "GCV 40K - 50K ( All Age Band ) All Make": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "All Age Band, All Make"},
# #     "GCV > 50K ( All Age Band ) Tata / AL/ Eicher Make": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "All Age Band, Tata / AL/ Eicher Make"},
# #     "GCV > 50K ( All Age Band ) All Make": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "All Age Band, All Make"},
# #     "GCV 3W Non Electric": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "Non Electric"},
# #     "GCV 3W -Electric": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "Electric"},
# #     "PCV Taxi 7+1 ( Only NND ) ( Maruti, Toyota, Mahindra, Kia, MG ) All Fuel Type": {"LOB": "TAXI", "Normalized Segment": "TAXI", "Remarks": "Only NND, Maruti, Toyota, Mahindra, Kia, MG, All Fuel Type"},
# #     "PCV Other bus > 8 STR & PCV 7+1 ( Other makes )": {"LOB": "TAXI", "Normalized Segment": "TAXI", "Remarks": "Other makes, 7+1 mostly taxi seating"},
# #     "Car Carrier": {"LOB": "MISD", "Normalized Segment": "Misd, Tractor", "Remarks": ""},
# #     "Flat Bed": {"LOB": "MISD", "Normalized Segment": "Misd, Tractor", "Remarks": ""},
# #     "MISD CPM (Only JCB, L&T and Caterpillar)": {"LOB": "MISD", "Normalized Segment": "Misd, Tractor", "Remarks": "Only JCB, L&T and Caterpillar"},
# #     "Tractor - (Agricultural without Trailer)": {"LOB": "MISD", "Normalized Segment": "Misd, Tractor", "Remarks": "Agricultural without Trailer"},
# #     "Bikes<150 cc Non Fresh": {"LOB": "TW", "Normalized Segment": "TW SAOD + COMP", "Remarks": "Non Fresh; If COMP then check TW COMP + SAOD; If SAOD then STP=TP"},
# #     "Scooters (Excluding Yamaha & EV)": {"LOB": "TW", "Normalized Segment": "TW SAOD + COMP", "Remarks": "Excluding Yamaha & EV; Fresh or 1+5 or 5+5 check 1+5 in TW LOB"},
# #     "PC STP ( Non Diesel Only )": {"LOB": "PVT CAR", "Normalized Segment": "PVT CAR TP", "Remarks": "Non Diesel Only"},
# #     "Private Car ( Comprehensive ) Payout on OD Premium Only": {"LOB": "PVT CAR", "Normalized Segment": "PVT CAR COMP + SAOD", "Remarks": "Comprehensive, Payout on OD Premium Only"},
# #     "Private Car ( SA OD ) Payout on OD Premium Only": {"LOB": "PVT CAR", "Normalized Segment": "PVT CAR COMP + SAOD", "Remarks": "SA OD, Payout on OD Premium Only"},
# #     "Employee Pick Up ( Only Deal Based )": {"LOB": "BUS", "Normalized Segment": "STAFF BUS", "Remarks": "Only Deal Based"}
# # }

# # st.title("Reliance Payout Processing")

# # def process_excel(file_buffer):
# #      try:
# #            df=pd.read_excel(file_buffer)
# #            logger.info("Excel file read into DataFrame.")
# #            company_name="Reliance"
# #            output_data=[]
# #            location_cols = df.columns[:3]
# #            payout_col=df.columns[3:]

# #         #    Process rows
# #            for row_idx in range(len(df)):
# #              row=df.iloc[row_idx]
# #              zone=row[0]
# #              rto_region=row[1]
# #              specific=row[2]
# #              location=f"{zone} {rto_region} {specific}"
                                                                         
# #            for col_idx, col in enumerate(payout_col):
# #                  segment=col[0]
# #                  policy_type=col[1]
# #                  payin=row[col]

          
# #      except Exception as e:
# #           logger.error(f"Error in process_excel: {e}")


# # uploaded_file = st.file_uploader("Upload you file here",type=["xlsx","xls"])

# # if uploaded_file:
# #     try:
# #         file_buffer=BytesIO(uploaded_file.read())
# #         logger.info("File uploaded successfully.")
# #         st.write("File uploaded successfully. ")

# #         df = pd.read_excel(file_buffer)
# #         st.write("Input Data Preview: ")
# #         st.dataframe(df)

# #         file_buffer.seek(0)
# #         output_df = process_excel(file_buffer)

# #         if output_df:
# #             st.write("Processed Data Preview: ")
# #             st.dataframe(output_df)
# #             output_buffer=BytesIO()
# #             output_df.to_excel(output_buffer,index=False)
# #             output_buffer.seek(0)
# #             st.download_button(
# #                 label="Download Processed File",
# #                 data=output_buffer,
# #                 file_name="processed_reliance_payout.xlsx",
# #                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
# #             )
# #             logger.info("File processed and ready for download.")
# #     except Exception as e:
# #             logger.error(f"Error processing file: {e}")
# #             st.error(f"Error processing file: {e}")
        


# import streamlit as st
# import pandas as pd
# import re
# import logging
# from io import BytesIO

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Streamlit app title
# st.title("Reliance Payout Processing")

# # Define FORMULA_DATA and SEGMENT_MAPPING as provided
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
#     {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "Rest of Companies", "PO": "90% of Payin", "REMARKS": "Zuno - 21"},
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

# SEGMENT_MAPPING = {
#     "PCV 3W (Non diesel)": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "Non diesel"},
#     "School Bus": {"LOB": "BUS", "Normalized Segment": "SCHOOL BUS", "Remarks": ""},
#     "PCV TAXI <6 Str ( NND)": {"LOB": "TAXI", "Normalized Segment": "TAXI", "Remarks": "NND"},
#     "GCV < 2.5 K TATA / Maruti & Mahindra - Jeeto, Supro & Maxximo": {"LOB": "CV", "Normalized Segment": "Upto 2.5 GVW", "Remarks": "TATA / Maruti & Mahindra - Jeeto, Supro & Maxximo"},
#     "GCV < 2.5 K Others": {"LOB": "CV", "Normalized Segment": "Upto 2.5 GVW", "Remarks": "Others"},
#     "GCV 2.5K - 3.5K": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": ""},
#     "GCV 3.5K-7.5K": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": ""},
#     "GCV 7.5K-12K": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": ""},
#     "GCV 12K-20K Tata / AL/ Eicher Make": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "Tata / AL/ Eicher Make"},
#     "GCV 12K-20K All Make": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "All Make"},
#     "GCV 20K-40K Tata / AL/ Eicher Make": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "Tata / AL/ Eicher Make"},
#     "GCV 20K-40K All Make": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "All Make"},
#     "GCV 40K - 50K ( All Age Band ) Tata / AL/ Eicher Make": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "All Age Band, Tata / AL/ Eicher Make"},
#     "GCV 40K - 50K ( All Age Band ) All Make": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "All Age Band, All Make"},
#     "GCV > 50K ( All Age Band ) Tata / AL/ Eicher Make": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "All Age Band, Tata / AL/ Eicher Make"},
#     "GCV > 50K ( All Age Band ) All Make": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "All Age Band, All Make"},
#     "GCV 3W Non Electric": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "Non Electric"},
#     "GCV 3W -Electric": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "Electric"},
#     "PCV Taxi 7+1 ( Only NND ) ( Maruti, Toyota, Mahindra, Kia, MG ) All Fuel Type": {"LOB": "TAXI", "Normalized Segment": "TAXI", "Remarks": "Only NND, Maruti, Toyota, Mahindra, Kia, MG, All Fuel Type"},
#     "PCV Other bus > 8 STR & PCV 7+1 ( Other makes )": {"LOB": "TAXI", "Normalized Segment": "TAXI", "Remarks": "Other makes, 7+1 mostly taxi seating"},
#     "Car Carrier": {"LOB": "MISD", "Normalized Segment": "Misd, Tractor", "Remarks": ""},
#     "Flat Bed": {"LOB": "MISD", "Normalized Segment": "Misd, Tractor", "Remarks": ""},
#     "MISD CPM (Only JCB, L&T and Caterpillar)": {"LOB": "MISD", "Normalized Segment": "Misd, Tractor", "Remarks": "Only JCB, L&T and Caterpillar"},
#     "Tractor - (Agricultural without Trailer)": {"LOB": "MISD", "Normalized Segment": "Misd, Tractor", "Remarks": "Agricultural without Trailer"},
#     "Bikes<150 cc Non Fresh": {"LOB": "TW", "Normalized Segment": "TW SAOD + COMP", "Remarks": "Non Fresh; If COMP then check TW COMP + SAOD; If SAOD then STP=TP"},
#     "Scooters (Excluding Yamaha & EV)": {"LOB": "TW", "Normalized Segment": "TW SAOD + COMP", "Remarks": "Excluding Yamaha & EV; Fresh or 1+5 or 5+5 check 1+5 in TW LOB"},
#     "PC STP ( Non Diesel Only )": {"LOB": "PVT CAR", "Normalized Segment": "PVT CAR TP", "Remarks": "Non Diesel Only"},
#     "Private Car ( Comprehensive ) Payout on OD Premium Only": {"LOB": "PVT CAR", "Normalized Segment": "PVT CAR COMP + SAOD", "Remarks": "Comprehensive, Payout on OD Premium Only"},
#     "Private Car ( SA OD ) Payout on OD Premium Only": {"LOB": "PVT CAR", "Normalized Segment": "PVT CAR COMP + SAOD", "Remarks": "SA OD, Payout on OD Premium Only"},
#     "Employee Pick Up ( Only Deal Based )": {"LOB": "BUS", "Normalized Segment": "STAFF BUS", "Remarks": "Only Deal Based"}
# }

# # Function to calculate payout based on formula
# def calculate_payout(payin, po_formula):
#     try:
#         payin = float(payin)  # Convert payin to float
#         if '% of Payin' in po_formula:
#             percentage = float(re.search(r'(\d+)%', po_formula).group(1)) / 100
#             return payin * percentage
#         elif po_formula.startswith('-') and '%' in po_formula:
#             deduction = float(po_formula.strip('%')) / 100
#             return payin + deduction  # Negative deduction subtracts
#         elif 'Less 2% of Payin' in po_formula:
#             return payin * 0.98
#         else:
#             logger.warning(f"Unknown PO formula: {po_formula}, returning payin")
#             return payin
#     except Exception as e:
#         logger.error(f"Error calculating payout for payin {payin} and formula {po_formula}: {e}")
#         return 0.0

# # Function to find matching formula rule
# def find_matching_rule(lob, normalized_segment, insurer, payin):
#     try:
#         payin_pct = float(payin) * 100  # Convert to percentage
#         candidates = []
#         for rule in FORMULA_DATA:
#             if rule['LOB'] == lob and rule['SEGMENT'] == normalized_segment:
#                 if 'All Companies' in rule['INSURER'] or insurer in rule['INSURER'].split(', '):
#                     candidates.append(rule)
#                 elif 'Rest of Companies' in rule['INSURER'] and insurer not in [i.strip() for sub in [r['INSURER'].split(', ') for r in FORMULA_DATA if 'Rest of Companies' not in r['INSURER']] for i in sub]:
#                     candidates.append(rule)
        
#         # Filter by payin conditions
#         matching_rule = None
#         for cand in candidates:
#             rem = cand['REMARKS']
#             if 'Below 20%' in rem and payin_pct < 20:
#                 matching_rule = cand
#                 break
#             elif '21% to 30%' in rem and 21 <= payin_pct <= 30:
#                 matching_rule = cand
#                 break
#             elif '31% to 50%' in rem and 31 <= payin_pct <= 50:
#                 matching_rule = cand
#                 break
#             elif 'Above 50%' in rem and payin_pct > 50:
#                 matching_rule = cand
#                 break
#             elif 'Above 20%' in rem and payin_pct > 20:
#                 matching_rule = cand
#                 break
#             elif rem == 'NIL' or not rem.startswith('Payin'):
#                 matching_rule = cand
#                 break
        
#         if matching_rule:
#             return matching_rule['PO'], matching_rule['REMARKS']
#         else:
#             for cand in candidates:
#                 if 'All Companies' in cand['INSURER']:
#                     return cand['PO'], cand['REMARKS']
#             logger.warning(f"No matching rule for LOB: {lob}, Segment: {normalized_segment}, Insurer: {insurer}, Payin: {payin}")
#             return "0%", "No matching rule"
#     except Exception as e:
#         logger.error(f"Error finding matching rule for LOB: {lob}, Segment: {normalized_segment}, Insurer: {insurer}: {e}")
#         return "0%", "Error in rule matching"

# # Function to process the Excel
# def process_excel(file_buffer):
#     try:
#         # Read the sheet with multi-header
#         df = pd.read_excel(file_buffer, sheet_name='Oct 25', header=[0, 1])
#         logger.info("Excel sheet 'Oct 25' read successfully")
        
#         # Company name (hardcoded as Reliance for now)
#         company_name = 'Reliance'
        
#         # Prepare output list
#         output_data = []
        
#         # The columns after the first 3 are the payout columns
#         location_cols = df.columns[:3]  # Zone, RTO Region, RTO Region
#         payout_cols = df.columns[3:]
        
#         # Process each row
#         for row_idx in range(len(df)):
#             row = df.iloc[row_idx]
#             zone = row[0]
#             rto_region = row[1]
#             specific = row[2]
#             location = f"{zone} - {rto_region} - {specific}"
            
#             for col_idx, col in enumerate(payout_cols):
#                 segment = col[0]  # Row1 header
#                 policy_type = col[1]  # Row2 header
#                 payin = row[col]
                
#                 if pd.isna(payin) or payin == 0:
#                     continue  # Skip zero or null payouts
                
#                 # Get mapping
#                 mapping = SEGMENT_MAPPING.get(segment, {"LOB": "Unknown", "Normalized Segment": "Unknown", "Remarks": ""})
#                 lob = mapping['LOB']
#                 normalized_segment = mapping['Normalized Segment']
#                 map_remarks = mapping['Remarks']
                
#                 # Find rule
#                 po_formula, rule_remarks = find_matching_rule(lob, normalized_segment, company_name, payin)
                
#                 # Calculate final payout
#                 payout = calculate_payout(payin, po_formula)
                
#                 # Combine remarks
#                 combined_remarks = f"{map_remarks}; {policy_type}; {rule_remarks}".strip('; ')
                
#                 output_data.append({
#                     "Company Name": company_name,
#                     "Segment": segment,
#                     "Policy Type": policy_type,
#                     "Location": location,
#                     "Payout": payout,
#                     "Remark": combined_remarks
#                 })
        
#         # Create output DataFrame
#         output_df = pd.DataFrame(output_data)
#         logger.info("Payout processing completed successfully")
#         return output_df
#     except Exception as e:
#         logger.error(f"Error processing Excel: {e}")
#         st.error(f"Error processing Excel: {e}")
#         return None

# # Streamlit file uploader
# uploaded_file = st.file_uploader("Upload your file here", type=["xlsx", "xls"])

# if uploaded_file:
#     try:
#         # Read file into memory
#         file_buffer = BytesIO(uploaded_file.read())
#         logger.info("File uploaded and read successfully")
#         st.write("File uploaded successfully!")
        
#         # Display input DataFrame
#         df = pd.read_excel(file_buffer, sheet_name='Oct 25', header=[0, 1])
#         st.write("Input Data Preview:")
#         st.dataframe(df)
        
#         # Reset buffer position
#         file_buffer.seek(0)
        
#         # Process the file
#         output_df = process_excel(file_buffer)
        
#         if output_df is not None:
#             st.write("Processed Output Preview:")
#             st.dataframe(output_df)
            
#             # Convert output to Excel for download
#             output_buffer = BytesIO()
#             output_df.to_excel(output_buffer, index=False)
#             output_buffer.seek(0)
            
#             st.download_button(
#                 label="Download Processed Output",
#                 data=output_buffer,
#                 file_name="processed_payouts.xlsx",
#                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#             )
#             logger.info("Output ready for download")
#     except Exception as e:
#         logger.error(f"Error reading or processing the uploaded file: {e}")
#         st.error("Error reading or processing the uploaded file. Please ensure it's a valid Excel file with an 'Oct 25' sheet.")


# import streamlit as st
# import pandas as pd
# import re
# import logging
# from io import BytesIO

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Streamlit app title
# st.title("Reliance Payout Processing")

# # Define FORMULA_DATA and SEGMENT_MAPPING as provided
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
#     {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "Rest of Companies", "PO": "90% of Payin", "REMARKS": "Zuno - 21"},
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

# SEGMENT_MAPPING = {
#     "PCV 3W (Non diesel)": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "Non diesel"},
#     "School Bus": {"LOB": "BUS", "Normalized Segment": "SCHOOL BUS", "Remarks": ""},
#     "PCV TAXI <6 Str ( NND)": {"LOB": "TAXI", "Normalized Segment": "TAXI", "Remarks": "NND"},
#     "GCV < 2.5 K TATA / Maruti & Mahindra - Jeeto, Supro & Maxximo": {"LOB": "CV", "Normalized Segment": "Upto 2.5 GVW", "Remarks": "TATA / Maruti & Mahindra - Jeeto, Supro & Maxximo"},
#     "GCV < 2.5 K Others": {"LOB": "CV", "Normalized Segment": "Upto 2.5 GVW", "Remarks": "Others"},
#     "GCV 2.5K - 3.5K": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": ""},
#     "GCV 3.5K-7.5K": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": ""},
#     "GCV 7.5K-12K": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": ""},
#     "GCV 12K-20K Tata / AL/ Eicher Make": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "Tata / AL/ Eicher Make"},
#     "GCV 12K-20K All Make": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "All Make"},
#     "GCV 20K-40K Tata / AL/ Eicher Make": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "Tata / AL/ Eicher Make"},
#     "GCV 20K-40K All Make": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "All Make"},
#     "GCV 40K - 50K ( All Age Band ) Tata / AL/ Eicher Make": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "All Age Band, Tata / AL/ Eicher Make"},
#     "GCV 40K - 50K ( All Age Band ) All Make": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "All Age Band, All Make"},
#     "GCV > 50K ( All Age Band ) Tata / AL/ Eicher Make": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "All Age Band, Tata / AL/ Eicher Make"},
#     "GCV > 50K ( All Age Band ) All Make": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "All Age Band, All Make"},
#     "GCV 3W Non Electric": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "Non Electric"},
#     "GCV 3W -Electric": {"LOB": "CV", "Normalized Segment": "All GVW & PCV 3W, GCV 3W", "Remarks": "Electric"},
#     "PCV Taxi 7+1 ( Only NND ) ( Maruti, Toyota, Mahindra, Kia, MG ) All Fuel Type": {"LOB": "TAXI", "Normalized Segment": "TAXI", "Remarks": "Only NND, Maruti, Toyota, Mahindra, Kia, MG, All Fuel Type"},
#     "PCV Other bus > 8 STR & PCV 7+1 ( Other makes )": {"LOB": "TAXI", "Normalized Segment": "TAXI", "Remarks": "Other makes, 7+1 mostly taxi seating"},
#     "Car Carrier": {"LOB": "MISD", "Normalized Segment": "Misd, Tractor", "Remarks": ""},
#     "Flat Bed": {"LOB": "MISD", "Normalized Segment": "Misd, Tractor", "Remarks": ""},
#     "MISD CPM (Only JCB, L&T and Caterpillar)": {"LOB": "MISD", "Normalized Segment": "Misd, Tractor", "Remarks": "Only JCB, L&T and Caterpillar"},
#     "Tractor - (Agricultural without Trailer)": {"LOB": "MISD", "Normalized Segment": "Misd, Tractor", "Remarks": "Agricultural without Trailer"},
#     "Bikes<150 cc Non Fresh": {"LOB": "TW", "Normalized Segment": "TW SAOD + COMP", "Remarks": "Non Fresh; If COMP then check TW COMP + SAOD; If SAOD then STP=TP"},
#     "Scooters (Excluding Yamaha & EV)": {"LOB": "TW", "Normalized Segment": "TW SAOD + COMP", "Remarks": "Excluding Yamaha & EV; Fresh or 1+5 or 5+5 check 1+5 in TW LOB"},
#     "PC STP ( Non Diesel Only )": {"LOB": "PVT CAR", "Normalized Segment": "PVT CAR TP", "Remarks": "Non Diesel Only"},
#     "Private Car ( Comprehensive ) Payout on OD Premium Only": {"LOB": "PVT CAR", "Normalized Segment": "PVT CAR COMP + SAOD", "Remarks": "Comprehensive, Payout on OD Premium Only"},
#     "Private Car ( SA OD ) Payout on OD Premium Only": {"LOB": "PVT CAR", "Normalized Segment": "PVT CAR COMP + SAOD", "Remarks": "SA OD, Payout on OD Premium Only"},
#     "Employee Pick Up ( Only Deal Based )": {"LOB": "BUS", "Normalized Segment": "STAFF BUS", "Remarks": "Only Deal Based"}
# }

# # Function to calculate payout based on formula
# def calculate_payout(payin, po_formula):
#     try:
#         payin = float(payin)  # Convert payin to float
#         if '% of Payin' in po_formula:
#             percentage = float(re.search(r'(\d+)%', po_formula).group(1)) / 100
#             return payin * percentage
#         elif po_formula.startswith('-') and '%' in po_formula:
#             deduction = float(po_formula.strip('%')) / 100
#             return payin + deduction  # deduction is negative
#         elif 'Less 2% of Payin' in po_formula:
#             return payin * 0.98
#         else:
#             logger.warning(f"Unknown PO formula: {po_formula}, returning payin")
#             return payin
#     except Exception as e:
#         logger.error(f"Error calculating payout for payin {payin} and formula {po_formula}: {e}")
#         return 0.0

# # Function to find matching formula rule
# def find_matching_rule(lob, normalized_segment, insurer, payin):
#     try:
#         payin_pct = float(payin) * 100  # Convert to percentage
#         candidates = []
#         for rule in FORMULA_DATA:
#             if rule['LOB'] == lob and rule['SEGMENT'] == normalized_segment:
#                 if 'All Companies' in rule['INSURER'] or insurer in rule['INSURER'].split(', '):
#                     candidates.append(rule)
#                 elif 'Rest of Companies' in rule['INSURER'] and insurer not in [i.strip() for sub in [r['INSURER'].split(', ') for r in FORMULA_DATA if 'Rest of Companies' not in r['INSURER']] for i in sub]:
#                     candidates.append(rule)
        
#         # Filter by payin conditions
#         matching_rule = None
#         for cand in candidates:
#             rem = cand['REMARKS']
#             if 'Below 20%' in rem and payin_pct < 20:
#                 matching_rule = cand
#                 break
#             elif '21% to 30%' in rem and 21 <= payin_pct <= 30:
#                 matching_rule = cand
#                 break
#             elif '31% to 50%' in rem and 31 <= payin_pct <= 50:
#                 matching_rule = cand
#                 break
#             elif 'Above 50%' in rem and payin_pct > 50:
#                 matching_rule = cand
#                 break
#             elif 'Above 20%' in rem and payin_pct > 20:
#                 matching_rule = cand
#                 break
#             elif rem == 'NIL' or not rem.startswith('Payin'):
#                 matching_rule = cand
#                 break
        
#         if matching_rule:
#             return matching_rule['PO'], matching_rule['REMARKS']
#         else:
#             for cand in candidates:
#                 if 'All Companies' in cand['INSURER']:
#                     return cand['PO'], cand['REMARKS']
#             logger.warning(f"No matching rule for LOB: {lob}, Segment: {normalized_segment}, Insurer: {insurer}, Payin: {payin}")
#             return "0%", "No matching rule"
#     except Exception as e:
#         logger.error(f"Error finding matching rule for LOB: {lob}, Segment: {normalized_segment}, Insurer: {insurer}: {e}")
#         return "0%", "Error in rule matching"

# # Function to process the Excel
# def process_excel(file_buffer):
#     try:
#         # Read the sheet with multi-header (rows 1,2,3 as headers)
#         df = pd.read_excel(file_buffer, sheet_name='Oct 25', header=[0, 1, 2])
#         logger.info("Excel sheet 'Oct 25' read successfully")
        
#         # Company name (hardcoded as Reliance for now)
#         company_name = 'Reliance'
        
#         # Prepare output list
#         output_data = []
        
#         # The columns after the first 2 are the payout columns (Zone and RTO Region)
#         location_cols = df.columns[:2]  # ('Zone', 'Zone', 'Zone'), ('RTO Region', 'RTO Region', 'RTO Region')
#         payout_cols = df.columns[2:]
        
#         # Process each row
#         for row_idx in range(len(df)):
#             row = df.iloc[row_idx]
#             zone = row[location_cols[0]]
#             rto_region = row[location_cols[1]]
#             location = f"{zone} - {rto_region}"
            
#             for col_idx, col in enumerate(payout_cols):
#                 broad_category = col[0]  # Row1 (broad category)
#                 segment = col[1]  # Row2 (specific segment for mapping)
#                 policy_type = col[2]  # Row3 (COMP, STP, etc.)
#                 payin = row[col]
                
#                 if pd.isna(payin) or payin == 0:
#                     continue  # Skip zero or null payouts
                
#                 # Get mapping
#                 mapping = SEGMENT_MAPPING.get(segment, {"LOB": "Unknown", "Normalized Segment": "Unknown", "Remarks": ""})
#                 lob = mapping['LOB']
#                 normalized_segment = mapping['Normalized Segment']
#                 map_remarks = mapping['Remarks']
                
#                 # Find rule
#                 po_formula, rule_remarks = find_matching_rule(lob, normalized_segment, company_name, payin)
                
#                 # Calculate final payout
#                 payout = calculate_payout(payin, po_formula)
                
#                 # Combine remarks
#                 combined_remarks = f"{map_remarks}; {broad_category}; {policy_type}; {rule_remarks}".strip('; ')
                
#                 output_data.append({
#                     "Company Name": company_name,
#                     "Segment": segment,
#                     "Policy Type": policy_type,
#                     "Location": location,
#                     "Payout": payout,
#                     "Remark": combined_remarks
#                 })
        
#         # Create output DataFrame
#         output_df = pd.DataFrame(output_data)
#         logger.info("Payout processing completed successfully")
#         return output_df
#     except Exception as e:
#         logger.error(f"Error processing Excel: {e}")
#         st.error(f"Error processing Excel: {e}")
#         return None

# # Streamlit file uploader
# uploaded_file = st.file_uploader("Upload your file here", type=["xlsx", "xls"])

# if uploaded_file:
#     try:
#         # Read file into memory
#         file_buffer = BytesIO(uploaded_file.read())
#         logger.info("File uploaded and read successfully")
#         st.write("File uploaded successfully!")
        
#         # Display input DataFrame
#         df = pd.read_excel(file_buffer, sheet_name='Oct 25', header=[0, 1, 2])
#         st.write("Input Data Preview:")
#         st.dataframe(df)
        
#         # Reset buffer position
#         file_buffer.seek(0)
        
#         # Process the file
#         output_df = process_excel(file_buffer)
        
#         if output_df is not None:
#             st.write("Processed Output Preview:")
#             st.dataframe(output_df)
            
#             # Convert output to Excel for download
#             output_buffer = BytesIO()
#             output_df.to_excel(output_buffer, index=False)
#             output_buffer.seek(0)
            
#             st.download_button(
#                 label="Download Processed Output",
#                 data=output_buffer,
#                 file_name="processed_payouts.xlsx",
#                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#             )
#             logger.info("Output ready for download")
#     except Exception as e:
#         logger.error(f"Error reading or processing the uploaded file: {e}")
#         st.error("Error reading or processing the uploaded file. Please ensure it's a valid Excel file with an 'Oct 25' sheet.")


# import streamlit as st
# import pandas as pd
# import re
# import logging
# from io import BytesIO

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Streamlit app title
# st.title("Reliance Payout Processing")

# # Define FORMULA_DATA as provided
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
#     {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "Rest of Companies", "PO": "90% of Payin", "REMARKS": "Zuno - 21"},
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

# # Derive LOB_DICT from FORMULA_DATA
# LOB_DICT = {}
# for rule in FORMULA_DATA:
#     if rule['SEGMENT'] not in LOB_DICT:
#         LOB_DICT[rule['SEGMENT']] = rule['LOB']

# # Function to calculate payout based on formula
# def calculate_payout(payin, po_formula):
#     try:
#         payin = float(payin)  # Convert payin to float
#         if '% of Payin' in po_formula:
#             percentage = float(re.search(r'(\d+)%', po_formula).group(1)) / 100
#             return payin * percentage
#         elif po_formula.startswith('-') and '%' in po_formula:
#             deduction = float(po_formula.strip('%')) / 100
#             return payin + deduction  # deduction is negative
#         elif 'Less 2% of Payin' in po_formula:
#             return payin * 0.98
#         else:
#             logger.warning(f"Unknown PO formula: {po_formula}, returning payin")
#             return payin
#     except Exception as e:
#         logger.error(f"Error calculating payout for payin {payin} and formula {po_formula}: {e}")
#         return 0.0

# # Function to find matching formula rule
# def find_matching_rule(lob, normalized_segment, insurer, payin):
#     try:
#         payin_pct = float(payin) * 100  # Convert to percentage
#         candidates = []
#         for rule in FORMULA_DATA:
#             if rule['LOB'] == lob and rule['SEGMENT'] == normalized_segment:
#                 if 'All Companies' in rule['INSURER'] or insurer in rule['INSURER'].split(', '):
#                     candidates.append(rule)
#                 elif 'Rest of Companies' in rule['INSURER'] and insurer not in [i.strip() for sub in [r['INSURER'].split(', ') for r in FORMULA_DATA if 'Rest of Companies' not in r['INSURER']] for i in sub]:
#                     candidates.append(rule)
        
#         # Filter by payin conditions
#         matching_rule = None
#         for cand in candidates:
#             rem = cand['REMARKS']
#             if 'Below 20%' in rem and payin_pct < 20:
#                 matching_rule = cand
#                 break
#             elif '21% to 30%' in rem and 21 <= payin_pct <= 30:
#                 matching_rule = cand
#                 break
#             elif '31% to 50%' in rem and 31 <= payin_pct <= 50:
#                 matching_rule = cand
#                 break
#             elif 'Above 50%' in rem and payin_pct > 50:
#                 matching_rule = cand
#                 break
#             elif 'Above 20%' in rem and payin_pct > 20:
#                 matching_rule = cand
#                 break
#             elif rem == 'NIL' or not rem.startswith('Payin'):
#                 matching_rule = cand
#                 break
        
#         if matching_rule:
#             return matching_rule['PO'], matching_rule['REMARKS']
#         else:
#             for cand in candidates:
#                 if 'All Companies' in cand['INSURER']:
#                     return cand['PO'], cand['REMARKS']
#             logger.warning(f"No matching rule for LOB: {lob}, Segment: {normalized_segment}, Insurer: {insurer}, Payin: {payin}")
#             return "0%", "No matching rule"
#     except Exception as e:
#         logger.error(f"Error finding matching rule for LOB: {lob}, Segment: {normalized_segment}, Insurer: {insurer}: {e}")
#         return "0%", "Error in rule matching"

# # Function to process the Excel
# def process_excel(file_buffer):
#     try:
#         # Read the sheet with multi-header (rows 1,2,3 as headers, 0-indexed 0,1,2)
#         df = pd.read_excel(file_buffer, sheet_name='Oct 25', header=[0, 1, 2])
#         logger.info("Excel sheet 'Oct 25' read successfully")
        
#         # Company name (hardcoded as Reliance for now)
#         company_name = 'Reliance'
        
#         # Prepare output list
#         output_data = []
        
#         # The columns after the first 2 are the payout columns (Zone and RTO Region)
#         location_cols = df.columns[:2]  # ('Zone', 'Zone', 'Zone'), ('RTO Region', 'RTO Region', 'RTO Region')
#         payout_cols = df.columns[2:]
        
#         # Process each row
#         for row_idx in range(len(df)):
#             row = df.iloc[row_idx]
#             zone = row[location_cols[0]]
#             rto_region = row[location_cols[1]]
#             location = f"{zone} - {rto_region}"
            
#             for col_idx, col in enumerate(payout_cols):
#                 segment = col[0]  # Row1: the segment to search in FORMULA_DATA, e.g., "All GVW & PCV 3W, GCV 3W"
#                 sub_segment = col[1]  # Row2: specific details, e.g., "PCV 3W (Non diesel)"
#                 policy_type = col[2]  # Row3: e.g., "COMP"
#                 payin = row[col]
                
#                 if pd.isna(payin) or payin == 0:
#                     continue  # Skip zero or null payouts
                
#                 # Get LOB from LOB_DICT using segment
#                 lob = LOB_DICT.get(segment, "Unknown")
                
#                 # Use segment directly as normalized_segment
#                 normalized_segment = segment
                
#                 # Find rule
#                 po_formula, rule_remarks = find_matching_rule(lob, normalized_segment, company_name, payin)
                
#                 # Calculate final payout
#                 payout = calculate_payout(payin, po_formula)
                
#                 # Combine remarks
#                 combined_remarks = f"{sub_segment}; {policy_type}; {rule_remarks}".strip('; ')
                
#                 output_data.append({
#                     "Company Name": company_name,
#                     "Segment": sub_segment,
#                     "Policy Type": policy_type,
#                     "Location": location,
#                     "Payout": payout,
#                     "Remark": combined_remarks
#                 })
        
#         # Create output DataFrame
#         output_df = pd.DataFrame(output_data)
#         logger.info("Payout processing completed successfully")
#         return output_df
#     except Exception as e:
#         logger.error(f"Error processing Excel: {e}")
#         st.error(f"Error processing Excel: {e}")
#         return None

# # Streamlit file uploader
# uploaded_file = st.file_uploader("Upload your file here", type=["xlsx", "xls"])

# if uploaded_file:
#     try:
#         # Read file into memory
#         file_buffer = BytesIO(uploaded_file.read())
#         logger.info("File uploaded and read successfully")
#         st.write("File uploaded successfully!")
        
#         # Display input DataFrame
#         df = pd.read_excel(file_buffer, sheet_name='Oct 25', header=[0, 1, 2])
#         st.write("Input Data Preview:")
#         st.dataframe(df)
        
#         # Reset buffer position
#         file_buffer.seek(0)
        
#         # Process the file
#         output_df = process_excel(file_buffer)
        
#         if output_df is not None:
#             st.write("Processed Output Preview:")
#             st.dataframe(output_df)
            
#             # Convert output to Excel for download
#             output_buffer = BytesIO()
#             output_df.to_excel(output_buffer, index=False)
#             output_buffer.seek(0)
            
#             st.download_button(
#                 label="Download Processed Output",
#                 data=output_buffer,
#                 file_name="processed_payouts.xlsx",
#                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#             )
#             logger.info("Output ready for download")
#     except Exception as e:
#         logger.error(f"Error reading or processing the uploaded file: {e}")
#         st.error("Error reading or processing the uploaded file. Please ensure it's a valid Excel file with an 'Oct 25' sheet.")


# import streamlit as st
# import pandas as pd
# import re
# import logging
# from io import BytesIO

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Streamlit app title
# st.title("Reliance Payout Processing")

# # Define FORMULA_DATA as provided
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
#     {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "Rest of Companies", "PO": "90% of Payin", "REMARKS": "Zuno - 21"},
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

# # Derive LOB_DICT from FORMULA_DATA for quick LOB lookup by segment
# LOB_DICT = {}
# for rule in FORMULA_DATA:
#     if rule['SEGMENT'] not in LOB_DICT:
#         LOB_DICT[rule['SEGMENT']] = rule['LOB']

# # Function to calculate payout based on PO formula
# def calculate_payout(payin, po_formula):
#     try:
#         payin = float(payin)
#         if '% of Payin' in po_formula:
#             percentage = float(re.search(r'(\d+)%', po_formula).group(1)) / 100
#             return payin * percentage
#         elif po_formula.startswith('-') and '%' in po_formula:
#             deduction = float(po_formula.strip('%')) / 100
#             return payin + deduction  # Since deduction is negative
#         elif 'Less 2% of Payin' in po_formula:
#             return payin * 0.98
#         else:
#             logger.warning(f"Unknown PO formula: {po_formula}, returning payin")
#             return payin
#     except Exception as e:
#         logger.error(f"Error calculating payout for payin {payin} and formula {po_formula}: {e}")
#         return 0.0

# # Function to find matching rule in FORMULA_DATA
# def find_matching_rule(lob, segment, insurer, payin):
#     try:
#         payin_pct = float(payin) * 100
#         # Extract candidates with matching LOB and SEGMENT
#         candidates = [
#             rule for rule in FORMULA_DATA
#             if rule['LOB'] == lob and rule['SEGMENT'] == segment
#         ]

#         # Filter candidates by insurer
#         filtered_candidates = []
#         for rule in candidates:
#             insurers = [i.strip() for i in rule['INSURER'].split(',')]
#             if 'All Companies' in insurers or insurer in insurers:
#                 filtered_candidates.append(rule)
#             elif 'Rest of Companies' in insurers:
#                 # Check if insurer is not in any specific list
#                 specific_insurers = set()
#                 for r in FORMULA_DATA:
#                     if r['LOB'] == lob and r['SEGMENT'] == segment and 'Rest of Companies' not in r['INSURER']:
#                         specific_insurers.update([i.strip() for i in r['INSURER'].split(',')])
#                 if insurer not in specific_insurers:
#                     filtered_candidates.append(rule)

#         # Find matching rule based on REMARKS and payin conditions
#         matching_rule = None
#         for cand in filtered_candidates:
#             rem = cand['REMARKS']
#             if 'Below 20%' in rem and payin_pct < 20:
#                 matching_rule = cand
#                 break
#             elif '21% to 30%' in rem and 21 <= payin_pct <= 30:
#                 matching_rule = cand
#                 break
#             elif '31% to 50%' in rem and 31 <= payin_pct <= 50:
#                 matching_rule = cand
#                 break
#             elif 'Above 50%' in rem and payin_pct > 50:
#                 matching_rule = cand
#                 break
#             elif 'Above 20%' in rem and payin_pct > 20:
#                 matching_rule = cand
#                 break
#             elif rem == 'NIL' or not rem.startswith('Payin'):
#                 matching_rule = cand
#                 break

#         if matching_rule:
#             return matching_rule['PO'], matching_rule['REMARKS']
#         else:
#             # Fallback to 'All Companies' if available
#             for cand in filtered_candidates:
#                 if 'All Companies' in cand['INSURER']:
#                     return cand['PO'], cand['REMARKS']
#             logger.warning(f"No matching rule for LOB: {lob}, Segment: {segment}, Insurer: {insurer}, Payin: {payin}")
#             return "0%", "No matching rule"
#     except Exception as e:
#         logger.error(f"Error finding matching rule: {e}")
#         return "0%", "Error in rule matching"

# # Function to process the Excel file
# def process_excel(file_buffer):
#     try:
#         # Read the sheet with multi-level headers (rows 1,2,3)
#         df = pd.read_excel(file_buffer, sheet_name='Sheet1', header=[0, 1, 2])
#         logger.info("Excel sheet 'Sheet1' read successfully")

#         # Hardcoded company name
#         company_name = 'Reliance'

#         # Prepare output list
#         output_data = []

#         # Location columns (first two)
#         location_cols = df.columns[:2]

#         # Payout columns (from third onwards)
#         payout_cols = df.columns[2:]

#         # Process each data row (starting from row 4 in original, iloc[0] after headers)
#         for row_idx in range(len(df)):
#             row = df.iloc[row_idx]
#             zone = row[location_cols[0]]
#             rto_region = row[location_cols[1]]
#             location = f"{zone}: {rto_region}"

#             for col in payout_cols:
#                 segment = col[0]  # Row 1: segment for FORMULA_DATA lookup
#                 sub_segment = col[1]  # Row 2: remark part
#                 policy_type = col[2]  # Row 3: remark part
#                 payin_value = row[col]

#                 if pd.isna(payin_value):
#                     continue  # Skip NaN values

#                 payin = float(payin_value)

#                 if payin == 0:
#                     payout = 0.0
#                     rule_remarks = "Payin is 0"
#                 else:
#                     # Get LOB
#                     lob = LOB_DICT.get(segment, "Unknown")

#                     # Use segment directly
#                     normalized_segment = segment

#                     # Find matching rule
#                     po_formula, rule_remarks = find_matching_rule(lob, normalized_segment, company_name, payin)

#                     # Calculate payout
#                     payout = calculate_payout(payin, po_formula)

#                 # Combine remarks (row2 and row3 + rule remarks)
#                 combined_remarks = f"{sub_segment}; {policy_type}; {rule_remarks}".strip('; ')

#                 # Append to output
#                 output_data.append({
#                     "Company Name": company_name,
#                     "Segment": segment,
#                     "Sub Segment": sub_segment,
#                     "Policy Type": policy_type,
#                     "Location": location,
#                     "Payin": payin,
#                     "Payout": payout,
#                     "Remarks": combined_remarks
#                 })

#         # Create output DataFrame
#         output_df = pd.DataFrame(output_data)
#         logger.info("Payout processing completed successfully")
#         return output_df
#     except Exception as e:
#         logger.error(f"Error processing Excel: {e}")
#         st.error(f"Error processing Excel: {e}")
#         return None

# # Streamlit file uploader
# uploaded_file = st.file_uploader("Upload your file here", type=["xlsx", "xls"])

# if uploaded_file:
#     try:
#         # Read file into memory
#         file_buffer = BytesIO(uploaded_file.read())
#         logger.info("File uploaded and read successfully")
#         st.write("File uploaded successfully!")

#         # Display input DataFrame
#         df = pd.read_excel(file_buffer, sheet_name='Sheet1', header=[0, 1, 2])
#         st.write("Input Data Preview:")
#         st.dataframe(df)

#         # Reset buffer position
#         file_buffer.seek(0)

#         # Process the file
#         output_df = process_excel(file_buffer)

#         if output_df is not None:
#             st.write("Processed Output Preview:")
#             st.dataframe(output_df)

#             # Convert output to Excel for download
#             output_buffer = BytesIO()
#             output_df.to_excel(output_buffer, index=False)
#             output_buffer.seek(0)

#             st.download_button(
#                 label="Download Processed Output",
#                 data=output_buffer,
#                 file_name="processed_payouts.xlsx",
#                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#             )
#             logger.info("Output ready for download")
#     except Exception as e:
#         logger.error(f"Error reading or processing the uploaded file: {e}")
#         st.error("Error reading or processing the uploaded file. Please ensure it's a valid Excel file with a 'Sheet1' sheet.")


# import streamlit as st
# import pandas as pd
# import re
# import logging
# from io import BytesIO

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Streamlit app title
# st.title("Reliance Payout Processing")

# # Define FORMULA_DATA as provided
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
#     {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "Rest of Companies", "PO": "90% of Payin", "REMARKS": "Zuno - 21"},
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

# # Derive LOB_DICT from FORMULA_DATA for quick LOB lookup by segment
# LOB_DICT = {rule['SEGMENT']: rule['LOB'] for rule in FORMULA_DATA}

# # Function to calculate payout based on PO formula
# # def calculate_payout(payin_pct, po_formula):
# #     try:
# #         payin = float(payin_pct) / 100  # Convert percentage to decimal
# #         if '% of Payin' in po_formula:
# #             percentage = float(re.search(r'(\d+)%', po_formula).group(1)) / 100
# #             return (payin * percentage) * 100  # Return as percentage
# #         elif po_formula.startswith('-') and '%' in po_formula:
# #             deduction = float(po_formula.strip('%'))
# #             return payin_pct + deduction  # Deduction is already in percentage
# #         elif 'Less 2% of Payin' in po_formula:
# #             return payin_pct - 2  # Interpret "Less 2% of Payin" as -2%
# #         else:
# #             logger.warning(f"Unknown PO formula: {po_formula}, returning payin")
# #             return payin_pct
# #     except Exception as e:
# #         logger.error(f"Error calculating payout for payin {payin_pct}% and formula {po_formula}: {e}")
# #         return 0.0

# def calculate_payout(payin_pct, po_formula):
#     try:
#         payin = float(payin_pct) / 100  # Convert percentage to decimal (for consistency, though not used here directly)
#         if '% of Payin' in po_formula:
#             percentage = float(re.search(r'(\d+)%', po_formula).group(1)) / 100
#             return (payin * percentage) * 100  # Return as percentage
#         elif po_formula.startswith('-') and '%' in po_formula:
#             deduction = float(po_formula.strip('%'))
#             return payin_pct + deduction  # Deduction is already in percentage
#         elif 'Less 2% of Payin' in po_formula:
#             return payin_pct - 2  # Subtract 2 percentage points as per your clarification
#         else:
#             logger.warning(f"Unknown PO formula: {po_formula}, returning payin")
#             return payin_pct
#     except Exception as e:
#         logger.error(f"Error calculating payout for payin {payin_pct}% and formula {po_formula}: {e}")
#         return 0.0
    
# # Function to find matching rule in FORMULA_DATA
# def find_matching_rule(lob, segment, insurer, payin_pct):
#     try:
#         payin_pct = float(payin_pct)
#         candidates = [rule for rule in FORMULA_DATA if rule['LOB'] == lob and rule['SEGMENT'] == segment]
        
#         # Filter candidates by insurer
#         filtered_candidates = []
#         for rule in candidates:
#             insurers = [i.strip() for i in rule['INSURER'].split(',')]
#             if 'All Companies' in insurers or insurer in insurers:
#                 filtered_candidates.append(rule)
#             elif 'Rest of Companies' in insurers:
#                 specific_insurers = set()
#                 for r in FORMULA_DATA:
#                     if r['LOB'] == lob and r['SEGMENT'] == segment and 'Rest of Companies' not in r['INSURER']:
#                         specific_insurers.update([i.strip() for i in r['INSURER'].split(',')])
#                 if insurer not in specific_insurers:
#                     filtered_candidates.append(rule)

#         # Find matching rule based on REMARKS
#         matching_rule = None
#         for cand in filtered_candidates:
#             rem = cand['REMARKS']
#             if 'Below 20%' in rem and payin_pct < 20:
#                 matching_rule = cand
#                 break
#             elif '21% to 30%' in rem and 21 <= payin_pct <= 30:
#                 matching_rule = cand
#                 break
#             elif '31% to 50%' in rem and 31 <= payin_pct <= 50:
#                 matching_rule = cand
#                 break
#             elif 'Above 50%' in rem and payin_pct > 50:
#                 matching_rule = cand
#                 break
#             elif 'Above 20%' in rem and payin_pct > 20:
#                 matching_rule = cand
#                 break
#             elif rem == 'NIL' or not rem.startswith('Payin'):
#                 matching_rule = cand
#                 break

#         if matching_rule:
#             return matching_rule['PO'], matching_rule['REMARKS']
#         else:
#             for cand in filtered_candidates:
#                 if 'All Companies' in cand['INSURER']:
#                     return cand['PO'], cand['REMARKS']
#             logger.warning(f"No matching rule for LOB: {lob}, Segment: {segment}, Insurer: {insurer}, Payin: {payin_pct}%")
#             return "0%", "No matching rule"
#     except Exception as e:
#         logger.error(f"Error finding matching rule: {e}")
#         return "0%", "Error in rule matching"

# # Function to process the Excel file
# def process_excel(file_buffer):
#     try:
#         # Read the sheet with multi-level headers (rows 1,2,3)
#         df = pd.read_excel(file_buffer, sheet_name='Sheet1', header=[0, 1, 2])
#         logger.info("Excel sheet 'Sheet1' read successfully")

#         # Hardcoded company name
#         company_name = 'Reliance'

#         # Prepare output list
#         output_data = []

#         # Location columns (first two)
#         location_cols = df.columns[:2]

#         # Payout columns (from third onwards)
#         payout_cols = df.columns[2:]

#         # Process each data row
#         for row_idx in range(len(df)):
#             row = df.iloc[row_idx]
#             zone = row[location_cols[0]]
#             rto_region = row[location_cols[1]]
#             location = f"{zone}: {rto_region}"

#             for col in payout_cols:
#                 segment = col[0]  # Row 1: segment for FORMULA_DATA lookup
#                 sub_segment = col[1]  # Row 2: remark part
#                 policy_type = col[2]  # Row 3: remark part
#                 payin_value = row[col]

#                 if pd.isna(payin_value):
#                     continue  # Skip NaN values

#                 payin_pct = float(payin_value) * 100  # Convert to percentage

#                 if payin_pct == 0:
#                     payout_pct = 0.0
#                     rule_remarks = "Payin is 0"
#                 else:
#                     # Get LOB
#                     lob = LOB_DICT.get(segment, "Unknown")

#                     # Use segment directly
#                     normalized_segment = segment

#                     # Find matching rule
#                     po_formula, rule_remarks = find_matching_rule(lob, normalized_segment, company_name, payin_pct)

#                     # Calculate payout
#                     payout_pct = calculate_payout(payin_pct, po_formula)

#                 # Combine remarks
#                 combined_remarks = f"{sub_segment}; {policy_type}; {rule_remarks}".strip('; ')

#                 # Append to output
#                 output_data.append({
#                     "Company Name": company_name,
#                     "Segment": segment,
#                     "Sub Segment": sub_segment,
#                     "Policy Type": policy_type,
#                     "Location": location,
#                     "Payin (%)": round(payin_pct, 2),
#                     "Payout (%)": round(payout_pct, 2),
#                     "Remarks": combined_remarks
#                 })

#         # Create output DataFrame
#         output_df = pd.DataFrame(output_data)
#         logger.info("Payout processing completed successfully")
#         return output_df
#     except Exception as e:
#         logger.error(f"Error processing Excel: {e}")
#         st.error(f"Error processing Excel: {e}")
#         return None

# # Streamlit file uploader
# uploaded_file = st.file_uploader("Upload your file here", type=["xlsx", "xls"])

# if uploaded_file:
#     try:
#         # Read file into memory
#         file_buffer = BytesIO(uploaded_file.read())
#         logger.info("File uploaded and read successfully")
#         st.write("File uploaded successfully!")

#         # Display input DataFrame
#         df = pd.read_excel(file_buffer, sheet_name='Sheet1', header=[0, 1, 2])
#         st.write("Input Data Preview:")
#         st.dataframe(df)

#         # Reset buffer position
#         file_buffer.seek(0)

#         # Process the file
#         output_df = process_excel(file_buffer)

#         if output_df is not None:
#             st.write("Processed Output Preview:")
#             st.dataframe(output_df)

#             # Convert output to Excel for download
#             output_buffer = BytesIO()
#             output_df.to_excel(output_buffer, index=False)
#             output_buffer.seek(0)

#             st.download_button(
#                 label="Download Processed Output",
#                 data=output_buffer,
#                 file_name="processed_payouts.xlsx",
#                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#             )
#             logger.info("Output ready for download")
#     except Exception as e:
#         logger.error(f"Error reading or processing the uploaded file: {e}")
#         st.error("Error reading or processing the uploaded file. Please ensure it's a valid Excel file with a 'Sheet1' sheet.")

import streamlit as st
import pandas as pd
import re
import logging
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Streamlit app title
st.title("Reliance Payout Processing")

# Define FORMULA_DATA as provided
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
    {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "Rest of Companies", "PO": "90% of Payin", "REMARKS": "Zuno - 21"},
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

# Derive LOB_DICT from FORMULA_DATA for quick LOB lookup by segment
LOB_DICT = {rule['SEGMENT']: rule['LOB'] for rule in FORMULA_DATA}

# Function to calculate payout based on PO formula
def calculate_payout(payin_pct, po_formula):
    try:
        # Check "Less 2% of Payin" FIRST before general "% of Payin"
        if 'Less' in po_formula and '% of Payin' in po_formula:
            return payin_pct - 2
        elif '% of Payin' in po_formula:
            # Extract percentage and multiply directly with payin_pct
            percentage = float(re.search(r'(\d+)%', po_formula).group(1)) / 100
            return payin_pct * percentage
        elif po_formula.startswith('-') and '%' in po_formula:
            deduction = float(po_formula.strip('%'))
            return payin_pct + deduction  # Deduction is negative
        else:
            logger.warning(f"Unknown PO formula: {po_formula}, returning payin")
            return payin_pct
    except Exception as e:
        logger.error(f"Error calculating payout for payin {payin_pct}% and formula {po_formula}: {e}")
        return 0.0

# Function to find matching rule in FORMULA_DATA
def find_matching_rule(lob, segment, insurer, payin_pct):
    try:
        payin_pct = float(payin_pct)
        
        # Find all rules for this LOB and SEGMENT
        candidates = [rule for rule in FORMULA_DATA if rule['LOB'] == lob and rule['SEGMENT'] == segment]
        
        if not candidates:
            logger.warning(f"No rules found for LOB: {lob}, Segment: {segment}")
            return "0%", "No matching rule"
        
        # Filter candidates by insurer
        filtered_candidates = []
        for rule in candidates:
            insurers = [i.strip().upper() for i in rule['INSURER'].split(',')]
            insurer_upper = insurer.upper()
            
            if 'ALL COMPANIES' in insurers or insurer_upper in insurers:
                filtered_candidates.append(rule)
            elif 'REST OF COMPANIES' in insurers:
                # Get all specific insurers for this segment
                specific_insurers = set()
                for r in candidates:
                    r_insurers = [i.strip().upper() for i in r['INSURER'].split(',')]
                    if 'REST OF COMPANIES' not in r_insurers and 'ALL COMPANIES' not in r_insurers:
                        specific_insurers.update(r_insurers)
                # If insurer is not in specific list, use "Rest of Companies" rule
                if insurer_upper not in specific_insurers:
                    filtered_candidates.append(rule)
        
        if not filtered_candidates:
            logger.warning(f"No filtered rules for LOB: {lob}, Segment: {segment}, Insurer: {insurer}")
            return "0%", "No matching rule"
        
        # Find matching rule based on REMARKS (payin percentage conditions)
        matching_rule = None
        default_rule = None
        
        for cand in filtered_candidates:
            rem = cand['REMARKS']
            
            # Check payin-based conditions first
            if 'Below 20%' in rem and payin_pct < 20:
                matching_rule = cand
                break
            elif '21% to 30%' in rem and 21 <= payin_pct <= 30:
                matching_rule = cand
                break
            elif '31% to 50%' in rem and 31 <= payin_pct <= 50:
                matching_rule = cand
                break
            elif 'Above 50%' in rem and payin_pct > 50:
                matching_rule = cand
                break
            elif 'Above 20%' in rem and payin_pct > 20:
                matching_rule = cand
                break
            # If no payin condition, it's a default rule
            elif rem in ['NIL', 'All Fuel', 'Zuno - 21'] or not rem.startswith('Payin'):
                if default_rule is None:
                    default_rule = cand
        
        # Use matching rule if found, otherwise use default rule
        if matching_rule:
            return matching_rule['PO'], matching_rule['REMARKS']
        elif default_rule:
            return default_rule['PO'], default_rule['REMARKS']
        else:
            logger.warning(f"No matching rule for LOB: {lob}, Segment: {segment}, Insurer: {insurer}, Payin: {payin_pct}%")
            return "0%", "No matching rule"
            
    except Exception as e:
        logger.error(f"Error finding matching rule: {e}")
        return "0%", "Error in rule matching"

# Function to process the Excel file
def process_excel(file_buffer):
    try:
        # Read the sheet with multi-level headers (rows 1,2,3)
        df = pd.read_excel(file_buffer, sheet_name='Sheet1', header=[0, 1, 2])
        logger.info("Excel sheet 'Sheet1' read successfully")

        # Hardcoded company name
        company_name = 'Reliance'

        # Prepare output list
        output_data = []

        # Location columns (first two)
        location_cols = df.columns[:2]

        # Payout columns (from third onwards)
        payout_cols = df.columns[2:]

        # Process each data row
        for row_idx in range(len(df)):
            row = df.iloc[row_idx]
            zone = row[location_cols[0]]
            rto_region = row[location_cols[1]]
            location = f"{zone}: {rto_region}"

            for col in payout_cols:
                segment = col[0]  # Row 1: segment for FORMULA_DATA lookup
                sub_segment = col[1]  # Row 2: remark part
                policy_type = col[2]  # Row 3: remark part
                payin_value = row[col]

                if pd.isna(payin_value):
                    continue  # Skip NaN values

                payin_pct = float(payin_value) * 100  # Convert to percentage

                if payin_pct == 0:
                    payout_pct = 0.0
                    rule_remarks = "Payin is 0"
                else:
                    # Get LOB
                    lob = LOB_DICT.get(segment, "Unknown")

                    # Use segment directly
                    normalized_segment = segment

                    # Find matching rule
                    po_formula, rule_remarks = find_matching_rule(lob, normalized_segment, company_name, payin_pct)

                    # Calculate payout
                    payout_pct = calculate_payout(payin_pct, po_formula)

                # Combine remarks
                combined_remarks = f"{sub_segment}; {policy_type}; {rule_remarks}".strip('; ')

                # Append to output
                output_data.append({
                    "Company Name": company_name,
                    "Segment": segment,
                    "Sub Segment": sub_segment,
                    "Policy Type": policy_type,
                    "Location": location,
                    "Payin (%)": round(payin_pct, 2),
                    "Payout (%)": round(payout_pct, 2),
                    "Remarks": combined_remarks
                })

        # Create output DataFrame
        output_df = pd.DataFrame(output_data)
        logger.info("Payout processing completed successfully")
        return output_df
    except Exception as e:
        logger.error(f"Error processing Excel: {e}")
        st.error(f"Error processing Excel: {e}")
        return None

# Streamlit file uploader
uploaded_file = st.file_uploader("Upload your file here", type=["xlsx", "xls"])

if uploaded_file:
    try:
        # Read file into memory
        file_buffer = BytesIO(uploaded_file.read())
        logger.info("File uploaded and read successfully")
        st.write("File uploaded successfully!")

        # Display input DataFrame
        df = pd.read_excel(file_buffer, sheet_name='Sheet1', header=[0, 1, 2])
        st.write("Input Data Preview:")
        st.dataframe(df)

        # Reset buffer position
        file_buffer.seek(0)

        # Process the file
        output_df = process_excel(file_buffer)

        if output_df is not None:
            st.write("Processed Output Preview:")
            st.dataframe(output_df)

            # Convert output to Excel for download
            output_buffer = BytesIO()
            output_df.to_excel(output_buffer, index=False)
            output_buffer.seek(0)

            st.download_button(
                label="Download Processed Output",
                data=output_buffer,
                file_name="processed_payouts.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            logger.info("Output ready for download")
    except Exception as e:
        logger.error(f"Error reading or processing the uploaded file: {e}")
        st.error("Error reading or processing the uploaded file. Please ensure it's a valid Excel file with a 'Sheet1' sheet.")


#         """
        
        
#         Let me dry run the code for the failing cases. I'll trace through each step:

# ## **Case 1: School Bus with 72.5% payin**

# **Step 1: Input**
# - Segment: "SCHOOL BUS"
# - Payin: 72.5%
# - Insurer: "Reliance"

# **Step 2: `find_matching_rule("BUS", "SCHOOL BUS", "Reliance", 72.5)`**

# Find candidates where LOB="BUS" and SEGMENT="SCHOOL BUS":
# ```python
# candidates = [
#     {"LOB": "BUS", "SEGMENT": "SCHOOL BUS", "INSURER": "TATA, Reliance, Digit, ICICI", "PO": "Less 2% of Payin", "REMARKS": "NIL"},
#     {"LOB": "BUS", "SEGMENT": "SCHOOL BUS", "INSURER": "Rest of Companies", "PO": "88% of Payin", "REMARKS": "NIL"}
# ]
# ```

# **Step 3: Filter by insurer "Reliance"**
# - First rule: insurers = ["TATA", "Reliance", "Digit", "ICICI"]
# - "Reliance" is IN the list ✓
# - Add to filtered_candidates

# Result: `filtered_candidates = [{"INSURER": "TATA, Reliance, Digit, ICICI", "PO": "Less 2% of Payin", "REMARKS": "NIL"}]`

# **Step 4: Match by REMARKS**
# - rem = "NIL"
# - Check: `rem == 'NIL' or not rem.startswith('Payin')` → **TRUE**
# - `matching_rule` = first rule with "Less 2% of Payin"

# **Step 5: `calculate_payout(72.5, "Less 2% of Payin")`**

# **🔴 PROBLEM FOUND HERE:**
# ```python
# if '% of Payin' in po_formula:  # "Less 2% of Payin" contains "% of Payin"!
#     percentage = float(re.search(r'(\d+)%', po_formula).group(1)) / 100
#     # Extracts "2" from the formula
#     # percentage = 0.02
#     return (payin * percentage) * 100
#     # = (0.725 * 0.02) * 100 = 1.45
# ```

# **The bug**: The condition `'% of Payin' in po_formula` matches **both** "88% of Payin" **AND** "Less 2% of Payin"!

# ---

# ## **Case 2: GCV < 2.5 with 70% payin**

# **Step 1: Input**
# - Segment: "Upto 2.5 GVW"
# - Payin: 70%
# - Insurer: "Reliance"

# **Step 2: Find rule**
# ```python
# {"LOB": "CV", "SEGMENT": "Upto 2.5 GVW", "INSURER": "Reliance, SBI, Tata", "PO": "-2%", "REMARKS": "NIL"}
# ```

# **Step 3: `calculate_payout(70, "-2%")`**
# ```python
# elif po_formula.startswith('-') and '%' in po_formula:
#     deduction = float(po_formula.strip('%'))  # deduction = -2
#     return payin_pct + deduction  # 70 + (-2) = 68 ✓
# ```

# This one is **CORRECT**.

# ---

# ## **The Fix**

# The problem is in `calculate_payout`. The conditions need to be reordered:**The Issue:** The `'% of Payin'` check was matching "Less 2% of Payin" before the specific "Less" check could run, because of the order of if-elif conditions.

# **The Fix:** Check for "Less" **FIRST** before checking the general "% of Payin" pattern.

# **Now the results will be:**
# - School Bus (72.5%): 72.5 - 2 = **70.5%** ✓
# - GCV (70%): 70 - 2 = **68%** ✓
# - Tractor (25%): 25 × 0.88 = **22%** ✓
# - MISD (40%): 40 × 0.88 = **35.2%** ✓

# Try it now!
        
#         """
