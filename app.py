import streamlit as st
import PyPDF2
import os
import openai
from io import BytesIO

# Use OpenAI's v1 SDK
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="KITKASH Invoice Extractor", layout="centered")

# Display logo and title
st.image("kitkash_logo.png", width=200)
st.title("KITKASH Invoice Extractor")
st.caption("Upload an invoice or insurance policy PDF to extract finance data.")

uploaded_file = st.file_uploader("📎 Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("🔍 Extracting text from PDF..."):
        reader = PyPDF2.PdfReader(uploaded_file)
        extracted_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"

    st.success("✅ Text extracted from PDF!")

    with st.expander("📄 View Extracted Text"):
        st.text_area("Raw Text", extracted_text, height=300)

    if st.button("🚀 Extract Invoice Data"):
        with st.spinner("🧠 Asking fine-tuned GPT model..."):
            prompt = f"""
You are a data extraction bot trained to pull specific values from insurance-related invoices and policies.

Extract the following fields from the raw PDF text below:
- Insurance Company Name
- General Agent
- Broker
- Policy Number
- Coverage Type
- Pure Premium
- Minimum Earned Premium %
- Cancellation Terms in Days
- Effective Date
- Expiration Date (if not listed, infer 12 months after effective date and label as 'inferred')
- Policy Fees
- Taxes
- Broker Fee
- Inspection Fee
- Payment Mailing Address
- Electronic Payment Link
- ACH/Wire Instructions

Respond only with structured data in JSON format.

RAW TEXT:
{extracted_text}
"""

            response = client.chat.completions.create(
                model="ft:gpt-3.5-turbo-0125:kash:kash-final:BOtVnn7m",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            extracted_json = response.choices[0].message.content.strip()
            st.success("✅ Data extracted successfully!")
            st.code(extracted_json, language="json")
