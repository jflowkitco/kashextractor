import streamlit as st
import openai
import PyPDF2
import tempfile

# Load OpenAI key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Title and file uploader
st.title("📄 KASH Invoice Extractor (Fine-tuned GPT)")
uploaded_file = st.file_uploader("Upload an invoice PDF", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Read PDF contents
    with open(tmp_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        raw_text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])

    st.subheader("📑 Raw Extracted Text")
    st.text_area("Text from PDF", raw_text, height=250)

    if st.button("🧠 Extract Info Using Fine-tuned Model"):
        with st.spinner("Extracting data..."):
            system_prompt = """
You are a data extraction bot trained on insurance invoices. 
Extract the following fields in JSON format:
- Insurance Company Name
- General Agent
- Broker
- Policy Number
- Coverage Type
- Pure Premium
- Minimum Earned Premium %
- Cancellation Terms in Days
- Effective Date
- Expiration Date
- Policy Fees
- Taxes
- Broker Fee
- Inspection Fee
- Payment Mail Address
- Electronic Payment Link
- ACH/Wire Instructions

Return only the JSON object, no commentary.
"""

            response = openai.chat.completions.create(
                model="ft:gpt-3.5-turbo-0125:kash:kash-final:BOtVnn7m",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": raw_text}
                ],
                temperature=0.3,
            )

            result = response.choices[0].message.content.strip()
            st.subheader("📊 Extracted Data")
            st.code(result, language="json")
