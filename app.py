import streamlit as st
import openai
import PyPDF2
import tempfile

st.set_page_config(page_title="KASH Extractor", layout="centered")
st.title("📄 KASH Premium Finance Extractor")

openai.api_key = st.secrets["OPENAI_API_KEY"]

uploaded_file = st.file_uploader("Upload an insurance PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with open(tmp_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        raw_text = "\n".join([page.extract_text() for page in pdf_reader if page.extract_text()])

    st.subheader("Extracted Text")
    st.text_area("Raw PDF Text", raw_text[:3000] + ("..." if len(raw_text) > 3000 else ""), height=200)

    if st.button("🔍 Extract Data from Fine-Tuned Model"):
        with st.spinner("Calling model..."):
            system_prompt = """
            You are an assistant trained to extract the following fields from an insurance-related PDF:
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
            Format your answer in JSON.
            """

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_text}
            ]

            try:
                response = openai.ChatCompletion.create(
                    model="ft:gpt-3.5-turbo-0125:kash:kash-final:BOtVnn7m",
                    messages=messages,
                    temperature=0
                )
                answer = response.choices[0].message["content"]
                st.subheader("🧠 Extracted Data")
                st.code(answer, language="json")
            except Exception as e:
                st.error(f"Error: {e}")
