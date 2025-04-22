import streamlit as st
import PyPDF2
import openai
import os

# Set page config
st.set_page_config(
    page_title="KASH Invoice Extractor (Fine-tuned GPT)",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("📄 KASH Invoice Extractor (Fine-tuned GPT)")

# Upload section
uploaded_file = st.file_uploader("Upload an insurance PDF", type="pdf")

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")

    # Extract text
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    raw_text = "\n".join([page.extract_text() for page in pdf_reader if page.extract_text()])

    st.markdown("### 📄 Raw Extracted Text")
    with st.expander("Text from PDF", expanded=False):
        st.code(raw_text, language='text')

    if st.button("🔍 Extract Info Using Fine-tuned Model"):
        with st.spinner("Talking to the fine-tuned model..."):
            openai.api_key = st.secrets["OPENAI_API_KEY"]

            response = openai.chat.completions.create(
                model="ft:gpt-3.5-turbo-0125:kash:kash-final:BOtVnn7m",
                messages=[
                    {"role": "system", "content": "You are a data extraction tool for insurance invoices. Extract the relevant fields in JSON format."},
                    {"role": "user", "content": raw_text}
                ],
                temperature=0.0
            )

            extracted_data = response.choices[0].message.content

        st.markdown("### 📊 Extracted Data")
        st.code(extracted_data, language='json')

        st.success("Extraction complete ✅")
else:
    st.info("Please upload a PDF to begin.")
