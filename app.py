import streamlit as st
import PyPDF2
import openai
import os

# Set the page config
st.set_page_config(
    page_title="KITKASH Invoice Extractor",
    page_icon="📄",
    layout="centered",
)

# App title
st.title("📄 KITKASH Invoice Extractor (Fine-tuned GPT)")

# File uploader
uploaded_file = st.file_uploader("Upload an insurance PDF", type="pdf")

if uploaded_file is not None:
    st.success(f"Uploaded file: {uploaded_file.name}")
    pdf_reader = PyPDF2.PdfReader(uploaded_file)

    # Extract raw text
    raw_text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            raw_text += extracted + "\n"

    # Show raw text
    with st.expander("📝 Raw Extracted Text", expanded=True):
        st.text_area("Text from PDF", raw_text, height=400)

    # Button to run fine-tuned model
    if st.button("🔍 Extract Info Using Fine-tuned Model"):
        with st.spinner("Extracting data using fine-tuned GPT model..."):
            try:
                openai.api_key = st.secrets["OPENAI_API_KEY"]

                response = openai.chat.completions.create(
                    model="ft:gpt-3.5-turbo-0125:kash:kash-final:BOtVnn7m",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a data extraction bot that pulls structured values from invoice and insurance documents."
                        },
                        {
                            "role": "user",
                            "content": raw_text
                        }
                    ],
                    temperature=0.2
                )

                extracted_data = response.choices[0].message.content
                st.success("✅ Extraction Complete")
                st.subheader("📊 Extracted Data")
                st.code(extracted_data, language="json")

            except Exception as e:
                st.error(f"⚠️ Error extracting data: {e}")
else:
    st.info("Please upload a PDF to begin.")
