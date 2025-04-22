import streamlit as st
import openai
import PyPDF2
import os

# Load OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# App title
st.set_page_config(page_title="KASH Premium Finance Extractor")
st.title("📄 KASH Premium Finance Extractor")

# Upload PDF
uploaded_file = st.file_uploader("Upload an insurance PDF", type="pdf")

if uploaded_file is not None:
    # Read PDF
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    raw_text = "\n".join([page.extract_text() for page in pdf_reader if page.extract_text()])

    # Display extracted text
    st.subheader("Extracted Text")
    st.caption("Raw PDF Text")
    st.code(raw_text)

    # Button to extract using fine-tuned model
    if st.button("🔍 Extract Data from Fine-Tuned Model"):
        with st.spinner("Contacting fine-tuned model..."):
            try:
                # Use v1 API call
                client = openai.OpenAI()
                response = client.chat.completions.create(
                    model="ft:gpt-3.5-turbo-0125:kash:kash-final:BOtVnn7m",
                    messages=[
                        {"role": "system", "content": "You extract structured finance data from insurance invoices and policies."},
                        {"role": "user", "content": raw_text}
                    ]
                )

                output = response.choices[0].message.content
                st.subheader("📋 Extracted Data")
                st.text(output)

            except Exception as e:
                st.error(f"❌ Error: {e}")

