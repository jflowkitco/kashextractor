import streamlit as st
import openai
import PyPDF2
from io import BytesIO

st.set_page_config(page_title="KASH Invoice Extractor (Fine-tuned GPT)", layout="centered")

st.title("📄 KASH Invoice Extractor (Fine-tuned GPT)")
st.caption("Upload an invoice PDF")

uploaded_file = st.file_uploader("Upload an insurance PDF", type=["pdf"])

if uploaded_file:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    raw_text = "\n".join(
        [page.extract_text() for page in pdf_reader.pages if page and page.extract_text()]
    )

    st.subheader("📄 Raw Extracted Text")
    with st.expander("Text from PDF", expanded=False):
        st.code(raw_text)

    if st.button("🔍 Extract Info Using Fine-tuned Model"):
        with st.spinner("Calling fine-tuned GPT model..."):
            openai.api_key = st.secrets["OPENAI_API_KEY"]
            try:
                response = openai.chat.completions.create(
                    model="ft:gpt-3.5-turbo-0125:kash:kash-final:BOtVnn7m",
                    messages=[
                        {
                            "role": "system",
                            "content": "Extract the following fields from the invoice text: "
                                       "Insurance Company Name, General Agent, Broker, Policy Number, "
                                       "Coverage Type, Pure Premium, Minimum Earned Premium %, "
                                       "Cancellation Terms in Days, Effective Date, Expiration Date, "
                                       "Policy Fees, Taxes, Broker Fee, Inspection Fee, "
                                       "Payment Mail Address, Electronic Payment Link, ACH/Wire Instructions."
                        },
                        {"role": "user", "content": raw_text}
                    ],
                    temperature=0.3
                )
                extracted = response.choices[0].message.content
                st.subheader("📊 Extracted Data")
                st.code(extracted)
            except Exception as e:
                st.error(f"❌ Failed to extract: {e}")
