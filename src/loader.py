import tempfile

from langchain_community.document_loaders import PyPDFLoader


# =====================================================
# Load PDF uploaded through Streamlit
# =====================================================
def load_uploaded_pdf(uploaded_file):

    # Create temporary file
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(
            uploaded_file.getvalue()
        )

        temp_pdf_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(
        temp_pdf_path
    )

    docs = loader.load()

    return docs