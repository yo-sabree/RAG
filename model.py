from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

st.set_page_config(page_title="RAG APP")
st.header("Question your PDF")

with st.sidebar:
    st.subheader("Upload your documents here.")
    pdfs = st.file_uploader("Click to upload", accept_multiple_files=True)
    if st.button("Submit"):
        if pdfs:
            with st.spinner("Processing"):
                texts = ""
                for pdf in pdfs:
                    read_pdf = PdfReader(pdf)
                    for page in read_pdf.pages:
                        texts += page.extract_text()

                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
                text_chunks = text_splitter.split_text(texts)
                db = Chroma.from_texts(text_chunks, embeddings)
                st.success("PDFs processed and stored successfully in the database. You can now enter your query.")
                st.session_state['db'] = db
        else:
            st.write("Please upload at least one PDF document.")

if 'db' in st.session_state:
    inp = st.text_input("Enter your question")
    if inp:
        dbs = st.session_state['db'].as_retriever(search_kwargs={"k": 1})
        response = dbs.invoke(inp)
        if response:
            st.write("Answer based on the most relevant document:")
            relevant_text = response[0].page_content
            st.write(relevant_text[:500] + "...")
        else:
            st.write("No relevant answer found.")
    else:
        st.write("Enter a question")
