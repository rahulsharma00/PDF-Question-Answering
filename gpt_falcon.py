from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub, OpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os 

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your pdf")
    st.header("Ask your pdf 🤓")

    # model selection
    selected_model = st.selectbox("Select Language Model", ["OpenAI", "Falcon-7B"])
    
    pdf = st.file_uploader("Upload your pdf", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator="\n",  # Defines a new line 
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Initialize embeddings to None
        embeddings = None

        if selected_model == 'OpenAI':
            embeddings = OpenAIEmbeddings()
            llm = OpenAI()
        elif selected_model == 'Falcon-7B':  # Corrected model name
            embeddings = HuggingFaceEmbeddings()
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
            llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.1, "max_length": 512})
        else:
            st.error('Invalid model selection')

        # Check if embeddings is not None before using it
        if embeddings is not None:
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            # show user input
            user_question = st.text_input("Ask a question about the PDF: ")
            if user_question:
                docs = knowledge_base.similarity_search(user_question)

                # Remove reinitialization of llm
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=user_question)

                st.write(response)

if __name__ == '__main__':
    main()
