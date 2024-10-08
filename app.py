import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# from collections import OrderedDict
# from streamlit_chromadb_connection.chromadb_connection import ChromaDBConnection  # Importa ChromaDBConnection
from langchain_chroma import Chroma

from dotenv import load_dotenv
import os
load_dotenv()

st.title("AUTISMO PARA PADRES - LIBRO CONSULTA")

import requests
from io import BytesIO

# PDF DATA LOADING
pdf_url = "https://github.com/davidcarrillo10288/Q-A_Autism/raw/master/05%20Autismo%20Manual%20Avanzado%20Padres.pdf"
response = requests.get(pdf_url)
pdf_file = BytesIO(response.content)
loader = PyPDFLoader(pdf_file)
data = loader.load()

# TEXT SPLITTING
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(data)

# TEXT EMBEDDING
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore = Chroma.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

retreiver = vectorstore.as_retriever(search_type="similarity", search_kwargs = {"k":10})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_output_tokens=500)

query = st.chat_input("Preguntame sobre Autismo: ")
prompt = query

system_prompt = (
    "Eres un asistente para actividades de preguntas y respuestas."
    "Usa las siguientes piezas de retreived context para responder"
    "La pregunta. Si no sabes la respuesta, tienes que decir que tú "
    "no sabes la respuesta. Usa cinco oraciones como máximo y manten"
    "las respuestas concisas"
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human"), "{input}"
    ]
)

if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retreiver, question_answer_chain)

    response = rag_chain.invoke({"input":query})
    # print(response["answer"])

    st.write(response["answer"])

