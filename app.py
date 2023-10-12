import streamlit as st
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from dotenv import load_dotenv
load_dotenv()

st.title("Andy's Chatbot")
prompt = st.text_input("Ask me anything:")

loader = TextLoader("about_me.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())

if st.button('Generate'):
    if prompt:
        # print("check")
        with st.spinner('Generating response...'):
            qa.run(prompt)
    else:
        st.warning('Please enter your prompt')



