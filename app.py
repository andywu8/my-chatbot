import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
import os 
from langchain.prompts import PromptTemplate
import constants
from dotenv import load_dotenv
from langchain.chains import LLMChain, SequentialChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
load_dotenv()

os.environ['OPENAI_API_KEY'] = constants.OPENAI_API_KEY

st.title("Andy's Chatbot")
question = st.text_input("Ask me anything:")

file_path = "about_me.txt"
loader = TextLoader(file_path)
docs = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)

prompt_template = """
Use the following pieces of context to answer the question at the end.
{context}
Question: {question}
Answer:"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

llm = OpenAI()
embeddings = OpenAIEmbeddings()
texts = text_splitter.split_documents(docs)
db = Chroma.from_documents(texts, embeddings)

# question = "What is your name?"
similar_doc = db.similarity_search(question, k=1)
context = similar_doc[0].page_content
query_llm = LLMChain(llm=llm, prompt=prompt)

# loader = TextLoader("about_me.txt")
# index = VectorstoreIndexCreator().from_loaders([loader])

# print("response", response)
# 
# print(index.query("Tell me about your school"))

# response = index.query(prompt)
if st.button('Generate'):
    if question:
        with st.spinner('Generating response...'):
            response = query_llm.run({"context": context, "question": question})
            st.write(response)
    else:
        st.warning('Please enter your prompt')



