import os 
import streamlit as st
import langchain
from typing_extensions import Protocol
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain, SequentialChain

from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
# print(langchain.__version__)


from dotenv import load_dotenv
load_dotenv()

st.title("LangChain Andy-Bot App")
prompt = st.text_input("Ask me a question:")

# Prompt Templates
answer_template = PromptTemplate(
    input_variables = ['question'],
    template='''Answer this question {question}
    pretending to be me, Andy, using this information about me: 
    I went to Yale University for college where I studied Computer Science and Economics and Statistics and Data Science. 
    I went to Lowell High School in San Francisco. I have a lot of asian friends and I like kpop, anime, and manga.
    I love movies and can name a lot of movie trivia. 
    I use lmao for when things are funny, wtf when I'm surprised, and smh when I'm disappointed.
    I like rabbits and I have a rabbit named Messi. I like to make fun of people and think I'm funny.
    I'm a pretty boring person. I use short responses.
    My girlfriend's name is Kiki Suarez who I love a lot! She's really cute and I love her. I like to tease her about her being fat.
    '''
)


# Memory 
answer_memory = ConversationBufferMemory(input_key='question', memory_key='answer_memory')


llm = OpenAI(temperature=.9)
answer_chain = LLMChain(llm=llm, prompt=answer_template, 
verbose=True, output_key='title', memory=answer_memory)


# print("script chain .prompt", script_chain.prompt)

if st.button('Generate'):
    if prompt:
        with st.spinner('Generating response...'):
            answer = answer_chain.run(question=prompt)
            st.write(answer)
            # st.write(script)
        with st.expander('Answer History'):
            st.info(answer_memory.buffer)
    else:
        st.warning('Please enter your prompt')



