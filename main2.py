#!pip install setuptools
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler
)
import asyncio
from deep_translator import GoogleTranslator

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import Chroma



from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import os


@st.cache_resource
def create_chain():
    # Callback Manager For Streaming Handler
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    # Loading Model
    llm = LlamaCpp(
        # model_path="llama-2-7b-chat.ggmlv3.q2_K.bin",
        model_path="C:/Users/Alaa AI/Python Projects/Ai Models/alaa_ai_model_llama2_K_M_v1.4.gguf",
        n_ctx=1000,
        n_gpu_layers=512,
        n_batch=30,
        callback_manager=callback_manager,
        temperature = 0.1,
        max_tokens = 30000,
        n_parts=1,
    )
    # embedding engine
    hf_embedding = HuggingFaceInstructEmbeddings()
    filePath = "C:/Users/Alaa AI/Python Projects/Projects/Streamlit chat app/Chat with PDF - Chroma/chroma_knowledge_base/"
    if(os.path.isdir(filePath)):
        st.markdown("Knowledge Base is Loading ...")
        # load from local
        db = Chroma(persist_directory = "chroma_knowledge_base", embedding_function=hf_embedding)
    else:
        # Loading Our New PDFs
        st.markdown("Loading new Knowledge Base ...")
        knowledge_base_loader = PyPDFDirectoryLoader("pdfs")
        knowledge_base = knowledge_base_loader.load()
        knowledge_base_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        knowledge_base_texts = knowledge_base_text_splitter.split_documents(knowledge_base)
        st.markdown("We've loaded "+str(len(knowledge_base_texts)) + " Pages ")
        st.markdown("Knowledge Base is Now Under Creation Progress ...")
        db = Chroma.from_documents(knowledge_base_texts, hf_embedding , persist_directory = "chroma_knowledge_base")

    # Prompt Template
    template = """Question: {question}

    Answer:"""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    # Creating LLM Chain
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    return db , llm_chain , prompt

# Set the webpage title
st.set_page_config(
    page_title="Alaa's Chat Robot With PDFs!"
)


# Create a header element
st.header("Alaa's Chat Robot With PDFs!")
# Create Select Box
lang_opts = ["ar", "en" , "fr" , "zh-CN"]
lang_selected = st.selectbox("Select Target Language " , options = lang_opts)
db , chain , prompt = create_chain()
# We store the conversation in the session state.
# This will be used to render the chat conversation.
# We initialize it with the first message we want to be greeted with.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I help you today?"}
    ]

if "current_response" not in st.session_state:
    st.session_state.current_response = ""


# We loop through each message in the session state and render it as
# a chat message.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
async def get_response(llm_chain , db , user_prompt , prompt , container):
    full_response = ""
    search = db.similarity_search(user_prompt, k=2)
    template = '''Context: {context}

    Based on Context provide me answer for following question
    Question: {question}

    Your name is Alaa Sayed. You are an expert in Egyptian Law and helpful experienced Lawyer who answers questions in Details and organized manner. You have all access to solve cases and provide intelligence solutions. Tell me the information about the fact. The answer should be from context
    use general knowledge to answer the query'''
    prompt = PromptTemplate(input_variables=["context", "question"], template= template)
    final_prompt = prompt.format(question=user_prompt, context=search)
    
    # Add the response to the chat window
    with container.chat_message("assistant"):
        full_response = llm_chain.run(final_prompt)
        container.markdown(full_response)
    full_response = GoogleTranslator(source='auto', target=lang_selected).translate(full_response)
    container.markdown(full_response)
    # Add the response to the session state
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
def disable():
    st.session_state.disabled = True
    
if "disabled" not in st.session_state:
    st.session_state.disabled = False
    
if user_prompt := st.chat_input("Your message here", key="user_input" , on_submit = disable , disabled=st.session_state.disabled):
    
    del st.session_state.disabled
    if "disabled" not in st.session_state:
        st.session_state.disabled = False
    #st.chat_input("Your message here", key="disabled_chat_input", disabled=True)
    st.markdown("in session")
    # Add our input to the session state
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )
    # Add our input to the chat window
    with st.chat_message("user"):
        st.markdown(user_prompt)

        
    user_prompt = GoogleTranslator(source='auto', target='en').translate(user_prompt)
    asyncio.run(get_response(chain , db , user_prompt  , prompt , st.empty()))

        
    st.rerun()       
