import  streamlit as st
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

st.header("Lendeavor's AI Chatbot!")

with st.sidebar:
    # To upload csv file through streamlit UI
    file = st.file_uploader("upload your scv file here!", type="csv")

if file is not None:
    # Read the csv file
    df = pd.read_csv(file)
    # st.write(df) # to show it  on UI

    # convert contents it into string for later embedding
    bot_training_content = df.to_string(index=False)

    # text splitting and chunking
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", "", " ", ",", "."],
        chunk_size = 1000,
        chunk_overlap = 200
    )

    chunks = text_splitter.split_text(bot_training_content)

    # st.write(chunks)

    # Embeddings
    embedding = OllamaEmbeddings(
        model = "mxbai-embed-large"
    )

    # vector store the embeddings
    vector_store = Chroma.from_texts(
        chunks, embedding
    )

    # get user input
    user_question = st.text_input("Enter your question here...")

    # will do the vector search and ranking
    retriever = vector_store.as_retriever(
        search_type = "mmr",
        search_kwargs = {"k":3}
    )

    # llm creation
    llm = ChatOllama(
        model = "llama3.2",
        num_predict = 300,
        temperature = 0.3
    )

    prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an AI assistant for Lendeavor.

            Answer the user's questions using ONLY the information provided in the CSV file.

            Rules:
            1. Do not guess or make up information.
            2. If the answer is not available in the provided context, say:
            "I don't have knowledge on that based on the provided information."
            3. If the requested information is not present in the CSV file, you may also say:
            "Please visit lndvr.com or contact support at info@lendeavor.com."
            4. Always provide answers based on facts present in the provided context.

            Context:{context}
            """
            ),
            ("human","{question}")
        ])

    # chain 
    chain = (
        {"context":retriever, "question":RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    if user_question:
        response = chain.invoke(user_question)
        st.write(response)
