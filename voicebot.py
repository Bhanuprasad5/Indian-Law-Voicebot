import io
import os
import tempfile

import pygame
import speech_recognition as sr
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from gtts import gTTS
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    MessagesPlaceholder)
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


def initialize_chat_model():
    GROQ_API_KEY = "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"

    chat_model = ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-8b-8192")
    return chat_model

chat_model = initialize_chat_model()

# Create Chat Template
chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=""" You are Law Buddy, a helpful and trustworthy AI legal assistant trained to assist users by summarizing and explaining legal content only based on the retrieved documents. You are not a lawyer and must never provide legal advice or opinions.
Your responsibilities:
- Respond only with information supported by the retrieved documents.
- If the information is not in the documents, clearly say "I could not find relevant legal information in the available documents."
- Avoid making assumptions or generating laws on your own.
- Use simple, clear, and professional language.
- Always be neutral and respectful.
- End every answer with: "This is not legal advice. Please consult a qualified legal professional for specific legal matters."
You are designed to help users understand legal documents, definitions, case summaries, procedures, and rules in a human-readable way without giving advice or opinions.
"""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}")
    ]
)

# Initialize the Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create an Output Parser
output_parser = StrOutputParser()

# Define a chain
chain = RunnablePassthrough.assign(
            chat_history=RunnableLambda(lambda human_input: memory.load_memory_variables(human_input)['chat_history'])
        ) | chat_prompt_template | chat_model | output_parser

# Define embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embeddings_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)
CHROMA_DB_DIR = "./chroma_db_"  # Directory for ChromaDB
# Initialize Chroma Database
db = Chroma(collection_name="vector_database",
            embedding_function=embeddings_model,
            persist_directory=CHROMA_DB_DIR)

def get_bhanu_response(input_query, db, chain, memory):
    """
    Function to take an input query, perform similarity search in embeddings,
    retrieve context, send to LLM, and return the response as Chouki Bhanu Prasad.
    
    Args:
        input_query (str): The question or input from the user.
        db: Chroma DB instance with embeddings.
        chain: LLM chain object for invoking the model.
        memory: Memory object to save context and response.
    
    Returns:
        str: The LLM response as Chouki Bhanu Prasad.
    """
    # Perform similarity search in embeddings
    docs_chroma = db.similarity_search_with_score(input_query, k=1)
    
    # Extract context from search results
    context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])
    
    # Construct the query for the LLM
    query = {
        "human_input": f"""Context: {context_text}, Query: {input_query}. Respond with a concise and impactful answer, limiting unnecessary details while retaining clarity and relevance in short."""

    }
    
    # Invoke the LLM chain to get the response
    response = chain.invoke(query)
    
    # Save the query and response to memory
    memory.save_context(query, {"output": response})
    
    # Return the response
    return response



# Initialize recognizer


def text_to_speech(Text):
    language = "en"
    text = Text
    speech = gTTS(text=Text, lang=language, slow=False)
    speech_fp = io.BytesIO()
    speech.write_to_fp(speech_fp)
    speech_fp.seek(0)
    st.audio(speech_fp, format="audio/mp3")

# Streamlit UI
st.title("Hii I am Indian Law Buddy")


# Record audio
audio_bytes = audio_recorder(
text="Click to record",
recording_color="#e8b62c",
neutral_color="#6aa36f",
icon_name="microphone",
icon_size="2x",
)
if audio_bytes:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name    
    
        # Convert speech to text
    recognizer = sr.Recognizer()
    with sr.AudioFile(temp_audio_path) as source:
        audio_data = recognizer.record(source)
        
        try:
            text = recognizer.recognize_google(audio_data)
            st.success("📝 Transcribed Text:")
            st.write(text)
        except sr.UnknownValueError:
            st.error("Could not understand the audio.")
        except sr.RequestError:
            st.error("Could not request results, please check your internet connection.")

    
    response = get_bhanu_response(text, db, chain, memory)
    st.write(response)

    text_to_speech(response)