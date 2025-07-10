import os
import pandas as pd
import numpy as np
import faiss
import streamlit as st
import time
from dotenv import load_dotenv
from supabase import create_client
from langchain_community.embeddings import OpenAIEmbeddings
from openai import OpenAI, OpenAIError

# Load API keys
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

st.set_page_config(page_title="🔧 Pump Chatbot", layout="centered")
st.title("🔧 Pump Selection Chatbot")

# Load pump data from Supabase
@st.cache_data
def load_data():
    response = supabase.table("pump_data").select("model_no, description").execute()
    return pd.DataFrame(response.data)

# Turn rows into text
@st.cache_data
def prepare_documents(df):
    return df.apply(lambda row: (
        f"Model: {row['model_no']}, Flow: {row.get('flow', '')} LPM, Head: {row.get('head', '')} m, "
        f"Power: {row.get('power', '')} kW, Description: {row.get('description', '')}"
    ), axis=1).tolist()

# Build vector index
@st.cache_resource
def build_index(documents):
    vectors = embeddings.embed_documents(documents)
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors))
    return index

# Retrieval function
def retrieve_similar(query, index, documents, k=5):
    query_vec = embeddings.embed_query(query)
    D, I = index.search(np.array([query_vec]), k)
    return [documents[i] for i in I[0]]

# GPT completion
def generate_answer(query, context):
    prompt = (
        f"You are a pump selection assistant. Use the data below to answer the question.\n\n"
        f"{context}\n\n"
        f"Question: {query}\nAnswer:"
    )
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except OpenAIError as e:
        return f"❌ OpenAI error: {e}"

# Load and build
df = load_data()
documents = prepare_documents(df)
index = build_index(documents)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "👋 Ask me anything about our pumps!"}]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input UI
if prompt := st.chat_input("Ask about flow, head, power, or a model number..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant thinking...
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Searching pump database..."):
            context = "\n".join(retrieve_similar(prompt, index, documents))
            if not context:
                full_response = "❌ I couldn't find any relevant pumps. Try asking differently."
            else:
                full_response = generate_answer(prompt, context)

        # Simulate typing
        displayed = ""
        for word in full_response.split():
            displayed += word + " "
            time.sleep(0.02)
            message_placeholder.markdown(displayed + "▌")
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
