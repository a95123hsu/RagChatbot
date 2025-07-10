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
import re

def convert_units(text):
    # Convert GPM to LPM
    text = re.sub(r"(\d+(?:\.\d+)?)\s*gpm", lambda m: f"{round(float(m.group(1)) * 3.785)} LPM", text, flags=re.I)
    
    # Convert ft to meters
    text = re.sub(r"(\d+(?:\.\d+)?)\s*ft", lambda m: f"{round(float(m.group(1)) * 0.3048, 2)} meters", text, flags=re.I)
    
    return text


# Load API keys
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

st.set_page_config(page_title="Pump Chatbot", layout="centered")
col1, col2 = st.columns([1, 4])

with col1:
    st.image("https://www.hungpump.com/images/340357", width=200)  # or any size that fits well

with col2:
    st.title("Pump Selection Chatbot")


if st.button("Reset Chat"):
    st.session_state.messages = [{"role": "assistant", "content": "👋 Ask me anything about our pumps!"}]
    st.rerun()

# Load pump data from Supabase
@st.cache_data
def load_data():
    response = supabase.table("pump_data").select("model_no, description").execute()
    return pd.DataFrame(response.data)

# Turn rows into text
@st.cache_data
def prepare_documents(df):
    return df.apply(lambda row: (
        f"Model: {row['model_no']}, "
        f"Rated Flow: {row.get('flow', '')} LPM, "
        f"Rated Head: {row.get('head', '')} meters, "
        f"Power: {row.get('power', '')} kW, "
        f"Description: {row.get('description', '')}"
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
    if not isinstance(query, str) or query.strip() == "":
        return []

    query_vec = embeddings.embed_query(query)
    D, I = index.search(np.array([query_vec]), k)
    return [documents[i] for i in I[0]]


def generate_answer(query, context, history, max_turns=3):
    chat_history = []

    # Get last `max_turns` user-assistant message pairs
    recent_turns = [msg for msg in history if msg["role"] in ["user", "assistant"]][-2 * max_turns:]

    # Preserve assistant and user exchange
    chat_history.extend(recent_turns)

    # Detect Traditional Chinese for language control
    if re.search(r'[\u4e00-\u9fff]', query):
        chat_history.insert(0, {"role": "system", "content": "請用繁體中文回答。"})

    # Add the current query with RAG context
    chat_history.append({
        "role": "user",
        "content": (
            "Use the following pump data to answer the user's question.\n"
            "- Filtered user's input based on rated head and rated flow.\n"
            "- Clearly mention: model number, frequency, rated flow (in LPM), rated head (in meters), power (kW), max head, max flow, and more details if available.\n"
            "- Convert units in the question to LPM/meters if needed.\n"
            "- Organize the response clearly with bullet points or numbered list.\n\n"
            "- If multiple pumps are relevant, list up to 10 of them.\n"
            f"Pump Data:\n{context}\n\n"
            f"User Question: {query}"
        )
    })

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=chat_history,
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
    prompt = convert_units(prompt)  # ⬅️ convert units
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant thinking...
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Searching pump database..."):
            context_list = retrieve_similar(prompt, index, documents, k=10)

            if not context_list:
                full_response = "❌ I couldn't find any relevant pumps. Try asking differently."
            else:
                context = "\n".join(context_list)
                full_response = generate_answer(prompt, context, st.session_state.messages)

        # Simulate typing
        displayed = ""
        for word in full_response.split():
            displayed += word + " "
            time.sleep(0.02)
            message_placeholder.markdown(displayed + "▌")
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

