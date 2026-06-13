import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# ─────────────────────────────────
# PAGE SETTINGS
# ─────────────────────────────────
st.set_page_config(
    page_title="BuildMind AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 BuildMind AI Assistant")
st.caption("Ask me anything about our services, pricing, and process.")

# ─────────────────────────────────
# LOAD RAG SYSTEM (only once)
# ─────────────────────────────────
@st.cache_resource
def load_rag():
    loader = TextLoader("buildmind_company_data.txt")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(
        model="openai/gpt-oss-20b:free",
        api_key=st.secrets["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1"
    )

    return retriever, llm

# Load retriever and LLM
with st.spinner("Loading BuildMind AI Assistant..."):
    retriever, llm = load_rag()

# ─────────────────────────────────
# INITIALIZE CHAT HISTORY
# ─────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hi! I am the BuildMind AI Assistant. Ask me about our services, pricing, or how to get started!"
    })

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ─────────────────────────────────
# CHAT FUNCTION WITH MEMORY
# ─────────────────────────────────
def ask_with_memory(question, chat_history):

    # Format chat history for context
    history_text = ""
    for msg in chat_history[-6:]:  # last 6 messages only
        if msg["role"] == "user":
            history_text += f"User: {msg['content']}\n"
        else:
            history_text += f"Assistant: {msg['content']}\n"

    # Get relevant docs from RAG
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Build prompt with memory + context
    prompt = ChatPromptTemplate.from_template("""
You are a professional sales assistant for BuildMind AI.

Previous conversation:
{history}

Company information:
{context}

Rules:
1. Remember everything from the conversation history
2. Answer using company information when relevant
3. If someone tells you their name — remember it
4. If answer not in context and not in history say:
   'Please contact us at buildmindai.solutions@gmail.com'
5. Be helpful, confident and professional

Current question: {question}
""")

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "history": history_text,
        "context": context,
        "question": question
    })

    return response

# ─────────────────────────────────
# USER INPUT
# ─────────────────────────────────
if prompt_input := st.chat_input("Ask me anything..."):

    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt_input
    })
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Get AI response with memory
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask_with_memory(
                prompt_input,
                st.session_state.messages
            )
        st.markdown(response)

    # Save response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
