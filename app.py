import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISSfrom langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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

    prompt = ChatPromptTemplate.from_template("""
You are a professional sales assistant for BuildMind AI.
Answer questions using ONLY the context provided.
Be helpful, confident and professional.
If answer not in context say:
'Please contact us at buildmindai.solutions@gmail.com'

Context: {context}
Question: {question}
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs,
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# Load the chain
with st.spinner("Loading BuildMind AI Assistant..."):
    rag_chain = load_rag()

# ─────────────────────────────────
# CHAT INTERFACE
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

# User input
if prompt_input := st.chat_input("Ask me anything..."):

    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt_input
    })
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(prompt_input)
        st.markdown(response)

    # Save response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
