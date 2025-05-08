import os
import chainlit as cl
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableMap, RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from dotenv import load_dotenv
import time
import uuid
import warnings
import tempfile
import shutil

# Suppress warnings from huggingface_hub
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

# Load environment variables
load_dotenv()

# Clear any cached Hugging Face token to avoid authentication issues
os.environ.pop("HF_TOKEN", None)
os.environ["HF_TOKEN"] = ""

# Environment variables setup
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY") or ""
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY") or ""

# LangSmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Chainlit Q&A and RAG Chatbot"

# Retry logic for embeddings
def load_embeddings(model_name, retries=3):
    for attempt in range(retries):
        try:
            return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"trust_remote_code": False})
        except Exception as e:
            if attempt == retries - 1:
                raise Exception(f"Failed to load embeddings for {model_name} after {retries} attempts: {str(e)}")
            time.sleep(5)

# Initialize embeddings
primary_model = "sentence-transformers/paraphrase-MiniLM-L3-v2"
fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
try:
    embeddings = load_embeddings(primary_model)
except Exception as e:
    print(f"Primary model {primary_model} failed: {str(e)}. Trying fallback model {fallback_model}")
    embeddings = load_embeddings(fallback_model)

# Simple Q&A Prompt Template
simple_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Provide concise and accurate answers to user queries."),
    ("user", "{question}")
])

# Document Q&A Prompt Templates
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question, "
    "formulate a standalone question that can be understood without the chat history. "
    "DO NOT answer the question, just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say so. Use three sentences or less.\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

@cl.on_chat_start
async def on_chat_start():
    # Initialize Groq LLM (Gemma2-9b-It)
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        await cl.Message(content="Error: GROQ_API_KEY not found in environment variables.").send()
        return

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model="Gemma2-9b-It",
        temperature=0.5,
        max_tokens=150
    )

    # Simple Q&A chain
    simple_qa_chain = simple_qa_prompt | llm | StrOutputParser()

    # Session history storage
    cl.user_session.set("store", {})
    cl.user_session.set("simple_qa_chain", simple_qa_chain)
    cl.user_session.set("llm", llm)

    # Create temporary directory for uploads
    temp_dir = tempfile.mkdtemp()
    cl.user_session.set("temp_dir", temp_dir)
    print(f"Created temporary directory: {temp_dir}")

    # Welcome message with mode selection
    await cl.Message(
        content="Welcome to the Chatbot! Choose a mode:\n- **General Q&A**: Ask any question.\n- **Document Q&A**: Upload PDFs to query their content with chat history."
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    content = message.content.strip().lower()
    store = cl.user_session.get("store")
    llm = cl.user_session.get("llm")
    temp_dir = cl.user_session.get("temp_dir")

    # Handle mode selection or PDF upload
    if content in ["general q&a", "general", "q&a"]:
        cl.user_session.set("mode", "general")
        await cl.Message(content="Switched to General Q&A mode. Ask your question!").send()
    elif content in ["document q&a", "document", "pdf", "doc"]:
        cl.user_session.set("mode", "document")
        await cl.Message(content="Switched to Document Q&A mode. Please upload PDF files.").send()
    elif message.elements:  # Handle PDF uploads
        mode = cl.user_session.get("mode")
        if mode != "document":
            await cl.Message(content="Please switch to Document Q&A mode to upload PDFs.").send()
            return

        documents = []
        for element in message.elements:
            print(f"Processing element: name={element.name}, mime={element.mime}, content_type={type(element.content)}, content_length={len(element.content) if element.content else 0}")
            if element.mime != "application/pdf":
                await cl.Message(content=f"Skipping {element.name}: Not a PDF file.").send()
                continue
            if not element.content:
                await cl.Message(content=f"Error: No content found for {element.name}. Please try uploading again.").send()
                continue

            # Save file to temporary directory
            temp_file_path = os.path.join(temp_dir, f"temp_{element.name}")
            try:
                with open(temp_file_path, "wb") as f:
                    f.write(element.content)
                print(f"Saved file to: {temp_file_path}")
                loader = PDFPlumberLoader(temp_file_path)
                docs = loader.load()
                documents.extend(docs)
                print(f"Loaded {len(docs)} documents from {element.name}")
            except Exception as e:
                await cl.Message(content=f"Error loading PDF {element.name}: {str(e)}").send()
                print(f"Error loading {element.name}: {str(e)}")
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    print(f"Removed temporary file: {temp_file_path}")

        if not documents:
            await cl.Message(content="No valid PDFs loaded. Please try again.").send()
            return

        # Create vectorstore
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(splits, embeddings)
        base_retriever = vectorstore.as_retriever()

        # Create RAG chain
        history_aware_retriever = create_history_aware_retriever(llm, base_retriever, contextualize_q_prompt)
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = RunnableMap({
            "context": history_aware_retriever,
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"]
        }) | question_answer_chain

        # Session history management
        session_id = cl.user_session.get("session_id", str(uuid.uuid4()))
        cl.user_session.set("session_id", session_id)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in store:
                store[session] = ChatMessageHistory()
            return store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        cl.user_session.set("rag_chain", conversational_rag_chain)
        await cl.Message(content="PDFs loaded successfully! Ask questions about the documents.").send()

    else:
        # Handle questions based on mode
        mode = cl.user_session.get("mode", "general")
        if mode == "general":
            simple_qa_chain = cl.user_session.get("simple_qa_chain")
            response = await simple_qa_chain.ainvoke(
                {"question": content},
                callbacks=[cl.AsyncLangchainCallbackHandler()]
            )
            await cl.Message(content=response).send()
        elif mode == "document":
            rag_chain = cl.user_session.get("rag_chain")
            if not rag_chain:
                await cl.Message(content="Please upload PDFs first to use Document Q&A mode.").send()
                return
            session_id = cl.user_session.get("session_id")
            session_history = store.get(session_id, ChatMessageHistory())
            config = {
                "configurable": {"session_id": session_id},
                "chat_history": session_history.messages
            }
            response = await rag_chain.ainvoke(
                {"input": content},
                config=config,
                callbacks=[cl.AsyncLangchainCallbackHandler()]
            )
            await cl.Message(content=response["answer"]).send()

@cl.on_chat_end
async def on_chat_end():
    # Clean up temporary directory
    temp_dir = cl.user_session.get("temp_dir")
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"Removed temporary directory: {temp_dir}")