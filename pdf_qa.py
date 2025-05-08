import os
import io
import chainlit as cl
import PyPDF2
from io import BytesIO
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_groq import ChatGroq
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Define system prompt template
system_template = """Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should reference the source of the document from which you got your answer.

Example response:
```
The answer is foo
SOURCES: xyz
```

Begin!
----------------
{summaries}"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

@cl.on_chat_start
async def on_chat_start():
    # Sending a welcome message with an image
    elements = [
        cl.Image(name="image1", display="inline", path="./groq.jpeg")
    ]
    await cl.Message(content="Hello there, Welcome to AskAnyQuery related to Data!", elements=elements).send()
    
    files = None
    # Wait for the user to upload a PDF file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    # Check if files is a list and has at least one file
    if not isinstance(files, list) or len(files) == 0:
        await cl.Message(content="Error: No file uploaded.").send()
        return

    file = files[0]
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Read the PDF file
    try:
        # Attempt to access file content
        file_content = None
        if hasattr(file, 'content'):
            file_content = file.content
        elif hasattr(file, 'read'):
            file_content = await file.read()
        elif hasattr(file, 'path'):
            with open(file.path, 'rb') as f:
                file_content = f.read()
        else:
            await cl.Message(content=f"Error: Unable to access content of `{file.name}`. No content, read, or path attribute found.").send()
            return

        if not file_content:
            await cl.Message(content=f"Error: File `{file.name}` is empty or unreadable.").send()
            return

        pdf_stream = BytesIO(file_content)
        pdf = PyPDF2.PdfReader(pdf_stream)
        pdf_text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pdf_text += page_text
    except Exception as e:
        await cl.Message(content=f"Error processing `{file.name}`: {str(e)}").send()
        return

    # Split the text into chunks
    texts = text_splitter.split_text(pdf_text)
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create FAISS vector store with Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docsearch = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    # Initialize the Groq model and create the retrieval chain
    try:
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            ChatGroq(temperature=0, api_key=GROQ_API_KEY, model="llama3-8b-8192"),
            chain_type="stuff",
            retriever=docsearch.as_retriever(),
            chain_type_kwargs=chain_type_kwargs,
        )
    except Exception as e:
        await cl.Message(content=f"Error initializing model: {str(e)}").send()
        return

    # Save metadata, texts, and chain in the user session
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts", texts)
    cl.user_session.set("chain", chain)

    # Notify user that processing is complete
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # Retrieve the chain from session
    if not chain:
        await cl.Message(content="Error: No chain found. Please upload a PDF first.").send()
        return

    # Initialize callback handler for streaming
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    # Call the chain with the user's message
    res = await chain.acall({"question": message.content}, callbacks=[cb])

    answer = res["answer"]
    sources = res.get("sources", "").strip()
    source_elements = []

    # Retrieve metadata and texts from session
    metadatas = cl.user_session.get("metadatas")
    texts = cl.user_session.get("texts")
    all_sources = [m["source"] for m in metadatas]

    if sources:
        found_sources = []
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            try:
                index = all_sources.index(source_name)
                text = texts[index]
                found_sources.append(source_name)
                source_elements.append(cl.Text(content=text, name=source_name))
            except ValueError:
                continue

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"
    else:
        answer += "\nNo sources provided"

    # Send the final response
    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer, elements=source_elements).send()