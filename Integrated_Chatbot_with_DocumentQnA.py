import os
import io
import chainlit as cl
import pdfplumber
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
import time
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Define system prompt template for PDF-based queries
pdf_system_template = """Use the following pieces of context to answer the user's question.
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

pdf_messages = [
    SystemMessagePromptTemplate.from_template(pdf_system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
pdf_prompt = ChatPromptTemplate.from_messages(pdf_messages)
chain_type_kwargs = {"prompt": pdf_prompt}

# Define system prompt for general queries
general_system_template = """You are a helpful AI assistant. Answer the user's question to the best of your ability. If you don't know the answer, say so. Do not provide sources unless specifically asked."""
general_messages = [
    SystemMessagePromptTemplate.from_template(general_system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
general_prompt = ChatPromptTemplate.from_messages(general_messages)

@cl.on_chat_start
async def on_chat_start():
    # Initialize Groq model for general queries
    try:
        llm = ChatGroq(temperature=0, api_key=GROQ_API_KEY, model="llama3-8b-8192")
        cl.user_session.set("llm", llm)
        logger.info("Groq model initialized successfully")
    except Exception as e:
        await cl.Message(content=f"Error initializing model: {str(e)}").send()
        logger.error(f"Error initializing model: {str(e)}")
        return

    # Sending a welcome message with an image
    elements = [
        cl.Image(name="image1", display="inline", path="./groq.jpeg")
    ]
    await cl.Message(
        content="Hello! Welcome to AskAnyQuery. You can ask general questions or upload a PDF file in the message bar to query its content!",
        elements=elements
    ).send()

async def process_pdf(file, msg):
    """Process a PDF file and return texts and metadatas."""
    start_time = time.time()
    try:
        logger.info(f"Starting PDF processing for {file.name}")
        # Log file object attributes and details
        logger.info(f"File object attributes: {dir(file)}")
        logger.info(f"File name: {file.name}, MIME: {file.mime}, Path: {getattr(file, 'path', 'None')}")

        file_content = None
        # Try file.path first (preferred for Windows)
        if hasattr(file, 'path') and getattr(file, 'path', None):
            try:
                with open(file.path, 'rb') as f:
                    file_content = f.read()
                logger.info(f"Accessed file.path: {len(file_content)} bytes")
            except Exception as e:
                logger.error(f"Error reading file.path: {str(e)}")
        # Try file.content
        if file_content is None and hasattr(file, 'content'):
            file_content = file.content
            logger.info(f"Accessed file.content: {'None' if file_content is None else f'{len(file_content)} bytes'}")
        # Try file.read()
        if file_content is None and hasattr(file, 'read'):
            file_content = await file.read()
            logger.info(f"Accessed file.read(): {'None' if file_content is None else f'{len(file_content)} bytes'}")

        if file_content is None:
            logger.error("File content is None after all access attempts")
            return None, None, f"Error: Unable to read `{file.name}`. The file may be corrupted, empty, or inaccessible."

        pdf_stream = BytesIO(file_content)
        pdf_text = ""
        max_pages = 20  # Limit to first 20 pages
        enable_ocr = False  # Disable OCR by default (set to True for scanned PDFs)

        with pdfplumber.open(pdf_stream) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"PDF has {total_pages} pages")
            if total_pages == 0:
                logger.error("PDF has no pages")
                return None, None, f"Error: File `{file.name}` has no pages."

            for page_number, page in enumerate(pdf.pages[:max_pages], 1):
                start_page_time = time.time()
                page_text = page.extract_text()
                if page_text:
                    pdf_text += page_text + "\n"
                    logger.info(f"Extracted {len(page_text)} characters from page {page_number} in {time.time() - start_page_time:.2f}s")
                elif enable_ocr:
                    # Try OCR for image-based pages
                    try:
                        from pdf2image import convert_from_bytes
                        import pytesseract
                        images = convert_from_bytes(file_content, first_page=page_number, last_page=page_number, dpi=150)
                        for image in images:
                            ocr_text = pytesseract.image_to_string(image)
                            if ocr_text.strip():
                                pdf_text += ocr_text + "\n"
                                logger.info(f"Extracted {len(ocr_text)} characters from page {page_number} via OCR in {time.time() - start_page_time:.2f}s")
                    except Exception as ocr_e:
                        logger.warning(f"OCR error on page {page_number}: {str(ocr_e)}")
                        continue

                # Send progress update every 2 pages
                if page_number % 2 == 0 or page_number == min(total_pages, max_pages):
                    msg.content = f"Processing `{file.name}`... ({page_number}/{min(total_pages, max_pages)} pages processed)"
                    await msg.send()
                    await asyncio.sleep(0.1)  # Allow UI to update

                # Check for timeout (30 seconds)
                if time.time() - start_time > 30:
                    logger.warning("PDF processing timed out after 30 seconds")
                    return None, None, f"Error: Processing `{file.name}` timed out after 30 seconds."

        if not pdf_text.strip():
            logger.error("No text extracted from PDF")
            return None, None, f"Error: No readable text found in `{file.name}`. It may be a scanned or image-based PDF. Try enabling OCR by setting enable_ocr=True."

        # Split the text into chunks
        start_split_time = time.time()
        texts = text_splitter.split_text(pdf_text)
        metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]
        logger.info(f"Split into {len(texts)} text chunks in {time.time() - start_split_time:.2f}s")

        logger.info(f"Total PDF processing time: {time.time() - start_time:.2f}s")
        return texts, metadatas, None
    except Exception as e:
        logger.error(f"PDF processing error: {str(e)}")
        return None, None, f"Error processing `{file.name}`: {str(e)}"

@cl.on_message
async def main(message: cl.Message):
    # Check for file attachments
    files = message.elements if message.elements else []
    pdf_processed = False

    if files:
        for file in files:
            if file.mime == "application/pdf":
                msg = cl.Message(content=f"Processing `{file.name}`...")
                await msg.send()

                texts, metadatas, error = await process_pdf(file, msg)
                if error:
                    msg.content = error
                    await msg.send()
                    # Continue to allow general questions
                    break

                # Create or update FAISS vector store
                start_faiss_time = time.time()
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                docsearch = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
                logger.info(f"FAISS vector store created in {time.time() - start_faiss_time:.2f}s")

                # Create retrieval chain
                try:
                    chain = RetrievalQAWithSourcesChain.from_chain_type(
                        ChatGroq(temperature=0, api_key=GROQ_API_KEY, model="llama3-8b-8192"),
                        chain_type="stuff",
                        retriever=docsearch.as_retriever(),
                        chain_type_kwargs=chain_type_kwargs,
                    )
                    # Save to session
                    cl.user_session.set("chain", chain)
                    cl.user_session.set("metadatas", metadatas)
                    cl.user_session.set("texts", texts)
                    pdf_processed = True
                    msg.content = f"Processing `{file.name}` done. You can now ask questions about the PDF or general queries!"
                    await msg.send()
                except Exception as e:
                    msg.content = f"Error initializing model: {str(e)}"
                    await msg.send()
                    # Continue to allow general questions
                    break

    # Handle the user's question
    question = message.content.strip()
    if not question and not pdf_processed:
        await cl.Message(content="Please ask a question or upload a PDF.").send()
        return

    if question:
        chain = cl.user_session.get("chain")
        llm = cl.user_session.get("llm")

        if chain:
            # Use retrieval chain for PDF-based questions
            cb = cl.AsyncLangchainCallbackHandler(
                stream_final_answer=True,
                answer_prefix_tokens=["FINAL", "ANSWER"]
            )
            cb.answer_reached = True
            res = await chain.acall({"question": question}, callbacks=[cb])

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

            # Send the response
            if cb.has_streamed_final_answer:
                cb.final_stream.elements = source_elements
                await cb.final_stream.update()
            else:
                await cl.Message(content=answer, elements=source_elements).send()
        else:
            # Use general LLM for non-PDF questions
            try:
                response = await llm.ainvoke(question)
                await cl.Message(content=response.content).send()
            except Exception as e:
                await cl.Message(content=f"Error answering general question: {str(e)}").send()
                return