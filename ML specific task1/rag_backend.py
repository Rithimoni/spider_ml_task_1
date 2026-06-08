import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- UPDATED TO USE THE CLASSIC PACKAGES ---
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

PDF_FOLDER = "papers"
DB_FOLDER = "chroma_db"

def initialize_folders():
    """Creates the papers folder if it doesn't exist."""
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"Created '{PDF_FOLDER}' directory. Please drop your research PDFs inside it!")

def create_vector_db():
    """Reads PDFs, splits them into text chunks, and saves them to a vector store."""
    initialize_folders()
    
    docs = []
    if not os.listdir(PDF_FOLDER):
        print(f"Error: No PDF files found in '{PDF_FOLDER}'. Please add some PDFs first.")
        return None
        
    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            file_path = os.path.join(PDF_FOLDER, file)
            print(f"Loading paper: {file}...")
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            for p in pages:
                p.metadata["source_paper"] = file
            docs.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    print(f"Split documents into {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Generating vectors and creating database. This might take a moment...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_FOLDER
    )
    print("Database built successfully!")
    return db

# --- THIS IS THE UPDATED STEP 2 PART EXCLUSIVELY ---
def build_qa_chain():
    """Loads the database and builds the modern Retrieval Chain pipeline."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=DB_FOLDER,
        embedding_function=embeddings
    )

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

# Load local Text Generation LLM pipeline (Updated for modern transformers tasks)
    generator = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens=256,
        trust_remote_code=True
    )
    llm = HuggingFacePipeline(pipeline=generator)
    
    system_prompt = (
        "You are an academic assistant.\n"
        "Answer the question concisely based strictly on the provided context.\n"
        "If you cannot find the answer in the context, respond with: "
        "'I could not find sufficient evidence in the papers.'\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain