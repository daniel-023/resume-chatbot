from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document  # Import Document class
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Read the resume text file
with open('data/resume.txt', 'r', encoding='UTF-8') as f:
    docs = f.read()

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)
splits = text_splitter.split_text(docs)

# Wrap each split in a Document object
documents = [Document(page_content=split) for split in splits]

# Initialize HuggingFace embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize the vector store with persistence
vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory="./chroma_db")