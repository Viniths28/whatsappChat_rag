from langchain_openai import AzureOpenAIEmbeddings
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader
#from langchain.document_loaders.csv_loader import CSVLoader
from dotenv import load_dotenv
import os

# Load env variables from .env file
load_dotenv()

# Load documents from CSV
loader = CSVLoader(file_path='unique_rag_context.csv')
docs = loader.load()

# Set up Azure OpenAI Embeddings
embedding = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),        
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
)

# Store in Chroma DB
db = Chroma.from_documents(docs, embedding, persist_directory="db")
db.persist()

print("✅ ChromaDB created with Azure OpenAI and saved.")
