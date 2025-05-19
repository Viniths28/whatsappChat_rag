from langchain_openai import AzureOpenAIEmbeddings
#from langchain.chat_models import AzureChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA
#from langchain.vectorstores import Chroma
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

llm = AzureChatOpenAI(
    deployment_name=os.getenv("OPENAI_DEPLOYMENT"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION"), 
    model_name="gpt-35-turbo"
)

embedding = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),        
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
)

def get_qa():
    retriever = Chroma(persist_directory="db", embedding_function=embedding).as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def ask_question(query: str):
    qa = get_qa()
    result = qa.invoke({"query":query})
    return result["result"]




#if __name__ == "__main__":
 #   while True:
  #      query = input("Ask a question (or type 'exit'): ")
   #     if query.lower() == "exit":
    #        break
     #   try:
      #      answer = ask_question(query)
       #     print(f"\n🔎 Answer: {answer}\n")
        #except Exception as e:
         #   print(f"❌ Error: {str(e)}")
