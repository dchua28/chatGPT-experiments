import os
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"]

raw_documents = TextLoader('LuftAir_chatlog.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=200,
                                      chunk_overlap=0,
                                      separator="\n",)
documents = text_splitter.split_documents(raw_documents)
db = FAISS.from_documents(documents, OpenAIEmbeddings())

retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=retriever
)

query_list = [
    "What airline was being discussed?",
    "Which city did the flight depart from?",
    "Did they fly economy class?",
    "How much was paid for an extra seat from Charlotte?",
    "How much was paid for an extra seat from Munich?",
    "How much was paid for an extra seat from Cebu?",
    "Did the passenger enjoy her flight?"
    ]

for query in query_list:
    print(f"{qa.invoke(query)}\n")
