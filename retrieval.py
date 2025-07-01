import os
from chromadb import PersistentClient
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

#============ LOAD LLM ===================
MODEL_NAME="gemini-1.5-flash"
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

def load_llm(model_name):
    if GOOGLE_API_KEY is None:
        raise ValueError("GOOGLE_API_KEY haven't already created in environment.")

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.5
    )
    return llm
llm=load_llm(MODEL_NAME)

#============== CONNECT LLM with CHROMA and CREATE CHAIN ================
CUSTOM_PROMPT_TEMPLATE = """
Dựa vào thông tin được cung cáp dưới đây, hãy trả lời câu hỏi của người dùng một cách chi tiết và rõ ràng.
Nếu không tìm thấy bất cứ thông tin nào, chỉ cần nói "Tôi không biết", không được bịa.

Ngữ cảnh: {context}
Câu hỏi: {question}

Hãy bắt đầu trả lời ngay. Không cần chào hỏi, nói ngắn gọn và chính xác.
"""
def setup_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["document", "question"])
    return prompt
prompt=setup_prompt(CUSTOM_PROMPT_TEMPLATE)

vector_store_folder = "./vectorstore"
DATA_PATH = os.path.join(vector_store_folder)
COLLECTION_NAME = "my_chunks"

model_embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)
client = PersistentClient(path=DATA_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)

# Load vector store
db = Chroma(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding_function=model_embedding,
)

# Create QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    return_source_documents=True,
    retriever=db.as_retriever(kwargs={'k':5}),
    chain_type_kwargs={'prompt': prompt}
)

# Now invoke with a single query
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
