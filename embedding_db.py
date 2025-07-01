from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb import PersistentClient
import uuid
import re

# Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

#============== LOAD QA DATA ====================
ds = load_dataset("thangvip/medical-data")
data = ds["train"]["text"]
data_docs = ''.join(data)
# print(len(data))

#============== EXTRACT QUESTION & SPLIT CHUNKS =================
def question_extract(text, prev_question=None):
    match = re.search(r"[^.?!]*\?", text)
    if match:
        return match.group(0).strip()
    else:
        return prev_question

def create_chunks(extract_data):
    prev_q=None
    documents = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    text_chunks = splitter.split_text(extract_data)
    for text in text_chunks:
        question = question_extract(text, prev_q)
        documents.append({
            "content": text,
            "metadata": {"question": question}
        })
        prev_q = question
    return documents

document = create_chunks(extract_data=data_docs)

# print("shape of document: ", document[4])

#============== LOAD MODEL EMBEDDING==============
def get_embedding():
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    return model


#================ VECTOR STORE =================
def embedding_store(document):
    persist_path = "./vectorstore"

    # Create db
    client = PersistentClient(path=persist_path)
    collection = client.get_or_create_collection("my_chunks")

    print("Vectorstore is ready, start add new data...")
    model = get_embedding()

    # Implement db
    for chunk in document:
        if not chunk.get("content"):
            continue

        # Auto save local
        vector = model.embed_documents([chunk["content"]])[0]
        collection.add(
            documents=[chunk["content"]],
            metadatas=[chunk["metadata"]],
            embeddings=[vector],
            ids=[str(uuid.uuid4())]
        )

    print(f"Added {len(document)} documents and save in {persist_path}")
embedding_store(document)