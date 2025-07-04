from datasets import load_dataset
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import re


ds = load_dataset("NTCong/medical_qa_vn")
print(len(ds))
question = ds["train"]["QUESTION"]
context = ds["train"]["CONTEXT"]
answer = ds["train"]["ANSWER"]


def convert_to_documents(raw_documents):
    documents = []
    for doc in raw_documents:
        c = doc.get("content")
        content = c if c is not None else ""

        documents.append(
            Document(
                page_content=content,
                metadata=doc.get("metadata"),
            )
        )
    return documents

def create_document(question, context, answer):
    raw_documents = []
    for q, c, a in zip(question, context, answer):
        if c is None:
            c = ""

        raw_documents.append({
            "content": c,
            "metadata": {
                "question": q,
                "answer": a,
            }
        })

    documents = convert_to_documents(raw_documents)
    return documents
documents = create_document(question=question, context=context, answer=answer)
# print("type(documents[0]):", type(documents[0]))

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40
    )
    all_chunks=[]
    for i, doc in enumerate(documents):
        chunks = splitter.split_documents([doc])
        for j, chunk in enumerate(chunks):
            all_chunks.append(chunk)
    return all_chunks
all_chunks = split_documents(documents)
print("Shape of chunk", all_chunks[2])


