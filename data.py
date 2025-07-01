from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

#============== LOAD QA DATA ====================
ds = load_dataset("thangvip/medical-data")
data = ds["train"]["text"]
# print(len(data))

#============== EXTRACT QUESTION & SPLIT CHUNKS =================
def question_extract(text, prev_question=None):
    match = re.search(r"[^.?!]*\?", text)
    if match:
        return match.group(0).strip()
    else:
        return prev_question

splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

def create_chunks(extract_data):
    prev_q=None
    documents = []

    # text_chunks = splitter.split_text(extract_data)
    for text in data:
        text_chunks = splitter.split_text(text)
        for chunk in text_chunks:
            question = question_extract(chunk, prev_q)
            documents.append({
                "content": chunk,
                "metadata": {"question": question}
            })
            prev_q = question
    return documents

document = create_chunks(extract_data=data)

print("shape of document: ", document[4])

