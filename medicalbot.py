import os
import streamlit as st

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb import PersistentClient
from langchain_google_genai import ChatGoogleGenerativeAI

# Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

vector_store_folder = "./vectorstore"
DATA_PATH = os.path.join(vector_store_folder)

# Load db
model_embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={'device': 'cpu'}
)
client = PersistentClient(path=DATA_PATH)
COLLECTION_NAME = "my_chunks"
collection = client.get_or_create_collection(COLLECTION_NAME)

@st.cache_resource
def get_vectorstore():
    # Load vector store
    db = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=model_embedding,
    )
    return db


def setup_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["document", "question"])
    return prompt

# load llm
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


def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Dựa vào thông tin trong ngữ cảnh dưới đây, hãy trả lời câu hỏi của người dùng một cách chi tiết và rõ ràng.
                Nếu không tìm thấy câu trả lời, chỉ cần nói "Tôi không biết", không được bịa.
                
                Ngữ cảnh: {context}
                Câu hỏi: {question}
                
                Hãy bắt đầu trả lời ngay. Không cần chào hỏi, nói ngắn gọn và chính xác.
                """

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(model_name=MODEL_NAME),
                chain_type="stuff",
                return_source_documents=True,
                retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
                chain_type_kwargs={'prompt': setup_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})

            result = response["result"]
            source_documents = response["source_documents"]
            result_to_show = result + "\nSource Docs:\n" + str(source_documents)
            # response="Hi, I am MediBot!"
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()