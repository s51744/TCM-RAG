from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
import os

# Step 1：讀book/內全部PDF
pdf_folder = "book"
pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
docs = []
for pdf_path in pdf_files:
    loader = PyPDFLoader(pdf_path)
    docs.extend(loader.load())

# Step 2：分段文本
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)
split_docs = splitter.split_documents(docs)
with open("split_docs.pkl", "wb") as f:
    pickle.dump(split_docs, f)
print(f"已分段文本數量：{len(split_docs)}")

# Step 3：文本嵌入並建立FAISS向量庫
embeddings_model_name = "BAAI/bge-large-zh-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

db = FAISS.from_documents(split_docs, embeddings)
db.save_local("tcm_vector_db")
print("向量庫建立完成！")
