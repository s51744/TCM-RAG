from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
import os
from tqdm import tqdm
import faiss # 引入 faiss 庫
import numpy as np # 引入 numpy 處理向量

# --- 參數設定 ---
PDF_FOLDER = "book"
VECTOR_DB_DIR = "tcm_vector_db_hnsw_batched" # 儲存至 HNSW 專用目錄
EMBEDDINGS_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
BATCH_SIZE = 1000 # 每批處理的文檔數量 (可調整)
HNSW_M_PARAM = 12 # <--- 關鍵 HNSW 內存優化：降低連線數 M 值 (建議 8-16)

# Step 1：讀取並分段文本 (保持不變)
pdf_files = [os.path.join(PDF_FOLDER, f) for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]
docs = []
for pdf_path in pdf_files:
    loader = PyPDFLoader(pdf_path)
    docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)
split_docs = splitter.split_documents(docs)
print(f"已分段文本總數量：{len(split_docs)}")

# Step 2：初始化嵌入器
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
# 確保我們知道嵌入器的維度 (BGE-large-zh-v1.5 維度為 1024)
D_DIMENSION = 1024 
print("Embedding 模型載入完成。")


# Step 3：分批嵌入並建立/合併 HNSW FAISS 向量庫
num_chunks = len(split_docs)
num_batches = (num_chunks + BATCH_SIZE - 1) // BATCH_SIZE
index_list = []
print(f"開始分批嵌入 HNSW 索引，總共 {num_batches} 批次。")

for i in tqdm(range(num_batches), desc="HNSW 批次處理進度"):
    start_index = i * BATCH_SIZE
    end_index = min((i + 1) * BATCH_SIZE, num_chunks)
    batch_docs = split_docs[start_index:end_index]
    
    print(f"\n   -> 處理批次 {i+1}/{num_batches} ({len(batch_docs)} 個文檔)...")

    # 1. 創建 HNSW 索引配置
    hnsw_index_params = {
        "ef_construction": 200, # 參數 ef_construction 影響建圖精度
        "M": HNSW_M_PARAM      # 使用您設定的 M=12 參數
    }

    # 2. 創建單批次的 FAISS 實例 (LangChain 會在內部處理 HNSW 索引的建構和元數據)
    # 這是標準的 HNSW 創建方法，並將該批次數據加入
    batch_db = FAISS.from_documents(
        batch_docs, 
        embeddings, 
        # 將 HNSW 參數傳遞給 FAISS 構造函數
        # LangChain 知道如何使用這些參數來創建 IndexHNSWFlat
        faiss_index_kwargs=hnsw_index_params
    )
    
    index_list.append(batch_db)

    # 3. 釋放該批次的內存，只保留索引物件
    del batch_db

# Step 4：合併所有索引
print("\n--- 開始合併所有 FAISS 索引 ---")
if not index_list:
    print("沒有索引可以合併。")
    exit()

# 以第一個索引作為起始點
main_index_db = index_list[0] 

# 將剩餘的索引逐一合併到 main_index_db 中 (merge_from 會自動處理 index 和 docstore)
for i in tqdm(range(1, len(index_list)), desc="合併索引進度"):
    main_index_db.merge_from(index_list[i])


# Step 4：保存最終的 HNSW 向量庫
if main_index_db:
    main_index_db.save_local(VECTOR_DB_DIR)
    print(f"\n✅ HNSW 向量庫建立完成！總共 {main_index_db.index.ntotal} 個文檔。儲存至 {VECTOR_DB_DIR}")
else:
    print("❌ 索引建構失敗。")