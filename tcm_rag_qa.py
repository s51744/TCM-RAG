import torch
import requests
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 參數
max_token = 512
top_k = 5

# 載入模型
model_name = "QLU-NLP/BianCang-Qwen2-7B-Instruct"
embeddings_model_name = "BAAI/bge-large-zh-v1.5"

max_retries = 5   #模型下載/初始化遇錯最多重試次數
retry_delay = 30  # 秒

for attempt in range(max_retries):
    try:
        print(f"Attempt {attempt + 1}/{max_retries} to load model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map="auto")
        model.eval()
        print("Model and tokenizer loaded successfully.")
        break
    except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError, 
            requests.exceptions.ReadTimeout, requests.exceptions.ProxyError, 
            requests.exceptions.SSLError) as e:
        print(f"Download failed: {e}")
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            print("Max retries reached. Could not download the model.")
            raise

# 載入等同建庫的 embedding，並讀取向量庫
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = FAISS.load_local("tcm_vector_db", embeddings=embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": top_k})

query = "請問桂枝湯適合什麼時候使用？"
retrieved_docs = retriever.invoke(query)

# 顯示檢索段落（Debug用）
context_with_refs = ""
for i, doc in enumerate(retrieved_docs):
    source = doc.metadata.get('source', '未知來源')
    page = doc.metadata.get('page', '未知頁數')  # 根據PyPDFLoader
    context_with_refs += f"[資料{i+1} 來源:{source}, 頁:{page}]\n{doc.page_content}\n\n"

# 組 prompt+流式生成
context = "\n".join([doc.page_content for doc in retrieved_docs])
rag_prompt = (
    f"以下為檢索到的知識段落，內容皆自中醫經典或教材，段落來源已標註：「來源:[檔名], [頁數]」\n"
    f"{context_with_refs}\n"
    f"請根據以上資料，用中文條列式簡明摘要來回答下列問題，並於每一條答案後引出對應的來源（如來源[資料2]...）。\n\n"
    f"問題：{query}\n"
)

inputs = tokenizer(rag_prompt, return_tensors="pt").to(model.device)
streamer = TextStreamer(tokenizer)

with torch.no_grad():
    model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_token,
        do_sample=False,
        temperature=0.7,
        top_k=50,
    )
