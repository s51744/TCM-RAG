import torch
import requests
import time
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 基本參數 ---
max_token = 512
top_k = 5
model_name = "QLU-NLP/BianCang-Qwen2-7B-Instruct"
embeddings_model_name = "BAAI/bge-large-zh-v1.5"
max_retries = 5
retry_delay = 30  # 秒

# --- 載入LLM ---
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

# --- 載入embedding及向量庫 ---
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = FAISS.load_local("tcm_vector_db", embeddings=embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": top_k})

# --- 解析 prediction.json/舌診結果並抽取所有舌象名 ---
def parse_tongue_names(path):
    with open(path, 'r', encoding='utf-8') as f:
        pred = json.load(f)
    names = []
    for det in pred.get("detections", []):
        if det.get("confidence", 0) > 0.8:
            names.append(det["name"])
    return list(set(names)) if names else ["未發現異常舌象"]

# --- 問答主函式，針對舌象名稱做RAG並條列體質、病徵 ---
conversation_history = []

def ask_llm(tongue_names, user_query, conversation_history):
    # 當前問題組合舌象做檢索（可改只用user_query、更智能可組合）
    rag_query = "、".join(tongue_names) + " " + user_query
    retrieved_docs = retriever.invoke(rag_query)
    context_with_refs = ""
    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get('source', '未知來源')
        page = doc.metadata.get('page', '未知頁數')
        context_with_refs += f"[資料{i+1} 來源:{source}, 頁:{page}]\n{doc.page_content}\n\n"
    history_txt = "".join([f"{r[0]}：{r[1]}\n" for r in conversation_history])
    prompt = (
        f"以下內容為舌診AI辨識到的舌象：{', '.join(tongue_names)}\n"
        f"{history_txt}"
        f"知識庫最新檢索段落如下（根據用戶最新問題動態取得，來源已標註）：\n"
        f"{context_with_refs}\n"
        f"請依據舌診、歷史上下文和上述資料，針對本輪問題以條列式生成有依據的答案，且每一點都要標注來源，不能重複AI舌診意義內容。\n"
        f"問題：{user_query}\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer)
    with torch.no_grad():
        model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_token,
            do_sample=False
        )

if __name__ == "__main__":
    tongue_names = parse_tongue_names("prediction.json")
    first_reply = f"根據AI分析，您的舌象為：{', '.join(tongue_names)}"
    print(first_reply)
    conversation_history.append(("system", first_reply))
    # <<== 不執行 ask_llm 直到真正 user_input
    while True:
        user_query = input("\n請輸入您的問題（如不再提問請輸入exit）：\n")
        if not user_query or user_query.lower() == "exit":
            print("感謝您的諮詢！")
            break
        conversation_history.append(("user", user_query))
        ask_llm(tongue_names, user_query, conversation_history)
        conversation_history.append(("system", "(以上為AI回覆內容)"))
