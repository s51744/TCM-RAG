import torch
import requests
import time
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

max_token = 512
top_k = 5
model_name = "QLU-NLP/BianCang-Qwen2-7B-Instruct"
embeddings_model_name = "BAAI/bge-large-zh-v1.5"
max_retries = 5
retry_delay = 30

# ---- 舌象分布統計 ----
def get_tongue_stat_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        pred = json.load(f)
    detections = pred.get("detections", [])
    total = sum([d["confidence"] for d in detections])
    stats = {}
    for det in detections:
        name = det["name"]
        stats[name] = stats.get(name, 0) + det["confidence"]
    out = []
    for name, conf in stats.items():
        percent = round(conf / total * 100, 2) if total > 0 else 0
        out.append(f"{name}{percent}%")
    return out

# ---- LLM模型與RAG向量庫載入 ----
for attempt in range(max_retries):
    try:
        print(f"Attempt {attempt + 1}/{max_retries} to load model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="cuda:0"
        )
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

embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = FAISS.load_local("tcm_vector_db", embeddings=embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": top_k})

# ---- 問答主函式，每輪都動態RAG + 多輪history + 舌象分布 ----
conversation_history = []

def ask_llm(tongue_stat_text, user_query, conversation_history):
    rag_query = "、".join(tongue_stat_text) + " " + user_query
    retrieved_docs = retriever.invoke(rag_query)
    context_with_refs = ""
    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get('source', '未知來源')
        page = doc.metadata.get('page', '未知頁數')
        context_with_refs += f"[資料{i+1} 來源:{source}, 頁:{page}]\n{doc.page_content}\n\n"
    history_txt = "".join([f"{r[0]}：{r[1]}\n" for r in conversation_history])
    prompt = (
        f"以下為AI舌診檢測分布：{ '、'.join(tongue_stat_text)}\n"
        f"{history_txt}"
        f"知識庫根據最新提問檢索內容如下（來源註明）：\n"
        f"{context_with_refs}\n"
        f"請依據舌象分布、歷史問答、知識內容，以繁體中文回答本次問題的答案，要標出引用來源，不得重複AI舌診意義。"
        f"\n問題：{user_query}\n"
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

# ---- 主流程 ----
if __name__ == "__main__":
    tongue_stat_text = get_tongue_stat_text("prediction.json")
    first_reply = f"根據AI分析，您的舌象為：{'、'.join(tongue_stat_text)}"
    print(first_reply)
    conversation_history.append(("system", first_reply))
    while True:
        user_query = input("\n請輸入您的問題（如不再提問請輸入exit）：\n")
        if user_query.lower() == "exit":
            print("感謝您的諮詢！")
            break
        conversation_history.append(("user", user_query))
        ask_llm(tongue_stat_text, user_query, conversation_history)
        conversation_history.append(("system", "(以上為AI回覆內容)"))
