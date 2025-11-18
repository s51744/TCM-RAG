import os
import json
import torch
import time
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ==== 基本參數 ====
max_token = 512
top_k = 5
max_retries = 5
retry_delay = 30
model_name = "QLU-NLP/BianCang-Qwen2-7B-Instruct"
embeddings_model_name = "BAAI/bge-large-zh-v1.5"
vector_db_dir = "./tcm_vector_db"
exam_dir = "./llm_exam"
result_dir = "./llm_result"
os.makedirs(result_dir, exist_ok=True)

# ==== 模型加載 ====
for attempt in range(max_retries):
    try:
        print(f"Attempt {attempt + 1}/{max_retries} to load model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
        model.eval()
        print("Model and tokenizer loaded successfully.")
        break
    except Exception as e:
        print(f"Download failed: {e}")
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            print("Max retries reached. Could not download the model.")
            raise

# ==== FAISS 向量庫載入 ====
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = FAISS.load_local(vector_db_dir, embeddings=embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": top_k})

def build_rag_prompt(question, options=None, task_type=None, retrieved_docs=None):
    """根據檢索、題型自動構建RAG prompt"""
    context_with_refs = ""
    if retrieved_docs:
        for i, doc in enumerate(retrieved_docs):
            source = doc.metadata.get('source', '未知來源')
            page = doc.metadata.get('page', '未知頁數')
            context_with_refs += f"[資料{i+1} 來源:{source}, 頁:{page}]\n{doc.page_content}\n\n"

    # 根據題型分類
    if task_type == "choice":  # 選擇題
        rag_prompt = (
            f"{context_with_refs}\n"
            f"請根據以上資訊，僅輸出最合適的選項代碼（A、B、C...），不附解釋。\n"
            f"問題：{question}\n選項：{options}\n"
        )
    elif task_type == "entity_extraction":
        rag_prompt = (
            f"{context_with_refs}\n"
            f"請根據以上資訊，結構化JSON格式列出所有中醫實體內容。\n"
            f"問題：{question}\n"
        )
    else:  # 開放題/簡答題
        rag_prompt = (
            f"{context_with_refs}\n"
            f"請根據以上資訊，以條列式中文摘要簡明回答，回答需引註對應來源編號。\n"
            f"問題：{question}\n"
        )
    return rag_prompt

def detect_task_type(q_obj):
    """自動判斷題型"""
    if "options" in q_obj and isinstance(q_obj["options"], list):
        return "choice"
    # 若有明顯結構化/抽取，可加條件
    # if "抽取題特徵" in q_obj["question"]: return "entity_extraction"
    return "open"

def rag_generate_answer(question, options=None, task_type=None):
    # 檢索知識文段
    retrieved_docs = retriever.invoke(question)
    # 組建prompt
    rag_prompt = build_rag_prompt(question, options, task_type, retrieved_docs)
    inputs = tokenizer(rag_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_token,
            do_sample=False,
            temperature=0.7,
            top_k=50,
        )
        answer_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # 後處理
        if task_type == "choice":
            import re
            match = re.search(r"\b[ABCDE]\b", answer_text)
            answer = match.group() if match else ""
        elif task_type == "entity_extraction":
            import json
            try:
                json_struct = json.loads(answer_text)
                answer = json.dumps(json_struct, ensure_ascii=False)
            except Exception:
                answer = answer_text
        else:  # 普通開放題
            answer = answer_text.strip()
        return answer

# ==== 批量處理所有題目 ====
for fname in os.listdir(exam_dir):
    if not fname.endswith(".jsonl"):
        continue
    src_path = os.path.join(exam_dir, fname)
    dst_path = os.path.join(result_dir, fname)
    print(f"開始處理：{fname}")

    with open(src_path, "r", encoding="utf-8") as fin, open(dst_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=fname):
            try:
                q_obj = json.loads(line)
            except Exception as e:
                print(f"JSON格式錯誤: {e}")
                continue

            task_type = detect_task_type(q_obj)
            question = q_obj.get("question")
            options = q_obj.get("options") if "options" in q_obj else None

            answer = rag_generate_answer(question, options, task_type)
            q_obj["answer"] = answer
            fout.write(json.dumps(q_obj, ensure_ascii=False)+'\n')

    print(f"完成：{fname}，結果已存 {dst_path}")

print("全部題庫處理完畢，llm_result 資料夾已生成！")
