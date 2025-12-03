import torch
import json
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

# --- 參數設定 ---
VECTOR_DB_DIR = "./tcm_vector_db_hnsw_batched"
SENTENCE_DIR = "./llm_exam/sentence"
ANSWER_DIR = "./llm_exam/answer"
RESULT_DIR = "./llm_result"
MODEL_NAME = "QLU-NLP/BianCang-Qwen2-7B-Instruct" # <--- 設定為 BianCang 模型
EMBEDDINGS_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
MAX_TOKEN = 3 # 保持 3 以確保速度
TOP_K = 3
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# --- 載入模型與向量庫 ---
print("--- 載入模型與向量庫 ---")
# 載入 BianCang 模型 (Qwen2 架構，不需 trust_remote_code)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if DEVICE != "cpu" else torch.float32,
    # 使用 device_map="auto" 確保模型可以被分層載入
    device_map="auto", 
)
model.eval()

# 載入 RAG 向量庫
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
db = FAISS.load_local(VECTOR_DB_DIR, embeddings=embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": TOP_K})
print("模型與向量庫載入完成。")

# --- RAG 推論函式 ---
def generate_rag_response(question, options):
    """執行 RAG 檢索並生成模型回覆"""
    if isinstance(options, dict):
        option_str = ", ".join([f"{k}: {v}" for k, v in options.items()])
    elif isinstance(options, list):
        option_str = ", ".join([f"{chr(65 + i)}: {option}" for i, option in enumerate(options)])
    else:
        option_str = str(options)

    rag_query = f"{question} {option_str}"
    
    # 1. RAG 檢索
    retrieved_docs = retriever.invoke(rag_query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # 2. 構建 Prompt (包含檢索到的上下文)
    prompt = (
        f"知識庫檢索內容如下：\n{context}\n\n"
        f"請根據檢索內容，以繁體中文選出最正確的選項代號（A, B, C, D, E）。"
        f"請只回答選項代號，不需解釋。\n"
        f"問題：{question}\n"
        f"選項：{option_str}\n"
        f"答案："
    )

    # 3. 模型生成
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKEN,
            do_sample=False,
            use_cache=False, # 確保沒有 KV 快取錯誤
            eos_token_id=tokenizer.eos_token_id 
        )
    
    # 4. 解碼與後處理
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
    match = re.search(r'[A-E]', response)
    predicted_answer = match.group(0) if match else response
    
    return predicted_answer, context

# --- 主評估流程 ---
def run_evaluation():
    """執行所有檔案的評估"""
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    exam_files = [f for f in os.listdir(SENTENCE_DIR) if f.endswith(".jsonl")]

    for filename in exam_files:
        print(f"\n--- 開始評估檔案: {filename} ---")
        sentence_path = os.path.join(SENTENCE_DIR, filename)
        answer_path = os.path.join(ANSWER_DIR, filename)
        result_path = os.path.join(RESULT_DIR, f"result_RAG_{filename}")
        
        # 載入題目 (限制前 500 題)
        with open(sentence_path, 'r', encoding='utf-8') as f:
            questions = [json.loads(line) for line in f][:500]
        
        # 載入標準答案 (限制前 500 題)
        with open(answer_path, 'r', encoding='utf-8') as f:
            answers = [json.loads(line) for line in f][:500]
        
        if len(questions) != len(answers):
            print(f"警告: {filename} 的題目數與答案數不匹配，跳過此檔案。")
            continue

        total_correct = 0
        all_results = []
        
        # 遍歷所有題目
        for q_data, a_data in tqdm(zip(questions, answers), total=len(questions), desc=f"評估 {filename}"):
            question = q_data['question']
            options = q_data['options']
            true_answer = a_data['answer']
            
            # 使用 RAG 模型推論
            predicted_answer, context = generate_rag_response(question, options)
            
            is_correct = (predicted_answer == true_answer)
            if is_correct:
                total_correct += 1
            
            # 儲存詳細結果
            result_entry = {
                "id": q_data['id'],
                "question": question,
                "options": options,
                "true_answer": true_answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "rag_context": context # RAG 版本儲存上下文
            }
            all_results.append(result_entry)

        # 寫入結果檔案
        with open(result_path, 'w', encoding='utf-8') as f:
            for entry in all_results:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        # 輸出總結
        total_questions = len(questions)
        accuracy = (total_correct / total_questions) * 100 if total_questions > 0 else 0
        print(f"--- {filename} 評估總結 (RAG BianCang) ---")
        print(f"總題數: {total_questions}")
        print(f"答對題數: {total_correct}")
        print(f"準確率: {accuracy:.2f}%")
        print(f"詳細結果已儲存至: {result_path}")
        print("----------------------------")

if __name__ == "__main__":
    run_evaluation()