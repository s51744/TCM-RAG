import torch
import json
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# --- 參數設定 ---
# 移除了 VECTOR_DB_DIR 和 Embeddings 相關參數
SENTENCE_DIR = "./llm_exam/sentence"
ANSWER_DIR = "./llm_exam/answer"
RESULT_DIR = "./llm_result"  # 更改結果輸出目錄，以區分 RAG 版本
MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite"
MAX_TOKEN = 3
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# --- 載入模型 ---
print("--- 載入模型 ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,  # 保持 float16 以節省空間
    device_map="auto",    # <--- 關鍵修改：使用 auto 進行 CPU 卸載
    trust_remote_code=True
)
model.eval()

# --- 基礎模型推論函式 (不使用 RAG) ---
def generate_base_response(question, options):
    """執行基礎 LLM 推論，不進行 RAG 檢索"""
    if isinstance(options, dict):
        option_str = ", ".join([f"{k}: {v}" for k, v in options.items()])
    elif isinstance(options, list):
        option_str = ", ".join([f"{chr(65 + i)}: {option}" for i, option in enumerate(options)])
    else:
        option_str = str(options) # Fallback for unexpected types
    
    # 構建 Prompt：直接問答，不加入任何外部知識庫上下文
    prompt = (
        f"請從下列選項中，以繁體中文選出最正確的選項代號（A, B, C, D, E）。"
        f"請只回答選項代號，不需解釋。\n"
        f"問題：{question}\n"
        f"選項：{option_str}\n"
        f"答案："
    )

    # 1. 模型生成
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKEN,
            do_sample=False,  # 評估時建議使用確定性的 Greedy Search
            use_cache=False,  #KV 快取關閉
            eos_token_id=tokenizer.eos_token_id 
        )
    
    # 2. 解碼與後處理
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
    
    # 提取第一個大寫字母作為答案
    match = re.search(r'[A-E]', response)
    predicted_answer = match.group(0) if match else response
    
    # 不使用 RAG，所以 context 為空字串
    context = "" 
    
    return predicted_answer, context

# --- 主評估流程 ---
def run_evaluation():
    """執行所有檔案的評估"""
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # 獲取所有題目檔案名稱 (TCM_ED_A.jsonl, TCM_ED_B.jsonl)
    exam_files = [f for f in os.listdir(SENTENCE_DIR) if f.endswith(".jsonl")]

    for filename in exam_files:
        print(f"\n--- 開始評估檔案: {filename} ---")
        sentence_path = os.path.join(SENTENCE_DIR, filename)
        answer_path = os.path.join(ANSWER_DIR, filename)
        result_path = os.path.join(RESULT_DIR, f"result_base_{filename}") # 更改輸出檔名
        
        # 載入題目
        with open(sentence_path, 'r', encoding='utf-8') as f:
            questions = [json.loads(line) for line in f][:500]
        
        # 載入標準答案
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
            
            # 使用基礎模型推論
            predicted_answer, context = generate_base_response(question, options)
            
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
                "rag_context": "N/A (Base Model Evaluation)" # 標記為 N/A
            }
            all_results.append(result_entry)

        # 寫入結果檔案
        with open(result_path, 'w', encoding='utf-8') as f:
            for entry in all_results:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        # 輸出總結
        total_questions = len(questions)
        accuracy = (total_correct / total_questions) * 100 if total_questions > 0 else 0
        print(f"--- {filename} 評估總結 (Base Model) ---")
        print(f"總題數: {total_questions}")
        print(f"答對題數: {total_correct}")
        print(f"準確率: {accuracy:.2f}%")
        print(f"詳細結果已儲存至: {result_path}")
        print("----------------------------")

if __name__ == "__main__":
    run_evaluation()