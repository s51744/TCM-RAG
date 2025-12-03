import torch
import json
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# --- 參數設定 ---
INPUT_FILE = "./llm_exam/split_data/cot_examples_A.jsonl"
OUTPUT_FILE = "./llm_exam/split_data/validated_cot_pool.jsonl" # 儲存最終驗證合格的 CoT 範例
MODEL_NAME = "QLU-NLP/BianCang-Qwen2-7B-Instruct"
MAX_NEW_TOKENS = 512  # CoT 推理過程需要較長的輸出長度
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# --- CoT 生成 Prompt 模板 ---
# 參考 Medprompt 論文的 CoT 模板，並針對中醫調整
# 我們要求模型輸出推理過程和最終答案，以便進行驗證
COT_GEN_TEMPLATE = (
    "你是一位專業的中醫醫生。請根據提供的問題和選項，逐步進行中醫辨證推理，並在最後一行給出最終的選項代號。\n"
    "## 問題: {question}\n"
    "## 選項: {options}\n"
    "## 推理過程與答案\n"
    "思維鏈 (Chain-of-Thought):" # 模型將從這裡開始推理
)

# --- 載入模型 ---
print("--- 載入 BianCang LLM ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if DEVICE != "cpu" else torch.float32,
    device_map="auto",
)
model.eval()

# --- 函數：生成 CoT 並提取答案 ---
def generate_and_extract_cot(item):
    question = item['question']
    answer_key = item['answer']
    options = item['options']
    
    # 格式化選項，方便模型讀取
    option_str = ", ".join([f"{k}: {v}" for k, v in options.items()])
    
    # 構建用於生成的完整 Prompt
    prompt = COT_GEN_TEMPLATE.format(question=question, options=option_str)

    # 1. 模型生成
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, 
            use_cache=False,
            eos_token_id=tokenizer.eos_token_id 
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
    
    # 2. 提取生成的答案（在輸出文本的最後尋找 [A-E]）
    # 嘗試尋找 '最終答案是：[A]' 或 '答案是 A' 這樣的模式
    final_answer_match = re.search(r'[是为是]\s*[A-E]\s*$', response.replace('\n', ' '))
    predicted_answer = final_answer_match.group(0)[-1].strip() if final_answer_match else None
    
    # 3. 驗證與過濾
    is_correct = (predicted_answer == answer_key)
    
    if is_correct:
        # 儲存完整的 CoT 範例，供 kNN 檢索使用
        item['cot_text'] = response
        item['predicted_answer'] = predicted_answer
        return item
    else:
        # 根據 Medprompt 原則，如果答案不正確，則不信任該推理鏈，直接捨棄
        return None

# --- 主執行區塊 ---
if __name__ == "__main__":
    print(f"--- 開始載入 {INPUT_FILE} ---")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            cot_examples = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"錯誤：找不到輸入檔案 {INPUT_FILE}。")
        exit()

    validated_cot_pool = []
    
    print(f"--- 開始生成與過濾 CoT 範例 ({len(cot_examples)} 題) ---")

    for item in tqdm(cot_examples, desc="生成 CoT 並驗證"):
        result = generate_and_extract_cot(item)
        if result:
            validated_cot_pool.append(result)
            
    # 儲存最終合格的 CoT 範例庫
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in validated_cot_pool:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n✅ CoT 範例生成與驗證完成！")
    print(f"總共生成並儲存了 {len(validated_cot_pool)} 個合格的 CoT 範例。")
    print(f"這些範例已儲存至 {OUTPUT_FILE}，可用於下一步的 HNSW 向量庫建構。")