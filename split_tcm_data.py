import json
import random
import os

# --- 參數設定 ---
# 題目和答案的路徑必須分開載入
SENTENCE_PATH = "./llm_exam/sentence/1.TCM_ED_A.jsonl" 
ANSWER_PATH = "./llm_exam/answer/1.TCM_ED_A.jsonl" 
OUTPUT_DIR = "./llm_exam/split_data"

EXAMPLE_SIZE = 100  # CoT 範例集 (Eyes-On)
TEST_SIZE = 500     # 最終測試集 (Eyes-Off)

# --- 主流程 ---

os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    # 1. 載入題目（只有問題和選項）
    questions_only = {}
    with open(SENTENCE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            questions_only[item['id']] = item
            
    # 2. 載入完整答案（題目 + 答案）
    answers_only = {}
    with open(ANSWER_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            answers_only[item['id']] = item['answer'] # 只提取答案
            
except FileNotFoundError as e:
    print(f"錯誤：找不到檔案 {e.filename}。請確認路徑是否正確。")
    exit()

# 3. 合併題目和答案
all_complete_questions = []
for q_id, q_data in questions_only.items():
    if q_id in answers_only:
        # 將答案合併到題目字典中
        q_data['answer'] = answers_only[q_id] 
        all_complete_questions.append(q_data)
    
total_size = len(all_complete_questions)
print(f"成功合併題目與答案，總題數：{total_size}")

if total_size < (EXAMPLE_SIZE + TEST_SIZE):
    print(f"警告：檔案中只有 {total_size} 題，不足以劃分 {EXAMPLE_SIZE + TEST_SIZE} 題。")

# 4. 打亂題目順序 (關鍵步驟)
random.seed(1) 
random.shuffle(all_complete_questions)

# 5. 劃分數據集
cot_examples = all_complete_questions[:EXAMPLE_SIZE]
final_test_set = all_complete_questions[EXAMPLE_SIZE : EXAMPLE_SIZE + TEST_SIZE]
remaining_questions = all_complete_questions[EXAMPLE_SIZE + TEST_SIZE :]

# 6. 儲存結果
def save_data(data, filename, remove_answer=False):
    """
    保存數據到檔案，可選地移除 'answer' 字段。
    """
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            item_to_save = item.copy() # 複製字典以避免修改原始列表
            if remove_answer and 'answer' in item_to_save:
                del item_to_save['answer'] # 移除答案
                
            f.write(json.dumps(item_to_save, ensure_ascii=False) + '\n')
    print(f"✅ 已儲存 {len(data)} 題至 {filepath}")

# 範例集 (CoT 庫來源) - 必須保留答案
save_data(cot_examples, "cot_examples_A.jsonl", remove_answer=False)

# 最終測試集 (Eyes-Off) - 必須移除答案
save_data(final_test_set, "final_test_set_A.jsonl", remove_answer=True) 

# 剩餘題目 - 移除答案
save_data(remaining_questions, "remaining_questions_A.jsonl", remove_answer=True)