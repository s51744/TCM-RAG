# cot_pipeline_4bit_debug.py
# 完整可執行：4-bit (bitsandbytes) 推理 + CoT 解析 + robust fallback + test_limit
# 儲存後以 python cot_pipeline_4bit_debug.py 執行
# 注意：請先安裝 transformers, torch, bitsandbytes (如要 4-bit)

import os
import re
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 嘗試導入 BitsAndBytesConfig（若沒有，會提示使用者）
try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except Exception:
    BNB_AVAILABLE = False

# ---------- 參數（可用 CLI 覆寫） ----------
DEFAULT_MODEL = "QLU-NLP/BianCang-Qwen2-7B-Instruct"
INPUT_FILE = "./llm_exam/split_data/cot_examples_A.jsonl"
OUT_CORRECT = "./llm_exam/split_data/validated_cot_pool_CORRECT.jsonl"
OUT_INCORRECT = "./llm_exam/split_data/validated_cot_pool_INCORRECT.jsonl"
OUT_DEBUG = "./llm_exam/split_data/debug_fallbacks.jsonl"

# 生成參數
MAX_NEW_TOKENS = 256
DO_SAMPLE = False
TEMPERATURE = 0.0

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------- 載入模型（4-bit） ----------
def load_model_4bit(model_name: str):
    if not BNB_AVAILABLE:
        raise RuntimeError("找不到 BitsAndBytesConfig / bitsandbytes。若要使用 4-bit，請安裝 bitsandbytes 並使用相容版本的 transformers。")

    logger.info(f"載入模型（4-bit）: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # 若 pad_token 未設定或與 eos 相同，設定 pad_token 為 eos
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token

    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=qconfig,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model

# 若無法使用 4-bit，可降級到 fp16（但會用較多 VRAM）
def load_model_fp16(model_name: str):
    logger.info(f"載入模型（fp16）: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    return tokenizer, model

# ---------- Prompt Template（固定格式） ----------
PROMPT_TEMPLATE = """題目：
{question}

選項：
A. {A}
B. {B}
C. {C}
D. {D}
E. {E}

請先逐步推理並寫出你的思考鏈（CoT），最後以「最終答案: X」或「最終答案：X」格式輸出選項字母（例如：最終答案: A）。請僅在最末行輸出最終答案字母，不要在推理過程中重複單獨字母作為答案。
"""

# ---------- 生成（只 decode 新 tokens） ----------
def run_inference_new_tokens(tokenizer, model, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(model.device)

    # 明確建立 attention_mask（避免警告）
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = (inputs["input_ids"] != tokenizer.pad_token_id).long()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

    # 只 decode 新產生 tokens（避免包含 prompt）
    gen_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    return generated_text

# ---------- Robust CoT 與答案解析器 ----------
def parse_cot_and_answer(full_generated: str, options: Dict[str, str]) -> Tuple[str, Optional[str], str]:
    """
    回傳 (reasoning, final_answer_or_None, parse_method)
    parse_method 標示使用哪種解析方式（'marker_regex','marker_near_regex','exact_option_match','partial_option_match','last_letter_fallback','none'）
    """
    text = full_generated.strip()
    # 1) 標準標記：最優先從「最終答案」標記後直接捕獲 A-E
    m = re.search(r"(?:最終答案|最终答案|Final Answer|final answer|答案)[\s:：\-]*([A-E])", text, flags=re.I)
    if m:
        ans = m.group(1).upper()
        reasoning = text[:m.start()].strip()
        return reasoning, ans, "marker_regex"

    # 2) 次要模式：比如「因此答案是B」「故答案為 D」等靠近 marker 的捕捉
    m2 = re.search(r"(?:最終答案|最终答案|final answer|答案).{0,40}([A-E])", text, flags=re.I|re.S)
    if m2:
        ans = m2.group(1).upper()
        reasoning = text[:m2.start()].strip()
        return reasoning, ans, "marker_near_regex"

    # 3) 若模型直接輸出某選項的完整文本（完全相同） -> exact match
    normalized = re.sub(r"\s+", "", text)  # 去掉空白與換行
    for key, val in options.items():
        if normalized == re.sub(r"\s+", "", str(val)):
            return "", key, "exact_option_match"

    # 4) 部分匹配：模型輸出包含某選項內容（例如只輸出「化湿，解暑」）
    for key, val in options.items():
        if str(val) in text:
            return "", key, "partial_option_match"

    # 5) 最後手段：全文找最後一個出現的 A~E（fallback）
    found = re.findall(r"\b([A-E])\b", text.upper())
    if found:
        ans = found[-1]
        # reasoning：到最後一個字母前
        last_pos = text.upper().rfind(ans)
        reasoning = text[:last_pos].strip()
        return reasoning, ans, "last_letter_fallback"

    # 6) 完全找不到
    return text, None, "none"

# ---------- 儲存 JSONL 輔助 ----------
def save_jsonl(path: str, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------- 主流程 ----------
def main(args):
    # 選擇載入方式
    try:
        if args.use_4bit:
            tokenizer, model = load_model_4bit(args.model_name)
        else:
            tokenizer, model = load_model_fp16(args.model_name)
    except Exception as e:
        logger.error(f"載入模型失敗：{e}")
        return

    # 讀入資料
    if not os.path.exists(args.input_file):
        logger.error(f"找不到輸入檔案：{args.input_file}")
        return

    with open(args.input_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    if args.test_limit is not None:
        lines = lines[: args.test_limit]

    total = len(lines)
    logger.info(f"準備處理 {total} 題（test_limit={args.test_limit}）")

    correct_list = []
    incorrect_list = []
    debug_list = []

    for idx, item in enumerate(lines, start=1):
        qid = item.get("id")
        question = item.get("question", "")
        options = item.get("options", {})
        gt = item.get("answer")

        # 建 prompt
        try:
            p = PROMPT_TEMPLATE.format(
                question=question,
                A=options.get("A", ""),
                B=options.get("B", ""),
                C=options.get("C", ""),
                D=options.get("D", ""),
                E=options.get("E", ""),
            )
        except Exception as e:
            logger.exception(f"構建 prompt 失敗 for id {qid}: {e}")
            continue

        # 生成
        gen = run_inference_new_tokens(tokenizer, model, p, max_new_tokens=args.max_new_tokens)

        # 解析
        reasoning, pred, parse_method = parse_cot_and_answer(gen, options)

        is_correct = (pred == gt)

        record = {
            "id": qid,
            "question": question,
            "options": options,
            "prompt_used": p,
            "model_output": gen,
            "reasoning": reasoning,
            "final_answer": pred,
            "answer": gt,
            "correct": is_correct,
            "parse_method": parse_method,
        }

        if is_correct:
            correct_list.append(record)
        else:
            incorrect_list.append(record)

        # debug 紀錄：若 parse_method 為 fallback 或 none，存入 debug list
        if parse_method in ("partial_option_match", "last_letter_fallback", "none"):
            debug_list.append(record)

        logger.info(f"[{idx}/{total}] ID {qid} -> pred: {pred} | gt: {gt} | ok: {is_correct} | parse: {parse_method}")

    # 儲存三個檔案
    save_jsonl(args.out_correct, correct_list)
    save_jsonl(args.out_incorrect, incorrect_list)
    save_jsonl(args.out_debug, debug_list)

    logger.info(f"完成：總 {total} 題，正確 {len(correct_list)} 題，錯誤 {len(incorrect_list)} 題")
    logger.info(f"合格檔案：{args.out_correct}")
    logger.info(f"不合格檔案：{args.out_incorrect}")
    logger.info(f"除錯檔案（fallback 及疑難）：{args.out_debug}")

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoT pipeline (4-bit) with robust parsing")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL, help="模型名稱或本地路徑")
    parser.add_argument("--input-file", type=str, default=INPUT_FILE, help="輸入 JSONL 檔案")
    parser.add_argument("--out-correct", type=str, default=OUT_CORRECT, help="輸出：正確題目 JSONL")
    parser.add_argument("--out-incorrect", type=str, default=OUT_INCORRECT, help="輸出：錯誤題目 JSONL")
    parser.add_argument("--out-debug", type=str, default=OUT_DEBUG, help="輸出：debug_fallbacks.jsonl")
    parser.add_argument("--test-limit", type=int, default=100, help="先跑前 N 題（設 None 則跑完整檔案）")
    parser.add_argument("--use-4bit", action="store_true", help="使用 bitsandbytes 4-bit 量化載入（需安裝）")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS, help="生成 token 數量上限")
    args = parser.parse_args()

    # 將 CLI 參數注入 main
    # 調整 args 屬性名稱對應 main
    class _A: pass
    a = _A()
    a.model_name = args.model_name
    a.input_file = args.input_file
    a.out_correct = args.out_correct
    a.out_incorrect = args.out_incorrect
    a.out_debug = args.out_debug
    a.test_limit = args.test_limit
    a.max_new_tokens = args.max_new_tokens
    a.use_4bit = args.use_4bit

    main(a)
