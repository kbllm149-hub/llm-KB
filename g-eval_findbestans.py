"""
Ollama + 本地 LLM（例如 gpt-oss-20b）跑 G-Eval 的單檔腳本（支援 Auto-CoT）。
- 支援：單模型評分（七維 1–100 分）、multi-model winner 比較、Auto-CoT（先自生評判步驟，再評分）。
- 輸入在程式最上方區塊設定或用參數控制。
- 評分維度：helpfulness, conciseness, correctness, relevance, clarity, completeness, faithfulness（各 1–100 分）。
- 輸出：
  1) 終端：單模型模式 → `[模型A] Rating: [[88]] ...`；multi-model 模式 → `Multi-model Winner: 模型A`
  2) 檔案：詳細評分/比較寫入 `geval_results.json`
  3) Auto-CoT 啟用時：自動產生 `evaluation_steps.json`（評判步驟）

需求：
  pip install requests
且本機已啟動 Ollama 服務，並已建立/拉取好對應模型（MODEL_NAME）。
"""

import os
import json
import time
import argparse
import requests
from typing import Dict, Any, List, Optional
import csv

# ===================== 可調整區 =====================
# Ollama 連線與模型設定
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.environ.get("GEVAL_MODEL", "gpt-oss:20b")  # 本地評審模型名
TIMEOUT_SEC = 300
TEMPERATURE = 0.2 # 評分時的溫度，數值越低越保守，越高越有創意(隨機性)

# 評分維度(crteria)與刻度（1~100）
CRITERIA = [
    "helpfulness",
    "conciseness",
    "correctness",
    "relevance",
    "clarity",
    "completeness",
    "faithfulness",
]
SCALE_MIN, SCALE_MAX = 1, 100

# === 測試用輸入（請自行修改） ===
USER_QUESTION = "山羊缺乏維生素A會導致視力出現什麼問題？繁體中文回答"
REFERENCE_ANSWER = (
    "會造成夜盲症，嚴重時導致失明。"
)

# 模型回答集合（名稱 -> 回答）。可任意增減。
MODEL_ANSWERS: Dict[str, str] = {
    "llama3:8b-instruct-q4_K_M (rag)ans1": (
        "根據上下文，山羊缺乏維生素A可能會導致夜盲症（Night blindness），即山羊在黑暗中無法正常地發揮視力。"
    ),
    "llama3:8b-instruct-q4_K_M (rag)ans2": (
        "山羊缺乏维生素A會導致視力出現嚴重問題。維生素A對眼睛功能至關重要，尤其是光敏細胞的正常運作。如果缺乏，可能會引發夜盲症或其他視力障礙，嚴重情況下甚至可能導致失明。"
    ),
    "llama3:8b-instruct-q4_K_M (rag)ans3": (
        "在山羊缺乏維生素A的情況下，可能會發生視力不明、夜間视力低下、虛白眼、水腫眼等問題。"
    ),
    "llama3:8b-instruct-q4_K_M (rag)ans4": (
        "山羊缺乏維生素A會導致夜盲症，也就是說在夜晚看不清。"
    ),
}
# ==================================================

# ---------- 基礎 Prompts（無 Auto-CoT 時使用） ----------
BASE_SYSTEM_PROMPT = f"""
請你扮演一位專業的山羊飼養與照護專家，並以嚴謹、公正的角度，評估 AI 模型針對使用者問題所提供的回答品質。

評估的依據包含（每項 {SCALE_MIN}~{SCALE_MAX} 分，{SCALE_MAX} 為最佳）：
- helpfulness：是否能實際幫助使用者解決問題，給出可操作建議
- conciseness：是否言簡意賅、避免冗贅
- correctness：是否符合正確的山羊飼養知識
- relevance：內容是否緊扣使用者問題
- clarity：表述是否清楚、易於理解
- completeness：是否涵蓋關鍵要點、無重大遺漏
- faithfulness：是否忠實於參考答案/事實，無臆測誤導

請先比較「AI 模型回答」與「參考答案」，指出差異與錯誤/不足，再說明其是否正確且有幫助。

嚴格輸出格式要求：只輸出一段 JSON（不可添加多餘文字）：
{{
  "per_criterion": {{
    "helpfulness": <int {SCALE_MIN}-{SCALE_MAX}>,
    "conciseness": <int {SCALE_MIN}-{SCALE_MAX}>,
    "correctness": <int {SCALE_MIN}-{SCALE_MAX}>,
    "relevance": <int {SCALE_MIN}-{SCALE_MAX}>,
    "clarity": <int {SCALE_MIN}-{SCALE_MAX}>,
    "completeness": <int {SCALE_MIN}-{SCALE_MAX}>,
    "faithfulness": <int {SCALE_MIN}-{SCALE_MAX}>
  }},
  "differences": "指出與參考答案的差異，並標註錯誤或不足",
  "assessment": "說明其是否正確且有幫助（中文）",
  "overall_rating": <int {SCALE_MIN}-{SCALE_MAX}>
}}
"""


# ---------- Auto-CoT：生成評判步驟 ----------
STEPS_SYSTEM_PROMPT = f"""
你是一位專業的山羊飼養與照護專家。請你根據以下評估任務與評判標準，自行產生一份「評判步驟（evaluation_steps）」：
- 任務：評估 AI 模型對山羊照護問題的回答品質
- 評分維度：{', '.join(CRITERIA)}（每項 {SCALE_MIN}~{SCALE_MAX} 分）
- 目標：可重複、可操作、易審計，避免主觀臆測，優先驗證事實與安全。

輸出格式（只輸出 JSON）：
{{
  "evaluation_steps": [
    "第一步 ……",
    "第二步 ……",
    "……"
  ]
}}
"""

STEPS_USER_PROMPT = "產生一組審查流程，專注於：與參考答案一致性、事實/安全檢核、關鍵要點覆蓋、邏輯與可操作性、語言清晰度與簡潔性。"

# 評審階段（有 Auto-CoT 時會把 steps 內嵌到系統提示）
EVAL_SYSTEM_PROMPT_WITH_STEPS_TEMPLATE = f"""
請你扮演一位專業的山羊飼養與照護專家，並以嚴謹、公正的角度，評估 AI 模型針對使用者問題所提供的回答品質。

請嚴格依循以下評判步驟（evaluation_steps）：
{{steps_block}}

評分規則：每項 {SCALE_MIN}~{SCALE_MAX} 分（{SCALE_MAX} 為最佳），維度：{', '.join(CRITERIA)}。
輸出只允許 JSON，格式同下：
{{
  "per_criterion": {{
    "helpfulness": <int {SCALE_MIN}-{SCALE_MAX}>,
    "conciseness": <int {SCALE_MIN}-{SCALE_MAX}>,
    "correctness": <int {SCALE_MIN}-{SCALE_MAX}>,
    "relevance": <int {SCALE_MIN}-{SCALE_MAX}>,
    "clarity": <int {SCALE_MIN}-{SCALE_MAX}>,
    "completeness": <int {SCALE_MIN}-{SCALE_MAX}>,
    "faithfulness": <int {SCALE_MIN}-{SCALE_MAX}>
  }},
  "differences": "指出與參考答案的差異，並標註錯誤或不足",
  "assessment": "說明其是否正確且有幫助（中文）",
  "overall_rating": <int {SCALE_MIN}-{SCALE_MAX}>
}}
"""


# ---------- 通用資料模板 ----------
USER_PROMPT_TEMPLATE = """
--
User Question:
{user_q}

Reference Answer:
{ref_a}

Model Answer:
{model_a}
"""


HEADERS = {"Content-Type": "application/json"}

# ===================== 公用函式 =====================

def call_ollama(messages: List[Dict[str, str]], temperature: float = TEMPERATURE, max_retries: int = 3) -> str:
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "options": {"temperature": temperature},
        "stream": False,
    }
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(url, headers=HEADERS, data=json.dumps(payload), timeout=TIMEOUT_SEC)
            r.raise_for_status()
            data = r.json()
            return data["message"]["content"]
        except Exception as e:
            last_err = e
            time.sleep(1.2 * attempt)
    raise RuntimeError(f"Ollama call failed after {max_retries} retries: {last_err}")


def safe_json_loads(raw: str) -> Optional[dict]:
    try:
        return json.loads(raw)
    except Exception:
        import re
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def validate_overall_and_criteria(output: dict, model_name: str) -> int:
    # overall_rating 檢查
    overall = output.get("overall_rating")
    try:
        overall = int(overall)
    except Exception:
        raise ValueError(f"overall_rating 非整數: {model_name} -> {overall}")
    if not (SCALE_MIN <= overall <= SCALE_MAX):
        raise ValueError(f"overall_rating 超出範圍({SCALE_MIN}~{SCALE_MAX}): {model_name} -> {overall}")

    # 各維度分數檢查
    per = output.get("per_criterion", {})
    for c in CRITERIA:
        if c not in per:
            raise ValueError(f"缺少維度分數 {c}: {model_name}")
        try:
            v = int(per[c])
        except Exception:
            raise ValueError(f"{c} 分數非整數: {model_name} -> {per[c]}")
        if not (SCALE_MIN <= v <= SCALE_MAX):
            raise ValueError(f"{c} 分數超出範圍({SCALE_MIN}~{SCALE_MAX}): {model_name} -> {v}")
    return overall

# ===================== Auto-CoT 相關 =====================

def generate_evaluation_steps(steps_path: str) -> List[str]:
    messages = [
        {"role": "system", "content": STEPS_SYSTEM_PROMPT},
        {"role": "user", "content": STEPS_USER_PROMPT},
    ]
    raw = call_ollama(messages)
    parsed = safe_json_loads(raw)
    if not parsed or "evaluation_steps" not in parsed or not isinstance(parsed["evaluation_steps"], list):
        raise ValueError(f"Auto-CoT 生成評判步驟失敗：{raw}")
    steps = parsed["evaluation_steps"]
    with open(steps_path, "w", encoding="utf-8") as f:
        json.dump({"evaluation_steps": steps}, f, ensure_ascii=False, indent=2)
    return steps


def build_eval_system_prompt(steps: Optional[List[str]]) -> str:
    if steps:
        steps_block = json.dumps(steps, ensure_ascii=False, indent=2)
        return EVAL_SYSTEM_PROMPT_WITH_STEPS_TEMPLATE.replace("{steps_block}", steps_block)
    else:
        return BASE_SYSTEM_PROMPT


# ===================== 評審函式 =====================

def judge_one(system_prompt: str, model_name: str, user_q: str, ref_a: str, model_a: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(user_q=user_q, ref_a=ref_a, model_a=model_a)},
    ]
    raw = call_ollama(messages)
    parsed = safe_json_loads(raw)
    if not parsed:
        raise ValueError(f"Judge output is not valid JSON for {model_name}: {raw}")
    return parsed

# ===================== 入口函式 =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-details", action="store_true", help="是否在終端機也印出完整 JSON")
    parser.add_argument("--autocot", action="store_true", help="啟用 Auto-CoT：先生成評判步驟再評分")
    parser.add_argument("--steps-out", default="evaluation_steps.json", help="Auto-CoT 產生的步驟輸出路徑")
    parser.add_argument("--repeat", type=int, default=1, help="重複執行次數")
    args = parser.parse_args()

    all_runs: List[Dict[str, Any]] = []

    for run_idx in range(1, args.repeat + 1):
        print(f"\n===== Run {run_idx} / {args.repeat} =====")

        steps: Optional[List[str]] = None
        if args.autocot:
            steps = generate_evaluation_steps(args.steps_out)

        detailed_results: Dict[str, Any] = {}
        results_line: List[str] = []
        system_prompt = build_eval_system_prompt(steps)
        for mname, manswer in MODEL_ANSWERS.items():
            out = judge_one(system_prompt, mname, USER_QUESTION, REFERENCE_ANSWER, manswer)
            overall = validate_overall_and_criteria(out, mname)
            detailed_results[mname] = out
            results_line.append(f"[{mname}] Rating: [[{overall}]]")

        print(" ".join(results_line))

        all_runs.append({
            "run": run_idx,
            "results": detailed_results
        })

        if args.show_details:
            print(json.dumps(detailed_results, ensure_ascii=False, indent=2))

    # 存成列表，包含多次的結果
    with open("geval_best_ans.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        # 先寫標題列
        writer.writerow([
            "run",
            "model",
            "helpfulness",
            "conciseness",
            "correctness",
            "relevance",
            "clarity",
            "completeness",
            "faithfulness",
            "overall_rating",
            "differences",
            "assessment",
            "winner"
        ])

        for run in all_runs:
            run_idx = run["run"]
            results = run["results"]


            for mname, out in results.items():
                per = out.get("per_criterion", {})
                writer.writerow([
                    run_idx,
                    mname,
                    per.get("helpfulness", ""),
                    per.get("conciseness", ""),
                    per.get("correctness", ""),
                    per.get("relevance", ""),
                    per.get("clarity", ""),
                    per.get("completeness", ""),
                    per.get("faithfulness", ""),
                    out.get("overall_rating", ""),
                    out.get("differences", "").replace("\n", " "),
                    out.get("assessment", "").replace("\n", " "),
                    ""
                ])

if __name__ == "__main__":
    main()
'''
新增參數：--autocot
會先呼叫一次評審模型產生 evaluation_steps.json（評判步驟），再把步驟自動嵌入系統提示進行評分／比較。

新增參數：--steps-out
可自訂步驟輸出檔名（預設 evaluation_steps.json）。

評分刻度改為 1–100 分，包含各維與總分的嚴格檢查。

--multi 仍可一次比較所有模型並輸出 Multi-model Winner: 模型X。

詳細結果照舊寫入 geval_results.json。

範例執行：

單模型評分（Auto-CoT）：
python your_script.py --autocot

多模型勝者（Auto-CoT）：
python your_script.py --multi --autocot

顯示細節：
python your_script.py --show-details
多次執行（例如 5 次）：
python your_script.py --multi --autocot --repeat 5

'''