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

# 評分維度與刻度（1~100）
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
    "llama3:8b-instruct-q4_K_M (rag)": (
        "根據上下文，山羊缺乏維生素A可能會導致夜盲症（Night blindness），即山羊在黑暗中無法正常地發揮視力。"
    ),
    "deepseek-r1:8b-llama-distill-q4_K_M (rag)": (
        "山羊缺乏维生素A會導致視力出現嚴重問題。維生素A對眼睛功能至關重要，尤其是光敏細胞的正常運作。如果缺乏，可能會引發夜盲症或其他視力障礙，嚴重情況下甚至可能導致失明。"
    ),
    "mistral:7b-instruct-v0.3-q4_K_M (RAG)": "在山羊缺乏維生素A的情況下，可能會發生視力不明、夜間视力低下、虛白眼、水腫眼等問題。",

    "gemma3:12b-it-q4_K_M  (rag)": (
        "山羊缺乏維生素A會導致夜盲症，也就是說在夜晚看不清。"
    ),
    "goat_llama3_8bit_q4_K_M_0819_1553:latest": (
        "缺乏維生素A,將導致山羊的眼睛發生夜盲症(夜間視力不良),尤其是在夜晚或低光照下。最嚴重者可能會完全失去視力,並且容易引起其他疾病。"
    ),
    "llama3:8b-instruct-q4_K_M": (
        "山羊缺乏維生素A可能會導致其視力出現下列問題：1.失明症（Night blindness）：維生素A對夜間視力非常重要，如果缺乏，山羊可能會發生失明症，無法在黑暗中看到。2.蜜網膜病變（Bitot’s spots）：維生素A缺乏可以導致眼球表面的白色或黃色斑點出現，這些斑點可能會損害眼睛的視力。3.角膜炎（Corneal ulcers）：維生素A缺乏可能會增加角膜炎的風險，導致角膜的傷害和感染。因此，如果山羊缺乏維生素A，需要進行適當的治療和營養改善，以避免眼部疾病的發生。"
    ),
    "gpt-oss:20b(rag)": (
        "缺乏維生素A主要導致夜盲、角膜乾燥、角膜炎、結膜炎、角膜硬化及視網膜退化等視覺障礙，最終可能引致視力喪失。及時補充維生素 A，並維持適當的營養攝取，可有效預防這些問題。"
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

MULTI_MODEL_SYSTEM_PROMPT = f"""
請你扮演一位專業的山羊飼養與照護專家，對比所有 AI 模型的回答，判斷哪一個整體更好。

你將會看到：
- 使用者問題（User Question）
- 參考答案（Reference Answer）
- 多個模型的回答 (Model Answers)

請逐一比較 **所有模型回答** 與參考答案的差異，
必須針對每一個模型回答，指出其正確性、完整性、優缺點，不能跳過任何模型。

最後必須從完整模型清單中選出優勝者（winner）。

嚴格輸出格式要求：只輸出一段 JSON，格式如下：
{{
  "comparison": {{
    "模型A": "優缺點說明",
    "模型B": "優缺點說明",
    "模型C": "優缺點說明",
    ...
  }},
  "winner": ["模型名稱1"]
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

MULTI_MODEL_SYSTEM_WITH_STEPS_TEMPLATE = f"""
請你扮演一位專業的山羊飼養與照護專家，對比所有 AI 模型的回答，判斷哪一個整體更好。
請嚴格依循以下評判步驟（evaluation_steps）：
{{steps_block}}

綜合考量多個指標（{', '.join(CRITERIA)}），以事實正確性與安全性為優先。
輸出只允許 JSON，格式如下：
{{
  "comparison": {{
    "模型A": "優缺點說明",
    "模型B": "優缺點說明",
    "模型C": "優缺點說明",
    ...
  }},
  "winner": ["模型名稱1"]
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

MULTI_MODEL_USER_PROMPT = """
--
User Question:
{user_q}

Reference Answer:
{ref_a}

Model Answers:
{model_answers}
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


def build_multi_model_system_prompt(steps: Optional[List[str]]) -> str:
    if steps:
        steps_block = json.dumps(steps, ensure_ascii=False, indent=2)
        return MULTI_MODEL_SYSTEM_WITH_STEPS_TEMPLATE.replace("{steps_block}", steps_block)
    else:
        return MULTI_MODEL_SYSTEM_PROMPT

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


def judge_multi_model(system_prompt: str, user_q: str, ref_a: str, model_answers: Dict[str, str]) -> Dict[str, Any]:
    answers_formatted = "\n".join([f"{mname}:\n{ans}" for mname, ans in model_answers.items()])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": MULTI_MODEL_USER_PROMPT.format(user_q=user_q, ref_a=ref_a, model_answers=answers_formatted)},
    ]
    raw = call_ollama(messages)
    parsed = safe_json_loads(raw)
    if not parsed:
        raise ValueError(f"Multi-model judge output invalid JSON: {raw}")

    # 驗證 winner
    if "winner" not in parsed or not isinstance(parsed["winner"], list):
        print("[Warning] LLM 輸出不合法，未找到 winner list。")
        parsed["winner"] = []
    else:
        parsed["winner"] = [w for w in parsed["winner"] if w in model_answers.keys()]

    return parsed

# ===================== 入口函式 =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-details", action="store_true", help="是否在終端機也印出完整 JSON")
    parser.add_argument("--multi", action="store_true", help="是否執行 multi-model winner 模式")
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

        if args.multi:
            system_prompt = build_multi_model_system_prompt(steps)
            out = judge_multi_model(system_prompt, USER_QUESTION, REFERENCE_ANSWER, MODEL_ANSWERS)
            detailed_results["multi_model"] = out
            results_line.append(f"Multi-model Winner: {out.get('winner')}")
        else:
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
    with open("geval_results.csv", "w", newline="", encoding="utf-8-sig") as f:
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

            # multi-model 模式
            if "multi_model" in results:
                winner = results["multi_model"].get("winner", "")
                writer.writerow([run_idx, "multi_model", "", "", "", "", "", "", "", "", "", "", winner])
            else:
                # 單模型模式
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