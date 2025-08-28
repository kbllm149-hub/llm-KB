import os
from bert_score import score
from moverscore_v2 import get_idf_dict, word_mover_score
from transformers import AutoTokenizer, AutoModel

# ========= Sliding Window =========
def sliding_window_split(text, tokenizer, max_len, stride=128):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    segments = []
    start = 0
    while start < len(tokens):
        end = min(start + max_len, len(tokens))
        segment = tokens[start:end]
        segments.append(tokenizer.decode(segment, 
        skip_special_tokens=True))
        if end == len(tokens):
            break
        start += (max_len - stride)
    return segments

# ========= 主評估函式 =========
def run_evaluation(references, predictions, model_name="hfl/chinese-roberta-wwm-ext-large", stride=128, device="cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    max_len = model.config.max_position_embeddings

    # ======================
    # BERTScore
    # ======================
    print("=== BERTScore 評估中 ===")
    bert_scores = []
    for ref, pred in zip(references, predictions):
        ref_segments = sliding_window_split(ref, tokenizer, max_len, stride)
        pred_segments = sliding_window_split(pred, tokenizer, max_len, stride)
        min_len = min(len(ref_segments), len(pred_segments))
        ref_segments, pred_segments = ref_segments[:min_len], pred_segments[:min_len]

        P, R, F1 = score(
            pred_segments,
            ref_segments,
            model_type=model_name,
            num_layers=model.config.num_hidden_layers,
            verbose=False,
            device=device,
            lang="zh"
        )
        bert_scores.append((P.mean().item(), R.mean().item(), F1.mean().item()))

    bert_precision = sum(x[0] for x in bert_scores) / len(bert_scores)
    bert_recall = sum(x[1] for x in bert_scores) / len(bert_scores)
    bert_f1 = sum(x[2] for x in bert_scores) / len(bert_scores)

    print("\n=== BERTScore (平均) ===")
    print(f"Precision: {bert_precision:.4f}")
    print(f"Recall:    {bert_recall:.4f}")
    print(f"F1:        {bert_f1:.4f}")

    for i, (p, r, f) in enumerate(bert_scores):
        print(f"樣本 {i+1} -> P: {p:.4f}, R: {r:.4f}, F1: {f:.4f}")

    # ======================
    # MoverScore
    # ======================
    print("\n=== MoverScore 評估中 ===")
    os.environ['MOVERSCORE_MODEL'] = model_name
    mover_scores_all = []

    for ref, pred in zip(references, predictions):
        ref_segments = sliding_window_split(ref, tokenizer, max_len, stride)
        pred_segments = sliding_window_split(pred, tokenizer, max_len, stride)
        min_len = min(len(ref_segments), len(pred_segments))
        ref_segments, pred_segments = ref_segments[:min_len], pred_segments[:min_len]

        idf_ref = get_idf_dict(ref_segments)
        idf_pred = get_idf_dict(pred_segments)

        mover_scores = word_mover_score(
            ref_segments,
            pred_segments,
            idf_ref,
            idf_pred,
            stop_words=[],
            n_gram=1,
            remove_subwords=True
        )
        mover_scores_all.append(sum(mover_scores) / len(mover_scores))

    avg_mover = sum(mover_scores_all) / len(mover_scores_all)

    print("\n=== MoverScore ===")
    for i, s in enumerate(mover_scores_all):
        print(f"樣本 {i+1}: {s:.4f}")
    print(f"\n平均 MoverScore: {avg_mover:.4f}")


if __name__ == "__main__":
    # 測試資料
    references = [
        "會造成夜盲症，嚴重時導致失明。",
        "會造成夜盲症，嚴重時導致失明。",
        "會造成夜盲症，嚴重時導致失明。",
        "會造成夜盲症，嚴重時導致失明。",
        "會造成夜盲症，嚴重時導致失明。",
        "會造成夜盲症，嚴重時導致失明。",
        "會造成夜盲症，嚴重時導致失明。"
    ]

    predictions = [
        "根據上下文，山羊缺乏維生素A可能會導致夜盲症（Night blindness），即山羊在黑暗中無法正常地發揮視力。",
        "山羊缺乏维生素A會導致視力出現嚴重問題。維生素A對眼睛功能至關重要，尤其是光敏細胞的正常運作。如果缺乏，可能會引發夜盲症或其他視力障礙，嚴重情況下甚至可能導致失明。",
        "在山羊缺乏維生素A的情況下，可能會發生視力不明、夜間视力低下、虛白眼、水腫眼等問題。",
        "山羊缺乏維生素A會導致夜盲症，也就是說在夜晚看不清。",
        "缺乏維生素A,將導致山羊的眼睛發生夜盲症(夜間視力不良),尤其是在夜晚或低光照下。最嚴重者可能會完全失去視力,並且容易引起其他疾病。",
        "山羊缺乏維生素A可能會導致其視力出現下列問題：1.失明症（Night blindness）：維生素A對夜間視力非常重要，如果缺乏，山羊可能會發生失明症，無法在黑暗中看到。2.蜜網膜病變（Bitot’s spots）：維生素A缺乏可以導致眼球表面的白色或黃色斑點出現，這些斑點可能會損害眼睛的視力。3.角膜炎（Corneal ulcers）：維生素A缺乏可能會增加角膜炎的風險，導致角膜的傷害和感染。因此，如果山羊缺乏維生素A，需要進行適當的治療和營養改善，以避免眼部疾病的發生。",
        f"""缺乏維生素 A（視黃醇）會導致山羊出現多種視覺相關的問題，主要包括：

夜盲症（Nyctalopia）

由於視紫紅質（視網膜中感光細胞的光感受器）合成受阻，山羊在光線較暗或夜間環境下無法正常視物，行動受限。
角膜乾燥與角膜炎

角膜表面缺乏足夠的維生素 A，失去保濕功能，易形成角膜乾燥、角膜結膜發炎（結膜炎）及角膜潰瘍。
角膜硬化（角膜硬化症）

長期缺乏可造成角膜組織硬化、角膜變白，視力進一步受損。
結膜炎與結膜潰瘍

結膜黏膜乾燥、發炎，可能出現結膜潰瘍或結膜腫脹。
視網膜退化

眼睛內部視網膜細胞因缺乏維生素 A 而退化，最終可能導致永久性視力喪失。
光敏感（Photophobia）

角膜和結膜乾燥後，對光線敏感，山羊可能因光線過亮而避免直視。
視覺功能進一步惡化

如果不及時補充維生素 A，視力可能從夜盲逐漸惡化為部分或完全失明。
總結
缺乏維生素 A 主要導致夜盲、角膜乾燥、角膜炎、結膜炎、角膜硬化及視網膜退化等視覺障礙，最終可能引致視力喪失。及時補充維生素 A，並維持適當的營養攝取，可有效預防這些問題。"""
    ]

    run_evaluation(references, predictions, model_name="hfl/chinese-roberta-wwm-ext-large", stride=128, device="cpu")
'''
llama3:8b-instruct-q4_K_M (rag)
deepseek-r1:8b-llama-distill-q4_K_M (rag)        9e54bf95550c    4.9 GB    13 days ago
mistral:7b-instruct-v0.3-q4_K_M (RAG)             6577803aa9a0    4.4 GB    13 days ago
gemma3:12b-it-q4_K_M  (rag)            f4031aab637d    8.1 GB    7 days ago

goat_llama3_8bit_q4_K_M_0819_1553:latest    343bbaa756e4    4.9 GB    6 days ago
llama3:8b-instruct-q4_K_M 
gpt-oss:20b(rag)                                 aa4295ac10c3    13 GB     4 days ago

'''