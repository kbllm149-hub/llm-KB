好的，這是您提供的 **GraphRAG** Jupyter Notebook 程式碼的完整 Markdown 註解與功能整理。

---

# GraphRAG 筆記本核心功能與函式註解

這份筆記本實作了一個結合 **Neo4j 知識圖譜** 與 **Ollama LLM** 的 **Graph Retrieval-Augmented Generation (GraphRAG)** 系統。

## 一、 參數設定與初始化 (Configuration and Initialization)

| 變數/配置 | 程式碼變數名 | 類型 | 功能說明 |
| :--- | :--- | :--- | :--- |
| **Neo4j 連線** | `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PW` | `str` | Neo4j 資料庫連線資訊。 |
| **資料路徑** | `DATA_FOLDER` | `str` | 存放原始文本文件（`.txt`）的路徑。 |
| **LLM 模型** | `OLLAMA_MODEL` | `str` | 用於知識抽取和 QA 生成的 LLM 名稱（例如 `llama3:8b-instruct-q4_K_M`）。 |
| **嵌入模型** | `EMBED_MODEL_REPO` | `str` | 中文向量模型（例如 `GanymedeNil/text2vec-large-chinese`）。 |
| **檢索數量** | `TOP_K` | `int` | 檢索階段（向量或圖形）希望取得的 **最相關 Chunk 數量**。 |
| **驅動程式** | `driver` | `neo4j.GraphDatabase` | Neo4j Python 驅動程式物件，管理連線。 |

---

## 二、 資料準備與入庫函式 (Data Ingestion Pipeline)

### 1. `chunk_text(text: str, max_length: int = 300, overlap: int = 50)`

* **功能**：將一篇長文本切割成帶有重疊的短文本塊（Chunk）。
* **關鍵參數**：
    * `max_length`：每個 Chunk 的最大長度（預設 **300**）。
    * `overlap`：相鄰 Chunk 之間的**重疊長度**（預設 **50**），確保語意連貫。
* **用途**：將非結構化文件轉換為 RAG 系統的基本檢索單元 `(:Chunk)`。

### 2. `load_txt_files(folder: str)`

* **功能**：從指定資料夾載入所有 `.txt` 文件並進行分塊處理。
* **輸出**：一個包含所有 Chunk 資訊（`text`、`source`、`id`）的字典列表。

### 3. `ensure_vector_index(...)`

* **功能**：在 Neo4j 中建立或確認 **向量索引**（`VECTOR INDEX`）已存在。
* **用途**：為 `(:Chunk)` 節點的 `embedding` 屬性建立索引，大幅加速向量相似性搜尋。

### 4. `ingest_documents(...)`

* **功能**：計算 Chunk 的向量嵌入，並將其作為 `(:Chunk)` 節點寫入 Neo4j。
* **儲存屬性**：`id`、`text`、`source`、**`embedding`**。

---

## 三、 知識圖譜抽取與建構函式 (Knowledge Graph Extraction)

### 5. `extract_triples_from_chunk(chunk_text: str, model: str = OLLAMA_MODEL)`

* **功能**：使用 **Ollama LLM** 從文本塊中抽取結構化的 **知識三元組**。
* **邏輯**：使用特定的 Prompt 指導 LLM 輸出 `head`、`relation`、`tail` 的 JSON 陣列。
* **用途**：將非結構化數據轉化為圖譜知識。

### 6. `ingest_triples_with_provenance(triples: List[Dict], chunk_id: str, driver)`

* **功能**：將抽取出的三元組寫入 Neo4j，並建立**溯源關係（Provenance）**。
* **建立節點/關係 (Cypher `MERGE`)**：
    * **實體**：`(:Entity {name: triple.head})`。
    * **關係**：`[:RELATION {type: triple.relation}]`。
        * **`type` 屬性**：儲存 LLM 提取的關係類型（例如「缺乏」、「造成」）。
    * **溯源關係**：`(:Chunk)-[:MENTIONS]->(:Entity)`，將實體與其來源的 Chunk 連結。
* **用途**：構建可供圖形擴展和推理的結構化知識庫。

---

## 四、 檢索與生成核心函式 (Retrieval and Generation Functions)

這些函式主要透過 **`neo4j_graphrag`** 套件中的 `GraphRAG` 類來實現。

### 7. `qa_only(query: str, top_k: int = TOP_K, temperature: float | None = None)`

* **功能**：**純向量檢索 RAG**。只使用向量相似度搜尋來獲取上下文。
* **用途**：作為對比其他 RAG 模式的**基準測試**。

### 8. `qa_hybrid(query: str, top_k: int = 5, ranker='linear', alpha: float = 0.6)`

* **功能**：**混合檢索 RAG**。同時使用向量相似度搜尋與全文關鍵字搜尋。
* **關鍵參數**：
    * `top_k`: 檢索到的最相關分塊數量（預設 **5**）。
    * `ranker`: 融合兩種搜尋結果的排名策略（預設 **`linear`**）。
    * **`alpha`**: **權衡參數**（介於 0 到 1）。預設 **0.6**，表示檢索時給予 **向量相似度** 較高的權重（$\alpha$）。
* **用途**：平衡語意理解和精確關鍵字匹配，提高召回率。

### 9. `qa_graph(query: str, max_depth: int = 1, ...)`

* **功能**：**圖形擴展 RAG**。利用知識圖譜結構來豐富 LLM 上下文。
* **核心步驟**：
    1.  **初始檢索**：找出相關的 Chunk。
    2.  **實體識別**：透過 `[:MENTIONS]` 找到相關實體。
    3.  **圖形擴展**：從實體出發，沿著 **`[:RELATION]`** 關係向外擴展知識。
    4.  **增強上下文**：將結構化圖譜內容轉換為文本，與初始 Chunk 內容一起傳給 LLM。
* **用途**：處理需要**多跳推理**和**連貫背景知識**的複雜問題。

---

## 五、 優化與輔助功能

| 程式碼區塊/功能 | 說明 |
| :--- | :--- |
| **`CREATE CONSTRAINT`** | 建立 `Chunk.id` 和 `Entity.name` 的**唯一性約束**，確保資料庫的資料品質和完整性。 |
| **`CREATE FULLTEXT INDEX`** | 為 `Chunk.text` 屬性建立 **全文索引**（例如 `chunk_text_fts`），是實現 **`qa_hybrid` 混合檢索**的基礎。 |
| **性能測量** | 透過 `time.perf_counter()` 計算推論延遲 (`latency_ms`) 和 LLM 吞吐量 (`tps`)，用於系統性能監控。 |
| **批量處理** | 筆記本末端示範將 **Chunk 嵌入和入庫** 從單次循序處理優化為 **`UNWIND` 批量寫入**，大幅提升數據入庫效率。 |