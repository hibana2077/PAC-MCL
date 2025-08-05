# 文件二：要跑的實驗與要收集的數據

> 目標：給出**最小但充足**的實驗矩陣，覆蓋 baseline、我們的方法（LE/Bures）、以及關鍵消融；同時規範**記錄的指標與數據**，確保可重現與可診斷。

## A. 資料集與拆分

* **UFG 基準**的 5 個子資料集（Cotton80, SoyLocal, SoyGene, SoyAgeing, SoyGlobal）。
* 每個資料集：官方拆分或 80/10/10（train/val/test）。
* 每個設定**固定 3 個隨機種子**（42/43/44），報告 mean ± std。

## B. Backbone 與超參數（timm）

* **Backbones**（小到大各一）：`resnet50`, `convnext_tiny`, `efficientnet_b0`。
* **優化器**：AdamW（lr=3e-4, wd=0.05，視 backbone 微調）；Scheduler：Cosine + 5 epoch warmup。
* **Batch**：64（視 GPU），Amp 混合精度。訓練 100–150 epochs。
* **損失權重**：λ=1.0，γ=0.5→1.0 網格搜尋。
* **部位**：P ∈ {4, 6, 8}；降維 d’=64；ε=1e-4。

## C. 實驗矩陣（核心 + 消融）

### C-1. 核心比較（每個資料集 × 每個 backbone × 3 seeds）

| ExpID | 方法                                | 損失/度量            | 對齊       | 負樣本             | 說明                |
| ----- | --------------------------------- | ---------------- | -------- | --------------- | ----------------- |
| B0    | **CE Baseline**                   | 只有 CE            | –        | –               | 純分類頭              |
| B1    | **SupCon (Euclidean)**            | CE + SupCon      | instance | batch negatives | 常見對比基線            |
| B2    | **CLE-style Triplet (Euclidean)** | CE + Triplet     | instance | hard mining     | 參考 CLE 的精神        |
| O1    | **PAC-MCL (LE)**                  | CE + PAC-MCL(LE) | BNN      | top-k hard      | 我們的方法（Log-Euclid） |
| O2    | **PAC-MCL (Bures)**               | CE + PAC-MCL(BW) | 匈牙利法     | top-k hard      | 我們的方法（Bures）      |

### C-2. 消融與設計選擇（以最佳 backbone+資料集執行 × 3 seeds）

| AblID | 變因    | 設定                       | 假設/目的            |
| ----- | ----- | ------------------------ | ---------------- |
| A1    | 部位數 P | 4 / 6 / 8 / 12           | 過少學不到細節；過多噪聲/成本↑ |
| A2    | 對齊策略  | BNN / 匈牙利法               | 全局匹配是否優於局部最近鄰    |
| A3    | 負樣本策略 | batch hard / memory bank | 訓練穩定性與效能權衡       |
| A4    | 度量    | LE / BW                  | BW 是否帶來額外提升      |
| A5    | 降維 d’ | 32 / 64 / 128            | SPD 計算成本 vs 表徵力  |
| A6    | ε 穩定化 | 1e-5 / 1e-4 / 1e-3       | 數值穩定性影響          |
| A7    | γ 權重  | 0.25 / 0.5 / 1.0         | 對比損失占比的靈敏度       |
| A8    | 視圖增強  | +遮罩洗牌 / 無遮罩洗牌            | CLE 風格增強對方法必要性   |

> 執行順序：先 C-1 找到**最佳主幹設定**，再針對該設定跑 C-2。

## D. 主要評估指標（分類/檢索/穩定/效率）

### D-1. 分類效能（主報）

* **Top-1 / Top-5 準確率**（test/val）。
* **Macro-F1**（解決類別不平衡）。
* **Calibration**：ECE / NLL。
* **混淆矩陣**與**每類精度**（了解困難類）。

### D-2. 檢索與度量學習質量（輔助報）

* **mAP\@R / Recall\@K**：以最終全局向量或部位聚合向量做最近鄰檢索。
* **類內 vs 類間流形距離**：

  * `Intra-BW` = 類內部位 SPD/Bures 距離的平均。
  * `Inter-BW` = 類間最小（或中位數）距離。
  * **Margin = Inter − Intra**（越大越好）。
* **部位對齊一致性**：正對配對距離的中位數/95分位。

### D-3. 數值穩定性與幾何統計（診斷）

* SPD 特徵的**最小特徵值分佈**（監控是否貼近 0）。
* 矩陣對數/平方根運算的**失敗率**（應為 0；若>0，回退到對角近似）。
* **梯度范數**與**loss 曲線**的平滑度/震盪度。

### D-4. 效率與資源

* **每 step 計算時間**（ms/iter）、**吞吐**（img/s）。
* **GPU 記憶體峰值**。
* **參數量 / FLOPs**（可用 fvcore/thop 粗估）。

> 彙整時：各指標報 `mean ± std (n=3)`；對主結果做 **t-test** 或 **bootstrap CI**。

## E. 成功準則（Gate）

* 在**至少 3/5 個 UFG 子集**上，O1/O2 **Top-1** 明顯優於 B2（Euclidean triplet）且 Macro-F1 ↑。
* `Margin (Inter−Intra)` 在 O1/O2 顯著大於 B2（p<0.05）。
* 訓練 150 epochs 內**無數值崩潰**（最小特徵值分佈 > 1e-6）。
* 計算開銷相對 B2 **增加不超過 35%**（以每 iter 時間衡量）或能以 d’=64/P=6 控制在門檻內。

## F. 記錄與可重現（Logging 規範）

* 固定：`seed, git commit, timm/backbone 版本, CUDA/cuDNN, 驗證集 hash`。
* 完整保存：`最佳權重(Top-1) + 週期性 checkpoint`、`config.yaml`。
* Log 工具：TensorBoard/W\&B；每 200 iter 記錄一次訓練損失與效率；每 epoch 記錄所有驗證指標。
* 將**部位可視化**（部位注意力/遮罩熱力圖）存檔，供論文圖使用。

## G. 風險與回退策略（跑不動/不穩時）

* **先 LE 後 BW**：先以 Log-Euclidean 收斂，再切換 Bures 微調 20–30 epochs。
* **降維更小**：d’=32；或以對角/Cholesky 近似加快 SPD 運算。
* **對齊改簡單**：先用 BNN 對齊再漸進切匈牙利法。
* **hard-negative 降溫**：用溫度 τ 提升穩定，或改 top-k 平均代替 max。