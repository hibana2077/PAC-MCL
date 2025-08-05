好的，Tom。我先給你兩份可直接落地的文件草案（可貼進 Overleaf/Notion）。如需把流程圖輸出成圖片（Graphviz/PowerPoint），我也可以幫你產出檔案。

---

# 文件一：Training + Validation 流程圖（PAC-MCL for timm CNN）

> 目標：清楚標示從資料載入 → 雙視圖增強 → 部位抽樣 → SPD/Bures 流形距離 → PAC-MCL 損失與分類頭共同訓練；以及驗證/推論路徑。

## A. 訓練流程（文字版流程圖）

```
[Start]
  ↓
[DataLoader]
  - 讀取影像 x 與標籤 y
  - 產生兩個視圖 v1, v2（強增強：遮罩、區塊洗牌、顏色/幾何）
  ↓
[timm CNN Backbone]
  - f1 = CNN(v1) ∈ R^{H×W×d}
  - f2 = CNN(v2) ∈ R^{H×W×d}
  ↓
[部位產生(無標註)]
  - 將 f1, f2 的空間特徵做注意力/聚合
  - 以 k-means / Sinkhorn 取得 P 個部位原型
  - 得到每視圖的部位特徵集 {F_p}, p=1..P
  ↓
[部位共變異(二階統計)]
  - C_p = Cov(F_p) + εI  (對每個 p)
  - 可做降維(PCA到 d’=64)
  ↓
[流形距離計算]
  - 選擇 δ = Log-Euclidean 或 Bures
  - 在 {C_p^v1} 與 {C_q^v2} 間計算距離矩陣 D
  ↓
[部位對齊]
  - 以雙向最近鄰(BNN) 或 匈牙利法求 π(p)
  - 正對距離: δ(C_p^v1, C_{π(p)}^v2)
  - 負對: 由批內其他影像的部位選 hard negatives
  ↓
[損失計算]
  - 分類頭：CE(y, head(GlobalPool(f1)))
  - PAC-MCL: 平均 over p 的 triplet-on-manifold
  - 總損失: L = CE + λ·CE_posview + γ·PAC-MCL
  ↓
[反向傳播 & 更新參數]
  - AdamW / Cosine LR / EMA(可選)
  ↓
[Logging]
  - 訓練損失/精度、流形距離統計、時間/記憶體
  ↓
[Next Iteration]
```

### 設計備忘

* `P`（部位數）建議 6–8；`ε=1e-4`；特徵降維到 64 維再算 SPD 可加速。
* Bures 計算使用矩陣平方根（eigh/SVD）；先用 Log-Euclidean 訓練穩定，再切 Bures 微調。

---

## B. 驗證 / 推論流程（文字版流程圖）

```
[Start]
  ↓
[DataLoader (val/test)]
  - 讀取影像 x 與標籤 y
  - 單視圖 (val/test 通常不做強遮罩/洗牌)
  ↓
[timm CNN Backbone]
  - f = CNN(x) ∈ R^{H×W×d}
  ↓
[分類預測(主路徑)]
  - ŷ = head(GlobalPool(f))
  - 計算 Top-1/Top-5, Macro-F1, ECE(校準) 等
  ↓
[部位-流形度量(分析用，非必需)]
  - 產生 P 個部位特徵集 {F_p} → C_p
  - 計算 intra/inter-class 的流形距離統計
  - 記錄穩定性(最小特徵值)、分散度等
  ↓
[匯總指標與可視化]
  - 每類精度、混淆矩陣
  - UMAP/t-SNE on manifold-induced kernel(可選)
  ↓
[End]
```

> 推論時若想降延遲，可移除對比分支；部位模組可保留以提升穩健或作可視化解釋。