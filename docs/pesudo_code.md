下面整理成**可直接落地的虛擬碼（pseudo-code）**，聚焦在你要的**三塊困難核心**：

1. **流形距離計算（Log-Euclidean / Bures–Wasserstein）**
2. **部位對齊與負樣本挖掘**
3. **損失函數（PAC-MCL 與總損失）**
   另外我也補上**穩定化與效能關鍵細節**（你可能會遺漏的地方），都以框架無關的「Python 風格」虛擬碼表示，便於移植到 PyTorch + timm。

---

## 0) 公用工具（數值穩定 & SPD 操作）

```python
# ===== 數值與線性代數輔助 =====
def symmetrize(M):
    return 0.5 * (M + M.T)

def clamp_eigvals(eigvals, eps):
    # 針對 SPD 的最小特徵值鉗制，避免接近 0 造成不穩
    return maximum(eigvals, eps)

def eig_psd(M, eps):
    # 假設 M ~ SPD；若非嚴格 SPD，做對稱化 + 特徵值鉗制
    M = symmetrize(M)
    U, S = eigh(M)                # S 非降序實數
    S = clamp_eigvals(S, eps)
    return U, S

def sqrtm_psd(M, eps):
    U, S = eig_psd(M, eps)
    return U @ diag(sqrt(S)) @ U.T

def invsqrtm_psd(M, eps):
    U, S = eig_psd(M, eps)
    return U @ diag(1.0 / sqrt(S)) @ U.T

def logm_psd(M, eps):
    U, S = eig_psd(M, eps)
    return U @ diag(log(S)) @ U.T

def trace(M):
    return sum(diag(M))

# ===== 部位共變異（帶收縮）的穩定估計 =====
def covariance_part(F: Matrix[n_p, d], eps, shrinkage_alpha=None):
    # F: 該部位收集到的特徵向量（已做降維，例如 d'=64）
    mu = mean(F, axis=0)
    X = F - mu
    C = (X.T @ X) / max(n_p, 1)
    # Ledoit-Wolf 式收縮（可選）：提高小樣本穩定性
    if shrinkage_alpha is not None:
        tr = trace(C) / C.shape[0]
        C = (1 - shrinkage_alpha) * C + shrinkage_alpha * (tr * I_like(C))
    # SPD 保證
    C = C + eps * I_like(C)
    return symmetrize(C)
```

> **要點**
>
> * 記得在整個訓練過程中**紀錄最小特徵值分佈**，監控是否貼近 0。
> * 若批量太小導致共變異不穩，開啟 `shrinkage_alpha`（例如 0.05–0.1）。

---

## 1) 流形距離計算（LE 與 Bures）

```python
# ===== Log-Euclidean 距離 =====
# δ_LE(A,B) = || log(A) - log(B) ||_F
def delta_log_euclidean(A, B, eps):
    LA = logm_psd(A, eps)   # 可「預先快取」對每個部位的 logm
    LB = logm_psd(B, eps)
    return frobenius_norm(LA - LB)

# ===== Bures–Wasserstein 距離 =====
# δ_BW^2(A,B) = tr(A) + tr(B) - 2 * tr( (A^{1/2} B A^{1/2})^{1/2} )
def delta_bures(A, B, eps):
    As = sqrtm_psd(A, eps)                 # A^{1/2}
    C  = As @ B @ As                       # A^{1/2} B A^{1/2}
    Cs = sqrtm_psd(C, eps)                 # ( ... )^{1/2}
    return sqrt( trace(A) + trace(B) - 2.0 * trace(Cs) )

# ===== 批次向量化（效能關鍵） =====
# 對整個 batch 的 P×P 部位對計算距離矩陣時，建議：
# 1) LE：先對所有 C 做一次 logm，之後只做歐氏差的 Fro 範數
# 2) BW：可預先計算每個 A 的 A^{1/2}；仍需對 (A^{1/2} B A^{1/2}) 做 sqrtm
# 若成本過高，可在 BW 階段只對「候選的少量對」做距離（例如 top-k 採樣）
```

> **要點**
>
> * 訓練「前半段」建議用 **LE**（較穩、可快取 logm），收斂後再切到 **BW** 微調 20–30 個 epoch。
> * BW 若太慢：僅對**正對 + 若干最難負對**計算；或採 **Newton–Schulz** 迭代近似（需良好條件化）。

---

## 2) 部位對齊（同圖雙視圖）與負樣本挖掘（跨圖）

```python
# 輸入：
#   C1 = {C1[p]}_{p=1..P}  ← 視圖 v1 的 P 個部位 SPD
#   C2 = {C2[q]}_{q=1..P}  ← 視圖 v2 的 P 個部位 SPD
#   metric ∈ { "LE", "BW" }
#   eps: SPD 穩定常數
# 輸出：
#   pi: 對齊映射 p -> q
#   D:  距離矩陣 (P x P)

def pairwise_distance_matrix(C1_list, C2_list, metric, eps):
    P = len(C1_list)
    D = zeros((P, P))
    if metric == "LE":
        # 快取 logm
        L1 = [logm_psd(C, eps) for C in C1_list]
        L2 = [logm_psd(C, eps) for C in C2_list]
        for p in range(P):
            for q in range(P):
                D[p,q] = frobenius_norm(L1[p] - L2[q])
    else:  # "BW"
        A_sqrt = [sqrtm_psd(C, eps) for C in C1_list]
        for p in range(P):
            Ap = A_sqrt[p]
            for q in range(P):
                Cs = sqrtm_psd(Ap @ C2_list[q] @ Ap, eps)
                D[p,q] = sqrt( trace(C1_list[p]) + trace(C2_list[q]) - 2.0 * trace(Cs) )
    return D

def align_parts(C1_list, C2_list, metric, eps, method="BNN"):
    D = pairwise_distance_matrix(C1_list, C2_list, metric, eps)
    if method == "BNN":
        # 雙向最近鄰：p 的最近 q 與 q 的最近 p 相互匹配才算
        p2q = argmin(D, axis=1)         # 每列最小
        q2p = argmin(D, axis=0)         # 每行最小
        pi = {p: q for p, q in enumerate(p2q) if q2p[q] == p}
        # 可能有未匹配的 p，為其分配次佳 q（不與已配衝突）
        unmatched = [p for p in range(P) if p not in pi]
        free_q = set(range(P)) - set(pi.values())
        for p in unmatched:
            # 選擇 D[p, q] 最小且 q ∈ free_q
            q_best = argmin_over_subset(D[p, :], free_q)
            pi[p] = q_best
            free_q.remove(q_best)
        return pi, D
    elif method == "Hungarian":
        # 最小化總距離的完美匹配
        pi = hungarian_min_cost_matching(D)   # 回傳 dict: p -> q
        return pi, D
```

**跨圖的負樣本挖掘（hard negatives）：**

```python
# 給定：
#   anchor 的 C1_list（視圖 v1）
#   negatives: 批內其他影像的部位 SPD 列表清單 [[Cneg_j[q]]_{q=1..P}]_{j=1..M}
# 目標：對每個 anchor 部位 p，挑出 top-k 最難負樣本部位索引 (j*, q*)
def mine_hard_negatives(C1_list, negatives, metric, eps, topk=5):
    hard_for_p = dict()  # p -> list of (delta, j_idx, q_idx)
    for p, Cp in enumerate(C1_list):
        cand = []
        for j, Cneg in enumerate(negatives):
            # 避免全配：先快速估計，必要時精算
            # 這裡直接精算，實作時可用快捷指標過濾
            for q, Cq in enumerate(Cneg):
                if metric == "LE":
                    d = delta_log_euclidean(Cp, Cq, eps)
                else:
                    d = delta_bures(Cp, Cq, eps)
                cand.append( (d, j, q) )
        cand.sort(key=lambda x: x[0])       # 距離小 → 更難
        hard_for_p[p] = cand[:topk]
    return hard_for_p  # 每個 p 有 top-k 難負列表
```

> **要點**
>
> * **效能**：先以**便宜的 proxy**（例如 trace 差、對角近似距離）做粗篩，留下 top-(k’≫k) 再計 BW；能省大量 sqrtm。
> * **去同類負樣本**：若批內有同類樣本，負樣本挑選時可排除同類（或降權）。

---

## 3) PAC-MCL 損失與總損失

```python
# ===== PAC-MCL: Part-Aware Manifold Contrastive Loss (triplet-on-manifold) =====
# 對每個 anchor 影像的每個部位 p：
#   正對距離 = 與正視圖對齊部位的流形距離
#   負對距離 = 從批內其他影像部位中，挑最難或 top-k 平均
# 損失 = 平均_p [ delta_pos - delta_neg + margin ]_+

def pac_mcl_loss(C_anchor, C_pos, negatives, metric, eps,
                 align_method="BNN", margin=0.2, topk=5, reduce_neg="min"):
    # 對齊 anchor 與 pos 視圖
    pi, D_pos = align_parts(C_anchor, C_pos, metric, eps, method=align_method)

    # 負樣本挖掘
    hard_map = mine_hard_negatives(C_anchor, negatives, metric, eps, topk=topk)

    losses = []
    for p in range(len(C_anchor)):
        # 正對距離（已對齊）
        q = pi[p]
        delta_pos = D_pos[p, q]   # 直接讀距離矩陣（或現算）

        # 負對距離（採 min 或 top-k 平均）
        H = hard_map[p]  # list of (delta, j, q)
        if reduce_neg == "min":
            delta_neg = H[0][0]
        elif reduce_neg == "mean":
            delta_neg = mean([h[0] for h in H])
        elif reduce_neg == "softmin":  # 溫度化的軟最小值，降低梯度方差
            tau = 0.05
            w = softmax( [-h[0]/tau for h in H] )
            delta_neg = sum( w_i * H[i][0] for i, w_i in enumerate(w) )
        else:
            delta_neg = H[0][0]

        l = relu(delta_pos - delta_neg + margin)
        losses.append(l)

    # 對 anchor 的 P 個部位取平均
    return mean(losses)

# ===== 總損失（分類 + 對比） =====
def total_loss(logits_v1, logits_v2, y,
               C_v1, C_v2, negatives_for_v1, negatives_for_v2,
               metric, eps, gamma=0.75, lam_posCE=1.0, **kwargs):
    # 分類（主視圖）
    L_ce_main = cross_entropy(logits_v1, y)
    # 分類（正視圖，可選，提升一致性）
    L_ce_pos  = cross_entropy(logits_v2, y)

    # PAC-MCL：對 v1 作 anchor（也可再對 v2 作一次對稱，取平均）
    L_pac_1 = pac_mcl_loss(C_v1, C_v2, negatives_for_v1, metric, eps, **kwargs)
    L_pac_2 = pac_mcl_loss(C_v2, C_v1, negatives_for_v2, metric, eps, **kwargs)
    L_pac   = 0.5 * (L_pac_1 + L_pac_2)

    L_total = L_ce_main + lam_posCE * L_ce_pos + gamma * L_pac
    return L_total, {"ce_main": L_ce_main, "ce_pos": L_ce_pos, "pac": L_pac}
```

> **要點**
>
> * **對稱化**（v1 當 anchor、v2 當 anchor 各一次）能提升穩定度。
> * **reduce\_neg** 用 `softmin` 能緩和 hard-negative 的高方差。
> * `margin` 可隨訓練**餵食策略**調整（例如先小後大）。

---

## 4) 前向流程（整合 backbone → 部位 → SPD）

```python
# feature_map: 來自 timm CNN 的最後/倒數特徵圖 (B, H, W, d)
# part_pooling: 例如注意力加權聚合 + k-means/Sinkhorn，輸出 P 個部位，每部位收集 n_p 個向量
# reduce_dim: 可用 1x1 conv 或 PCA 投影到 d'（提升 SPD 操作效率）

def view_to_part_spd(feature_map, P, d_reduced, eps, shrinkage_alpha=None):
    tokens = spatial_tokens(feature_map)          # (N_tokens, d)
    F_parts = part_pooling(tokens, num_parts=P)   # List[Matrix[n_p, d]]
    F_parts = [ reduce_dim(F) for F in F_parts ]  # (n_p, d') each

    C_parts = [ covariance_part(F, eps, shrinkage_alpha) for F in F_parts ]
    return C_parts  # List[SPD(d' x d')]
```

---

## 5) 訓練一個 iteration（含 LE→BW 課程切換）

```python
def train_iter(batch, model, epoch, switch_epoch=60,
               metric_start="LE", metric_after="BW",
               eps=1e-4, P=6, d_reduced=64, **loss_kwargs):

    x, y = batch
    # 生成兩個視圖（含遮罩/洗牌等強增強）
    v1, v2 = strong_augment_pair(x)  # 與 CLE 風格一致

    # 前向：兩視圖 backbone
    f1 = model.backbone(v1)          # (B, H, W, d)
    f2 = model.backbone(v2)

    # 分類 logits
    logits_v1 = model.head(global_pool(f1))
    logits_v2 = model.head(global_pool(f2))

    # 產生部位 SPD
    C_v1 = [ view_to_part_spd(f1[b], P, d_reduced, eps) for b in range(B) ]
    C_v2 = [ view_to_part_spd(f2[b], P, d_reduced, eps) for b in range(B) ]

    # 構建 negatives（批內其他樣本）
    negatives_for_v1 = build_negatives(C_v2)  # 對 v1 anchor，用 v2 其他樣本為負
    negatives_for_v2 = build_negatives(C_v1)

    # 流形度量課程：先 LE 後 BW
    metric = metric_after if epoch >= switch_epoch else metric_start

    # 逐樣本計算 PAC-MCL，並彙總
    L_total = 0.0
    logs    = avg_meter_dict()
    for b in range(B):
        Lt, stat = total_loss(logits_v1[b], logits_v2[b], y[b],
                              C_v1[b], C_v2[b],
                              negatives_for_v1[b], negatives_for_v2[b],
                              metric=metric, eps=eps, **loss_kwargs)
        L_total += Lt
        logs.update(stat)
    L_total = L_total / B
    return L_total, logs
```

---

## 6) 你可能會遺漏但很關鍵的細節

1. **快取 / 向量化**

   * **LE**：先把所有部位 SPD 做一次 `logm_psd`，之後距離只做 `||LA - LB||_F`（O(d'^2)）。
   * **BW**：預先計算每個 anchor 的 `sqrtm_psd(A)`，再對候選負樣本做 `sqrtm_psd(A B A)`；同時只對**正對 + 最難 k 個負對**做 BW，其他用 proxy 篩掉。

2. **Proxy 距離（加速 BW 挖掘）**

   * 可先用 `||A - B||_F`、`|trace(A)-trace(B)|`、或 `||diag(A)-diag(B)||_2` 做粗篩，僅保留前 k’（例如 25）再精算。

3. **數值保險**

   * 共變異加 **收縮** + `eps I`，並在 `eigh` 前**對稱化**矩陣。
   * 若 `sqrtm/logm` 偶發數值錯誤，回退到**對角近似**（僅取 `diag(C)`），保證訓練不中斷。

4. **正則化（可選）**

   * **跨類間隔正則**：以類別為單位，對「部位中心」的流形距離加上**margin**上推。
   * **對齊一致性**：在連續數個 epoch 中，限制對齊映射變動過大（提高穩定性）。

5. **梯度方差控制**

   * **softmin** 負樣本彙總或 **top-k 平均**，比單一最難負樣本更穩。
   * 搭配 **EMA**（exponential moving average）更新 backbone 參數，使對比分支更平滑。

6. **單機多卡**

   * 檢索負樣本時可用 **DDP + all\_gather** 擴大負樣本池；記得只回傳對應子 batch 的梯度。

7. **推論路徑**

   * 若追求**低延遲**，推論時可只保留 `global_pool → head`；需要可解釋性再計算部位 SPD 作分析。

---

如果你希望，我可以把以上虛擬碼**轉成可執行的 PyTorch 原型**（含 batched `logm/sqrtm`、Hungarian 實作、DDP 支援），以及**加上單元測試**（SPD 檢查、數值穩定測試、效能基準）。要我直接幫你生一個 `pac_mcl.py` + `metrics.py` 的最小倉庫骨架嗎？
