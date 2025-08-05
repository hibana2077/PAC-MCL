下面給你一個**小而完整、以「目標函數／度量學習」為核心**、可直接往 WACV 投的題目。構想緊扣你附的兩篇 paper（CLE-ViT 與 UFG benchmark），並明確落在你指定的\*\*「部位感知對比（part-aware contrast）」＋「流形（manifold）上定義 loss」**的方向，同時保證**能在 timm 的所有 CNN backbone 上實作\*\*。

---

## 論文題目（暫定）

**PAC-MCL：部位感知的流形對比損失，用於超細粒度視覺分類（Part-Aware Contrast on Manifold for Ultra-Fine-Grained Visual Categorization）**

---

## 研究核心問題

在**超細粒度**任務中，同類差異大、跨類差異小，導致用單一全局向量做對比學習時，模型容易**對局部微差異不敏感**、泛化性差。UFG 基準指出資料具**大內類變異與小跨類差異**，且樣本偏少，易過擬合【】【】；CLE-ViT 展示**自監督對比＋影像內部遮罩／洗牌**有助於學到更能容忍內類變異且拉開跨類距離的表徵【】【】。
\*\*核心問題：\*\*如何將「**部位級**」的微差異對齊與區分，**在適合的幾何流形上**建立可優化、穩定且計算可行的對比損失，以同時放大跨類間隔、又不犧牲對內類變異的容忍度？

---

## 研究目標

1. \*\*部位感知表徵：\*\*在不依賴額外標註下，從 CNN 特徵圖中自動萃取一組可解釋的「部位」子區域向量／特徵統計。
2. \*\*流形建模：**將每個部位的**二階統計（共變異）\*\*嵌入到 **SPD 流形**（或以 Bures–Wasserstein 幾何等價）上，於流形上量測部位間距離。
3. **目標函數：**提出**部位感知流形對比損失（Part-Aware Manifold Contrastive Loss, MCL）**，結合跨圖、跨部位的對齊與分離；並與分類交叉熵共同訓練。
4. **理論保證：**給出損失的**下界／上界**與**收斂性**分析（在合適假設下），以及\*\*幾何凸性（geodesic convexity）\*\*的充要條件（在 Log-Euclidean 近似或 Bures 幾何下）。
5. **實作與可移植性：**提供在 **timm 全部 CNN backbone**（ResNet/RegNet/ConvNeXt/EfficientNet…）上的**即插即用**實作。

---

## 方法概要

### 1) 部位產生（自監督、無標註）

* 從 backbone（timm CNN）最後一層或倒數幾層的特徵圖取出 spatial tokens。
* 參考 CLE-ViT 的**強增強＋遮罩／洗牌**策略生成另一視圖，作為**同一影像的正對**【】【】；批內其他影像為負對。
* 透過簡單的\*\*注意力聚合＋k-means（或 Sinkhorn/OT 對齊）\*\*產生 **P 個部位原型**，避免依賴顯式關鍵點。
* （與 CLE-ViT 相容）此步驟不需額外標註，且能自然增加**內類變異**、拉開**跨類間距**【】。

### 2) 流形嵌入（SPD / Bures）

* 對每個部位，計算其特徵向量集合的**樣本共變異矩陣** $C_p \in \mathbb{S}_{++}^{d}$。
* 在 SPD 流形上採用下列任一等價度量：

  * **Log-Euclidean**：$\delta_{\text{LE}}(C_i,C_j)=\|\log C_i-\log C_j\|_F$。
  * **Bures–Wasserstein**（= 2-Wasserstein for zero-mean Gaussians）：
    $\delta_{\text{BW}}^2(C_i,C_j)=\mathrm{Tr}(C_i)+\mathrm{Tr}(C_j)-2\mathrm{Tr}\!\left[(C_i^{1/2}C_j C_i^{1/2})^{1/2}\right]$。
* 我們偏向 **Bures**：有良好幾何性質，且與「把分佈視為高斯近似」的直覺一致。

### 3) **部位感知流形對比損失（MCL）**

* 對一個 anchor 影像 $x$ 的第 $p$ 個部位共變異 $C_p^x$，與其正對影像 $x^+$ 的對應部位 $C_{\pi(p)}^{x^+}$ 之間，最小化流形距離；對任意負對 $x^-$ 的部位 $C_q^{x^-}$，最大化距離。
* **匹配 $\pi$** 可用雙向最近鄰或匈牙利法在流形距離上求得。
* **Triplet-on-manifold 損失：**

  $$
  \mathcal{L}_{\text{PAC-MCL}}
  = \frac{1}{P}\sum_{p=1}^{P}
  \big[\, \delta(C_p^x, C_{\pi(p)}^{x^+})
   - \delta(C_p^x, C_{q^\*(p)}^{x^-}) + m \,\big]_+ ,
  $$

  其中 $\delta$ 可為 $\delta_{\text{BW}}$ 或 $\delta_{\text{LE}}$，$q^\*(p)$ 為最難負樣本的部位索引，$m$ 為邊際。
* **跨類分離正則**（可選）：對批內不同類的部位原型，加入中心級別的**流形散度正則**，進一步擴大跨類間隔。
* 與分類頭的 CE 共同優化：

  $$
  \mathcal{L}=\mathcal{L}_{\text{CE}} + \lambda \,\mathcal{L}_{\text{CE}}^{(\text{pos-view})}
  + \gamma\, \mathcal{L}_{\text{PAC-MCL}},
  $$

  形式上與 CLE-ViT 的**分類＋對比**的多目標設計相容【】【】。

> 為何有效？UFG 任務的瓶頸在於**微細部位差異**與**巨大的內類變異**【】。在**部位層級**用**二階統計**並於**流形**上度量，可同時（i）強化對**微差異**的敏感度、（ii）對**變形／光照**等一階變動具穩健性（由共變異吸收），並（iii）保留 CLE-ViT 證明有效的**影像內增強對比精神**【】。

---

## 貢獻與創新（偏數學，並含可行性驗證）

1. **新損失：部位感知的流形對比（PAC-MCL）**

   * 在**SPD/Bures 幾何**上定義的**部位級** triplet-style 對比損失；兼顧**對齊**（正對）與**分離**（負對）。
   * 與 CLE-ViT 的自監督正對生成方式**兼容**，但我們把對比移至**部位的流形統計**，而非單純歐氏特徵空間【】。

2. **幾何保證（可行性）**

   * **（命題 1）**在 **Log-Euclidean** 度量下，$\delta_{\text{LE}}$ 等價於對稱雙曲線空間的歐氏距離於矩陣對數域，因此對**仿射縮放**穩健且\*\* geodesic-convex\*\*（在 $\log$ 空間的凸性）。
   * **（命題 2）**在 **Bures** 幾何下，$\mathcal{L}_{\text{PAC-MCL}}$ 對於**位移插值**（displacement interpolation）具**弱 geodesic-convex**性質（在常見條件下），保證**最優距離差**不會在訓練中惡化。
   * \*\*（引理 1）距離下界：\*\*若 $\forall p$，負對最小距離 $\delta(C_p^x, C_{q^\*(p)}^{x^-})\ge \delta^-$ 且正對最大距離 $\delta(C_p^x, C_{\pi(p)}^{x^+})\le \delta^+$，則
     $\mathcal{L}_{\text{PAC-MCL}}=0$ 的充分條件為 $\delta^- \ge \delta^+ + m$。
   * \*\*（引理 2）Lipschitz 性：\*\*採 **Log-Euclidean** 時計算圖 $\log(\cdot)$ 的譜範數 Lipschitz 常數受最小特徵值下界控制，給出數值穩定條件（實作可加 $\epsilon I$）。

3. **收斂性（隨機最優化）**

   * 在批次均勻抽樣與步長滿足 Robbins-Monro 條件下，$\mathbb{E}\|\nabla\mathcal{L}\|^2 \to 0$；hard-negative 挖掘以**溫度化**（top-k 近似）避免高方差梯度，與 CLE-ViT 對「易負對」的觀察一致【】。

> 以上性質確保**能訓得動且會收斂**；在 SPD 上以 Log-Euclidean 或 Bures 計算距離的成本主要來自**本徵分解／矩陣平方根**，我們在「實作計畫」段落給出線性化與近似策略，確保在 timm CNN 上的**可行速度**。

---

## 數學推導與（簡要）證明

> 下面只列核心，正文可展開為附錄 A/B/C。

\*\*定義 1（部位共變異）：\*\*對影像 $x$ 的第 $p$ 部位，取其特徵集合 $\{f_i\}_{i=1}^{n_p}\subset\mathbb{R}^d$，$\mu_p=\frac{1}{n_p}\sum f_i$，
$C_p=\frac{1}{n_p}\sum (f_i-\mu_p)(f_i-\mu_p)^\top + \epsilon I\in\mathbb{S}_{++}^d$。

**定義 2（流形距離）：**

* Log-Euclidean：$\delta_{\text{LE}}(C_i,C_j)=\|\log C_i-\log C_j\|_F$。
* Bures：$\delta_{\text{BW}}^2(C_i,C_j)=\mathrm{Tr}(C_i)+\mathrm{Tr}(C_j)-2\mathrm{Tr}\!\left[(C_i^{1/2}C_j C_i^{1/2})^{1/2}\right]$。

\*\*損失：\*\*見上式 $\mathcal{L}_{\text{PAC-MCL}}$。

**命題 1（LE-geodesic convexity，概要）：**
令 $\Phi(C)=\log C$。則 $\delta_{\text{LE}}(C_i,C_j)=\|\Phi(C_i)-\Phi(C_j)\|_F$ 是歐氏度量。對每個部位，三元組損失
$[\,\delta_{\text{LE}}(C_a,C_p)-\delta_{\text{LE}}(C_a,C_n)+m]_+$ 在 $(\Phi(C_a),\Phi(C_p),\Phi(C_n))$ 的**歐氏空間**中為**凸的 hinge**（對距離差是 1-Lipschitz 的凸上界）。由於 $\Phi$ 為雙射且保持 geodesic 到直線，故在 SPD 上得到 geodesic-convex 上界。*（證略）*

\*\*引理 1（零損條件，下界）：\*\*若 $\max_p\delta(C_p^x,C_{\pi(p)}^{x^+})\le \delta^+,\ \min_p\delta(C_p^x,C_{q^\*(p)}^{x^-})\ge \delta^-$，且 $\delta^- \ge \delta^+ + m$，則 $\mathcal{L}_{\text{PAC-MCL}}=0$。*（直接代入）*

**命題 2（Bures-穩定性，概要）：**
$\delta_{\text{BW}}$ 為零均值高斯在 2-Wasserstein 下的距離。對 SPD 的位移插值（displacement geodesic）$C(t)$，$\delta_{\text{BW}}(C(t),\cdot)$ 具弱凸；故 $\mathcal{L}_{\text{PAC-MCL}}$ 的期望在隨機批次下**不增**，給出**下降序列**。*（以矩陣平方根的 Fréchet 導數與 trace 凸性工具證明，正文展開）*

**收斂性（概要）：**
令 $g_t=\nabla\mathcal{L}(\theta_t)$，步長 $\eta_t$ 遵守 $\sum\eta_t=\infty,\ \sum\eta_t^2<\infty$，且梯度二階矩有界，則 $\mathbb{E}\|g_t\|^2\to 0$。hard-negative 以 top-k 溫度化避免梯度爆炸，與 CLE-ViT 的對比模組設計相容【】。

---

## 實作規格（針對 **timm** 所有 CNN backbones）

* \*\*Backbone：\*\*任選 timm CNN（ResNet 家族、RegNet、ConvNeXt、EfficientNet…），凍結前半段、微調後半段。
* \*\*部位抽樣：\*\*從最後一層特徵圖（$H\times W\times d$）取 $K$ 個 token，以注意力引導的聚合得到 $P$ 個部位（預設 $P=6\sim 8$）。
* **增強／正對生成：**依 CLE-ViT，加入**遮罩＋區塊洗牌**的強增強作為正對視圖【】【】；批內其他影像為負對【】。
* \*\*共變異與數值：\*\*每部位特徵做去均值、加 $\epsilon I$（$\epsilon=10^{-4}$），以 **eigh** 求矩陣平方根／對數。

  * **加速**：小維度投影（PCA 到 $d=64$）、**Nyström** 近似或對角近似可將每 batch SPD 操作控制在可訓練時間。
* **損失權重：**$\lambda=1,\ \gamma=0.5\sim 1.0$ 起步；與 CLE-ViT 的多目標配比一致，避免單邊主導【】。
* \*\*整體複雜度：\*\*主要成本為 $P$ 個 $d\times d$ 的本徵分解（$O(Pd^3)$），在 $P\le 8,d\le 64$ 時對 224–448 輸入解析度是可行的。
* \*\*推論時開銷：\*\*部位模組可保留（提升穩健）或僅用全局 head；訓練時的對比支路可像 CLE-ViT 一樣於推論時移除【】。

---

## 評測建議（與資料對齊）

* **UFG 五個子集**（Cotton80, SoyLocal, SoyGene, SoyAgeing, SoyGlobal）【】；這些資料明確呈現「小跨類、大內類」的特性，正是本法瞄準的痛點【】。
* **對照方法**：

  1. 相同 backbone + CE；
  2. class-level SupCon；
  3. CLE-ViT 的 instance-level triplet 版本（歐氏）；
  4. **本法 PAC-MCL（LE / BW）**。
* **可視化**：t-SNE/UMAP 於**部位-流形距離**誘導的核上；預期呈現**跨類距離與內類分散**同時增大，類似 CLE-ViT 的觀察但更穩健【】。

---

## 潛在風險與緩解

* **數值不穩（特徵值接近 0）**：$+\epsilon I$、特徵白化、降維；優先用 **Log-Euclidean** 訓練，再切換 **Bures** 微調。
* **部位對齊錯配**：先以雙向最近鄰對齊，再用匈牙利法做全局修正；或加入**跨視圖稀疏一致性**正則。
* **硬負樣本昂貴**：採 **top-k** 近似與溫度縮放，並結合記憶庫。

---

## 與現有工作的連結（並行、非重疊）

* CLE-ViT 證明**影像內部**的自監督對比（遮罩＋洗牌）在超細粒度有效【】【】；我們**保留其資料增強與正對構造**，但把度量從**歐氏空間**換到**SPD/Bures 流形**，且**下沉到部位層級**，屬**目標函數與度量幾何**的創新。
* UFG 資料的特性（**大內類／小跨類**）正是本法在幾何上能受益的場景【】。

---

## 摘要（50–70 行內可直接投稿版）

> 我們提出 PAC-MCL，一種**部位感知的流形對比損失**，用於超細粒度分類。方法首先在無標註情況下，透過遮罩與洗牌增強生成影像內的正對視圖，並由注意力引導聚合得到部位級特徵。接著以**二階統計**將部位嵌入 **SPD/Bures 流形**，在流形上對齊正對部位並分離負對部位，構成新的 **triplet-on-manifold** 目標，與分類損失共同訓練。理論上，我們證明在 Log-Euclidean 近似下損失具 geodesic-convex 上界，並給出零損條件與收斂性分析。實作上，PAC-MCL 可即插即用於 **timm** 全系列 **CNN**。在 **UFG** 基準上，該任務的「小跨類、大內類」特性使流形度量顯著優於歐氏對比學習，並維持對內類變異的容忍度（與 CLE-ViT 的觀察一致【】【】）。

---

如果你要，我可以把**論文骨架（LaTeX overleaf 結構＋伪代碼＋理論附錄）**與**timm 最小可跑的 PyTorch 原型**（含 SPD/Bures 距離與穩定化技巧）也一起產出。要不要我先幫你把 **method 圖**與 **loss 流程圖**畫好？
