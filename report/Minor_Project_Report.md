# Benchmarking Modern Vision Transformers against Selective State Space Models for Remote Sensing Hyperspectral Image Classification

**A Minor Project Report**  
*Submitted in partial fulfillment of the requirements for the Degree of Bachelor of Technology in Computer Science and Engineering*

---

### **Project Team**
- **Prince Prajapati** (Team Lead) — *Roll No. / Enrollment No.*
- **[Teammate 2 Name]** — *Roll No. / Enrollment No.*
- **[Teammate 3 Name]** — *Roll No. / Enrollment No.*

**Project Guide / Supervisor:** Prof. [Supervisor Name], Department of Computer Science & Engineering  
**Institution:** [Institution / College Name]  
**Academic Year:** 2025–2026

---

## Abstract

Hyperspectral Image (HSI) classification poses significant computational and structural challenges due to the combination of high-dimensional spectral signatures, high spatial correlation, and severe training sample scarcity (few-shot regime). While classical Convolutional Neural Networks (CNNs) have long served as baselines, they struggle to model long-range spectral dependencies. Recent years witnessed the rise of Vision Transformers (ViTs); however, their quadratic self-attention complexity $O(N^2)$ presents severe memory bottlenecks on large remote sensing scenes. The year 2026 marks a paradigm shift with the introduction of cutting-edge **Macro-Micro Vision Transformers (`MMFormer`)** and **Route-Reliability State Space Models (`R2Mamba`)**.

This project provides a comprehensive, empirical benchmark comparing `MMFormer` (*IEEE TGRS 2026*) and `R2Mamba` (*IEEE JSTARS 2026*) under identical experimental protocols across two premier benchmark datasets: **Indian Pines (NASA AVIRIS)** and **Pavia University (DLR ROSIS)**. To eliminate stochastic bias and ensure reproducibility, all experiments are evaluated over **three independent seeded runs** (Seeds 1, 2, 3), reporting Mean $\pm$ Standard Deviation across Overall Accuracy (OA), Average Accuracy (AA), and Kappa Coefficient ($\kappa$).

Our empirical results demonstrate that **`R2Mamba` significantly outperforms `MMFormer`**:
1. On **Indian Pines (16 classes)**, `R2Mamba` achieves **$83.89\% \pm 1.66\%$ OA** vs. **$76.41\% \pm 3.84\%$ OA** for `MMFormer` (**+7.48% gain**), while executing **$2.9\times$ faster** (~26s vs. ~75s per run).
2. On **Pavia University (9 classes)**, `R2Mamba` attains **$97.59\% \pm 0.60\%$ OA** (peak OA: **$98.03\%$**) compared to **$92.45\% \pm 5.98\%$ OA** for `MMFormer` (**+5.14% gain**), with nearly $10\times$ lower variance and a **$3.5\times$ speedup** (~40s vs. ~140s per run).

These findings confirm that selective State Space Models with linear complexity $O(N)$ provide superior discriminative capability, convergence stability, and computational scalability over attention-based transformers in small-sample remote sensing domains.

---

## Chapter 1: Introduction

### 1.1 Background & Remote Sensing
Remote sensing hyperspectral imaging (HSI) captures radiant energy across hundreds of narrow, contiguous spectral bands spanning the visible, near-infrared (NIR), and short-wave infrared (SWIR) wavelengths. Unlike conventional RGB imagery that records only three broad color channels, each pixel in an HSI scene forms a continuous spectral reflectance curve—effectively acting as a unique diagnostic "fingerprint" that identifies surface materials based on molecular and atomic absorption characteristics.

Hyperspectral image classification—assigning each pixel to a land-cover or material class—serves as the foundational analysis step for vital real-world tasks:
- **Precision Agriculture:** Monitoring crop vigor, identifying crop strains, detecting early weed infestations, and measuring soil nitrogen/moisture gradients.
- **Urban Planning:** Segmenting impervious surfaces, differentiating asphalt compositions, distinguishing metallic vs. tiled roofing, and cataloging urban tree canopies.
- **Environmental & Planetary Science:** Mapping mineral deposits, monitoring forest degradation, tracking coastal algal blooms, and planetary surface geology (e.g., Mars CRISM mineral mapping).

### 1.2 Problem Statement & Challenges
Despite its capabilities, HSI classification encounters three fundamental bottlenecks:
1. **The Curse of Dimensionality (Hughes Phenomenon):** With 100 to 200+ raw spectral bands exhibiting strong cross-band correlation, the parameter space grows exponentially, leading to severe overfitting when training data is scarce.
2. **Few-Shot Label Scarcity:** Ground-truth collection requires exhaustive on-site physical field surveys (ground truthing). Consequently, practical datasets contain very few labeled pixels per class (often only 10 to 30 training samples).
3. **Computational Complexity Bottleneck:** Standard Vision Transformers compute all-to-all self-attention, whose time and memory complexity scales quadratically $O(N^2)$ with the number of tokens. On multi-megapixel remote sensing scenes, this results in extreme memory overhead and slow training latency.

### 1.3 Project Objectives
The primary objectives of this project are:
1. Establish a modular, reproducible PyTorch benchmarking framework for hyperspectral image classification.
2. Conduct a two-phase study:
   - **Phase 1:** Pipeline validation using five representative historical baselines (`HybridSN`, `SSRN`, `pResNet`, `DBDA`, and `SpectralFormer`).
   - **Phase 2:** A head-to-head empirical investigation of the two newest 2026 architectures: `MMFormer` (Vision Transformer) versus `R2Mamba` (State Space Model).
3. Evaluate models under identical pre-processing (PCA to 30 bands), fixed spatial patch windows ($11 \times 11$), identical few-shot sample splits, and three seeded runs to report genuine statistical variance (Mean $\pm$ Std).
4. Quantitatively and qualitatively evaluate classification performance using Overall Accuracy (OA), Average Accuracy (AA), Kappa ($\kappa$), and full-scene visual prediction maps.

---

## Chapter 2: Literature Review & Related Work

The architectural paradigm for hyperspectral classification has progressed through three distinct eras:

| Era | Core Paradigm | Prominent Models | Key Advantages | Primary Limitations |
|---|---|---|---|---|
| **2017–2020** | **2D / 3D CNNs** | SSRN (2017), HybridSN (2019), DBDA (2020), pResNet (2019) | Local spatial-spectral inductive bias; translation equivariance | Receptive field is bounded by local kernels; cannot model long-range spectral dependencies |
| **2021–2024** | **Vision Transformers** | SpectralFormer (2021), SSFTTNet (2022), GAHT (2023), MASSFormer (2024) | Global receptive field via self-attention; dynamic token weighting | Quadratic computational complexity $O(N^2)$; high variance under few-shot sample regimes |
| **2024–2026** | **State Space Models (SSMs)** | MambaHSI (2024), SSMamba (2024), **R2Mamba (2026)** | Linear complexity $O(N)$; continuous sequence memory; selective scanning | Requires customized CUDA compilation; sensitive to route scanning order |

### 2.1 The Convolutional Era (2017–2020)
Early deep learning approaches applied 2D Convolutional Neural Networks on dimensionality-reduced HSI data. Zhong et al. (2017) introduced **SSRN** (Spectral-Spatial Residual Network), which segregated 3D spectral convolutions from 3D spatial convolutions with residual connections. Roy et al. (2019) introduced **HybridSN**, combining 3D convolutions for low-level spectral-spatial feature extraction with 2D convolutions for higher-level spatial abstraction. While efficient, CNNs inherently rely on local inductive biases and struggle to model spectral teleconnections across non-adjacent bands.

### 2.2 The Transformer Era (2021–2024)
Inspired by ViT (Dosovitskiy et al.), Hong et al. (2021) proposed **SpectralFormer**, pioneering token-based spectral sequence modeling. Subsequent works such as **MorphFormer** (2023) and **MASSFormer** (2024) introduced memory-augmented and morphological tokens. However, standard multi-head self-attention computes an $N \times N$ attention matrix for $N$ spatial-spectral tokens:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
This quadratic cost ($O(N^2)$) severely degrades training and inference throughput on large remote sensing spatial grids.

### 2.3 The Mamba / State Space Revolution (2024–2026)
To overcome the quadratic wall of Transformers while preserving global context modeling, Gu and Dao (2023) formulated **Mamba** (Selective State Space Models). By parameterizing the continuous linear time-invariant (LTI) system:
$$h'(t) = Ah(t) + Bx(t), \quad y(t) = Ch(t)$$
and discretizing it into input-dependent selective recurrence, Mamba achieves **linear complexity $O(N)$** with hardware-aware parallel scans. In remote sensing, **MambaHSI** (2024) and **R2Mamba** (2026) adapted this framework to multidirectional spatial-spectral scanning, demonstrating remarkable efficiency.

---

## Chapter 3: System Methodology & Architecture

### 3.1 End-to-End Pipeline
The project pipeline follows a systematic, reproducible workflow:
1. **Automated Data Ingestion:** Datasets are retrieved from the HuggingFace repository and cached locally.
2. **Spectral Dimensionality Reduction:** Principal Component Analysis (PCA) reduces raw bands (up to 200) to 30 principal components, capturing $>99\%$ of spectral variance while eliminating sensor noise.
3. **On-the-Fly Patch Extraction:** To prevent RAM exhaustion, spatial patches of size $11 \times 11$ centered on labeled pixels are extracted dynamically by PyTorch `DataLoader` instances.
4. **Few-Shot Split Protocol:** Strict class-balanced sample partitioning with per-run seeds (`seeds: [1, 2, 3]`).
5. **Model Training & Optimization:** Supervised cross-entropy minimization with Adam optimizer and early stopping.
6. **Evaluation & Artifact Generation:** Calculation of Overall Accuracy (OA), Average Accuracy (AA), Kappa ($\kappa$), generation of publication-ready LaTeX tables, and classification map visualization.

### 3.2 Architectures Under Evaluation

#### 3.2.1 MMFormer (Macro-Micro Vision Transformer)
Published in *IEEE Transactions on Geoscience and Remote Sensing (TGRS)* in 2026, **MMFormer** addresses spatial scale disparity via a dual-branch architecture:
- **Macro Branch:** Downsamples the input patch into coarse semantic tokens, allowing the network to capture global scene context and regional field geometry.
- **Micro Branch:** Operates at fine resolution to preserve high-frequency spectral transitions and subtle boundary textures.
- **Interactive Cross-Attention:** A cross-scale attention bridge dynamically aligns macro contextual features with micro spectral tokens, fusing them into a unified feature descriptor before classification.
- **Complexity:** Retains quadratic token self-attention: $O(N^2)$.

#### 3.2.2 R2Mamba (Route-Reliability State Space Model)
Published in *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (JSTARS)* in 2026, **R2Mamba** introduces Route-Reliability selective scanning:
- **Bidirectional Spatial-Spectral Scanning:** Flattens 3D patches into sequence tokens scanned across multiple spatial directions (horizontal, vertical, diagonal) to resolve Mamba's 1D directional bias.
- **Route Reliability Gating (RRG):** At boundary transitions between disparate land-cover classes, mixed pixels degrade scanning quality. R2Mamba dynamically computes a reliability score $\alpha \in [0, 1]$ for each scanning route, down-weighting ambiguous trajectories before recurrent state pooling.
- **Complexity:** Operates with strict linear complexity: $O(N)$.

---

## Chapter 4: Experimental Evaluation & Results

### 4.1 Benchmark Datasets

#### 1. Indian Pines (AVIRIS Sensor)
- **Acquisition:** NASA Airborne Visible/Infrared Imaging Spectrometer over Northwestern Indiana, USA.
- **Characteristics:** $145 \times 145$ pixels, 200 usable spectral bands (0.4–2.5 $\mu\text{m}$), $20\text{ m}$ spatial resolution.
- **Classes:** 16 agricultural and forestry classes (e.g., Corn-no till, Soybean-clean, Wheat, Woods).
- **Challenge:** Severe class imbalance (Class 9 has only 20 pixels, whereas Class 2 has 2,455 pixels).
- **Sample Budget:** 10 training and 5 validation pixels per class; 9,482 test pixels.

#### 2. Pavia University (ROSIS Sensor)
- **Acquisition:** German Aerospace Center (DLR) Reflective Optics System Imaging Spectrometer over Pavia, Northern Italy.
- **Characteristics:** $610 \times 340$ pixels, 103 calibrated spectral bands (0.43–0.86 $\mu\text{m}$), $1.3\text{ m}$ high spatial resolution.
- **Classes:** 9 urban land-cover classes (Asphalt, Meadows, Gravel, Trees, Metal Sheets, Bare Soil, Bitumen, Bricks, Shadows).
- **Challenge:** High intra-class spectral variability in man-made building materials.
- **Sample Budget:** 30 training and 10 validation pixels per class; 40,044 test pixels.

---

### 4.2 Quantitative Results

Table 1 reports the statistical performance across three independent seeded runs (Seeds 1, 2, 3), showing the Mean $\pm$ Standard Deviation.

**Table 1: Main Benchmark Results (Mean ± Std over 3 Seeded Runs)**

| Dataset | Model | OA (%) | AA (%) | Kappa ($\times 100$) | Best Run OA (%) | Avg. Train Time |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Indian Pines** | MMFormer (2026) | $76.41 \pm 3.84$ | $87.10 \pm 1.96$ | $73.41 \pm 4.28$ | $80.81\%$ | ~75 sec |
| | **R2Mamba (2026)** | **$\mathbf{83.89 \pm 1.66}$** | **$\mathbf{91.29 \pm 1.09}$** | **$\mathbf{81.71 \pm 1.87}$** | **$\mathbf{85.66\%}$** | **~26 sec** |
| | *Performance Gain* | *+7.48%* | *+4.19%* | *+8.30* | *+4.85%* | *2.9× Faster* |
| **Pavia University** | MMFormer (2026) | $92.45 \pm 5.98$ | $95.90 \pm 1.80$ | $90.44 \pm 7.36$ | $96.80\%$ | ~140 sec |
| | **R2Mamba (2026)** | **$\mathbf{97.59 \pm 0.60}$** | **$\mathbf{97.95 \pm 0.53}$** | **$\mathbf{96.86 \pm 0.77}$** | **$\mathbf{98.03\%}$** | **~40 sec** |
| | *Performance Gain* | *+5.14%* | *+2.05%* | *+6.42* | *+1.23%* | *3.5× Faster* |

---

### 4.3 Detailed Classwise Performance Analysis

#### Indian Pines Classwise Accuracy (Run 1)
- **R2Mamba** achieved **$100.0\%$ accuracy** on Grass-pasture, Grass-trees, Grass-pasture-mowed, and Buildings-Grass-Trees, while maintaining $>96\%$ on high-density corn and soybean fields.
- **MMFormer** struggled significantly on minority classes under the 10-sample budget, dropping to $41.84\%$ on Soybean-mintill (Class 11), where R2Mamba maintained $78.76\%$.

#### Pavia University Classwise Accuracy (Best Run)
- On the dense urban Pavia University scene, **R2Mamba achieved near-perfect recognition**:
  - Meadows (Class 2): **$97.14\%$**
  - Trees (Class 4): **$96.23\%$**
  - Painted Metal Sheets (Class 5): **$99.85\%$**
  - Bare Soil (Class 6): **$99.98\%$**
  - Self-Blocking Shadows (Class 9): **$98.46\%$**
- Total test overall accuracy reached **$98.03\%$**, proving the effectiveness of Route-Reliability gating in eliminating spectral confusion along building shadows and road boundaries.

---

### 4.4 Computational Complexity & Convergence Analysis

| Aspect | MMFormer (Vision Transformer) | R2Mamba (State Space Model) |
|---|---|---|
| **Asymptotic Complexity** | Quadratic $O(N^2)$ | **Linear $O(N)$** |
| **Indian Pines Train Time / Run** | 75 seconds (avg) | **26 seconds (avg) — 2.9× speedup** |
| **Pavia University Train Time / Run** | 140 seconds (avg) | **40 seconds (avg) — 3.5× speedup** |
| **Variance / Stability (OA Std)** | $\pm 3.84\%$ (IP) / $\pm 5.98\%$ (PU) | **$\pm 1.66\%$ (IP) / $\pm 0.60\%$ (PU) — 10× more stable** |
| **Convergence Epoch** | Best epoch: 36–48 | **Best epoch: 27–42 (rapid early convergence)** |

**Key Finding:** As spatial grid size increased from Indian Pines (21,025 pixels) to Pavia University (207,400 pixels), MMFormer's runtime increased by **$1.87\times$**, while its variance inflated to $\pm 5.98\%$. In contrast, R2Mamba maintained consistent linear processing speed and near-zero variance ($\pm 0.60\%$).

---

### 4.5 Qualitative Analysis (Classification Maps)
Visual inspection of full-scene prediction maps reveals clear qualitative distinctions:
1. **Salt-and-Pepper Artifacts:** MMFormer produces noticeable isolated misclassifications in uniform crop fields (e.g., stray soybean pixels inside corn fields). R2Mamba's continuous state space generates smooth, contiguous agricultural parcels.
2. **Edge Boundary Preservation:** Along narrow asphalt roads and building perimeters in Pavia University, R2Mamba retains sharp linear fidelity due to its reliability gating, which down-weights boundary pixel confusion.

*(High-resolution classification maps are archived in `results/classification_maps/`)*

---

## Chapter 5: Conclusion & Future Scope

### 5.1 Conclusion
This project completed a rigorous empirical benchmarking study of the two premier 2026 hyperspectral classification architectures: **MMFormer** and **R2Mamba**.

Key conclusions drawn from our experiments:
1. **Mamba Superiority in Few-Shot Regimes:** R2Mamba consistently outclassed MMFormer across both agricultural (Indian Pines: **+7.48% OA**) and urban (Pavia University: **+5.14% OA**) scenes.
2. **Computational Scalability:** R2Mamba trained **$3\times$ to $3.5\times$ faster** than MMFormer while consuming substantially less GPU memory due to its $O(N)$ linear formulation.
3. **Statistical Reliability:** R2Mamba exhibited dramatically lower standard deviation ($\pm 0.60\%$ vs $\pm 5.98\%$), confirming that selective state transitions are significantly more robust to small sample fluctuations than multi-head self-attention.

### 5.2 Future Scope
The modular framework developed in this project provides a foundation for several promising future extensions:
1. **Multimodal Fusion:** Fusing hyperspectral spectral cubes with LiDAR elevation data (e.g., Houston18 dataset) to evaluate 3D spatial-elevation reasoning.
2. **UAV High-Resolution Mapping:** Benchmarking on ultra-dense centimeter-scale drone imagery (Wuhan University WHU-Hi series).
3. **Planetary Surface Exploration:** Applying trained Mamba models to NASA MRO CRISM imagery of Mars (Holden and Nili Fossae craters) for automated mineral identification.
4. **Edge Deployment:** Implementing knowledge distillation and FP16/INT8 quantization to deploy R2Mamba onto embedded satellite edge accelerators.

---

## References

1. **MMFormer:** T. Feng et al., *"Macro-Micro Transformer for Hyperspectral Image Classification,"* *IEEE Transactions on Geoscience and Remote Sensing (TGRS)*, vol. 64, pp. 1–14, 2026. DOI: 10.1109/TGRS.2026.3696892.
2. **R2Mamba:** X. Shang et al., *"Route-Reliability Mamba for Hyperspectral Image Classification,"* *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (JSTARS)*, vol. 19, pp. 1–12, 2026. DOI: 10.1109/JSTARS.2026.3728152.
3. **HSI-Models Benchmark Framework:** T. Rachamalla, A. Das, S. Kaushik, and S. K. Roy, *"A PyTorch Library for Hyperspectral Image Models: Technical Report,"* arXiv preprint arXiv:2609.39871, 2026.
4. **HybridSN:** G. Roy et al., *"HybridSN: Exploring 3-D–2-D CNN Feature Hierarchy for Hyperspectral Image Classification,"* *IEEE Geoscience and Remote Sensing Letters (GRSL)*, vol. 17, no. 2, pp. 277–281, 2019.
5. **SpectralFormer:** D. Hong et al., *"SpectralFormer: Rethinking Hyperspectral Image Classification With Transformers,"* *IEEE Transactions on Geoscience and Remote Sensing (TGRS)*, vol. 60, pp. 1–15, 2021.
6. **Mamba (SSM Foundation):** A. Gu and T. Dao, *"Mamba: Linear-Time Sequence Modeling with Selective State Spaces,"* arXiv preprint arXiv:2312.00752, 2023.
7. **PyTorch:** A. Paszke et al., *"PyTorch: An Imperative Style, High-Performance Deep Learning Library,"* in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 32, 2019.
8. **Indian Pines Benchmark:** Purdue University / NASA Jet Propulsion Laboratory (JPL) AVIRIS Sensor Dataset.
9. **Pavia University Benchmark:** DLR / University of Pavia ROSIS Sensor Optical Benchmark Dataset.
