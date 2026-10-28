---
title: "Hyperspectral Image Classification: MMFormer vs. R2Mamba"
subtitle: "Benchmarking 2026 Vision Transformers against Selective State Space Models"
author: "Prince Prajapati (Lead), Team Members | Dept. of Computer Science & Engineering"
date: "Academic Year 2026"
---

# Overview & Agenda

- **1. Background & Motivation** — What is Hyperspectral Imaging (HSI)?
- **2. Architectural Evolution** — CNNs $\rightarrow$ Transformers $\rightarrow$ Mamba
- **3. Problem Formulation** — Why compare 2026 architectures?
- **4. Methodology & Models** — MMFormer (*IEEE TGRS*) vs. R2Mamba (*IEEE JSTARS*)
- **5. Experimental Setup** — Datasets, sample budgets, protocol
- **6. Quantitative Results** — OA, AA, Kappa (Mean $\pm$ Std across 3 seeds)
- **7. Qualitative Analysis** — Full-scene classification maps
- **8. Computational Efficiency** — Linear $O(N)$ vs Quadratic $O(N^2)$
- **9. Conclusion & Next Steps**

---

# 1. Background: Hyperspectral Imaging (HSI)

- **Beyond Standard RGB:**
  - Regular photos capture 3 broad bands (Red, Green, Blue).
  - Hyperspectral sensors capture **100–200+ contiguous narrow spectral bands** across visible and infrared spectrums.
- **The Spectral Signature:**
  - Each pixel contains a continuous reflectance curve revealing the physical and chemical composition of materials.
- **Critical Real-World Applications:**
  - **Precision Agriculture:** Crop health monitoring, weed identification, soil moisture analysis.
  - **Urban Planning:** Distinguishing asphalt, concrete, roofing materials, and water bodies.
  - **Environmental & Planetary:** Deforestation tracking, mineral exploration, Mars exploration.

---

# 2. Architectural Evolution in HSI

| Era | Paradigm | Representative Models | Core Limitations |
|---|---|---|---|
| **2017–2020** | **2D / 3D CNNs** | SSRN, HybridSN, DBDA | Local receptive fields; struggles to capture long-range spectral relationships |
| **2021–2024** | **Vision Transformers** | SpectralFormer, MFT, GAHT | Captures global context via self-attention, but suffers from **quadratic complexity $O(N^2)$** |
| **2024–2026** | **State Space Models (SSMs)** | MambaHSI, **R2Mamba** | Combines global receptive fields with **linear complexity $O(N)$** via selective memory states |

---

# 3. Problem Statement & Research Objectives

- **Research Gap:**
  - Modern remote sensing requires classifying complex scenes under **extremely limited training samples** (e.g., only 10–30 labeled pixels per class).
  - While Transformers have been dominant, 2026 introduces cutting-edge **Macro-Micro Transformers** and **Route-Reliability State Space Models**.
- **Our Objectives:**
  1. Conduct a rigorous, head-to-head empirical benchmark comparing the two newest 2026 models under **identical experimental conditions**.
  2. Implement a statistically robust protocol using **3 independent seeded runs** (reporting Mean $\pm$ Standard Deviation).
  3. Analyze both **quantitative accuracy** and **computational efficiency** (training latency, stability).

---

# 4. Competing 2026 Architectures

### MMFormer (*IEEE TGRS 2026*)
- **Macro-Micro Transformer** designed specifically for HSI.
- **Macro Branch:** Learns coarse, global contextual semantics across broad spatial tokens.
- **Micro Branch:** Focuses on fine-grained spectral-spatial details.
- **Cross-Attention Fusion:** Interactively exchanges information between macro and micro scales.

### R2Mamba (*IEEE JSTARS 2026*)
- **Route-Reliability Mamba** built on Selective State Space Models.
- **Continuous State Transition:** Scans spatial-spectral tokens sequentially with linear complexity $O(N)$.
- **Route Reliability Gating:** Automatically assigns reliability weights to scanning paths, downweighting boundary/noisy pixels.

---

# 5. Phase 1: Pipeline Validation

- Before executing resource-intensive 2026 models, we validated the end-to-end framework using **5 baseline architectures**:
  - `HybridSN` (3D-2D CNN)
  - `SSRN` (Spectral-Spatial ResNet)
  - `pResNet` (Pyramidal ResNet)
  - `DBDA` (Dual-branch Attention)
  - `SpectralFormer` (Baseline Transformer)
- **Outcome:**
  - Verified on-the-fly memory-efficient patch extraction.
  - Confirmed automated HuggingFace dataset caching and PCA preprocessing.
  - Validated early stopping and visualization map generation.

---

# 6. Benchmark Datasets & Preprocessing

| Dataset | Sensor | Scene Type | Spatial Dimensions | Spectral Bands (Raw $\rightarrow$ PCA) | Target Classes |
|---|---|---|:---:|:---:|:---:|
| **Indian Pines** | NASA AVIRIS | Agricultural / Rural | $145 \times 145$ | $200 \rightarrow 30$ | 16 Crop Classes |
| **Pavia University** | DLR ROSIS | Dense Urban Infrastructure | $610 \times 340$ | $103 \rightarrow 30$ | 9 Urban Classes |

- **Preprocessing Pipeline:**
  - **PCA Reduction:** Reduced to 30 principal components (captures $>99\%$ spectral variance while eliminating noise & redundancy).
  - **Spatial Window:** Fixed $11 \times 11$ patch size centered on each target pixel.

---

# 7. Experimental Protocol

- **Few-Shot Sample Budget:**
  - *Indian Pines:* 10 training / 5 validation pixels per class (remaining $\sim 9,482$ pixels for testing).
  - *Pavia University:* 30 training / 10 validation pixels per class (remaining $\sim 40,044$ pixels for testing).
- **Optimization & Hyperparameters:**
  - Optimizer: **Adam** ($\text{learning rate} = 0.001$, $\text{batch size} = 64$).
  - Max Epochs: 80 with **Early Stopping Patience = 12**.
  - Hardware: NVIDIA Tesla T4 GPU (accelerated with custom CUDA kernels for Mamba).
- **Statistical Rigor:**
  - 3 independent seeded runs (`seeds: [1, 2, 3]`).
  - Evaluated on: **Overall Accuracy (OA)**, **Average Accuracy (AA)**, and **Kappa Coefficient ($\kappa$)**.

---

# 8. Results: Indian Pines Benchmark (16 Classes)

| Model | OA (%) | AA (%) | Kappa ($\times 100$) | Best Run OA | Avg. Train Time |
|---|:---:|:---:|:---:|:---:|:---:|
| **MMFormer** *(2026 Transformer)* | $76.41 \pm 3.84$ | $87.10 \pm 1.96$ | $73.41 \pm 4.28$ | $80.81\%$ | ~75s / run |
| **R2Mamba** *(2026 Mamba / SSM)* | **$\mathbf{83.89 \pm 1.66}$** | **$\mathbf{91.29 \pm 1.09}$** | **$\mathbf{81.71 \pm 1.87}$** | **$\mathbf{85.66\%}$** | **~26s / run** |
| **Performance Delta** | **+7.48%** | **+4.19%** | **+8.30** | **+4.85%** | **~3x Faster** |

- **Key Observations:**
  - `R2Mamba` outperforms `MMFormer` by **+7.48% in Overall Accuracy**.
  - `R2Mamba` exhibits significantly lower standard deviation ($\pm 1.66\%$ vs $\pm 3.84\%$), indicating superior training stability on small agricultural samples.

---

# 9. Results: Pavia University Benchmark (9 Classes)

| Model | OA (%) | AA (%) | Kappa ($\times 100$) | Best Run OA | Avg. Train Time |
|---|:---:|:---:|:---:|:---:|:---:|
| **MMFormer** *(2026 Transformer)* | $92.45 \pm 5.98$ | $95.90 \pm 1.80$ | $90.44 \pm 7.36$ | $96.80\%$ | ~140s / run |
| **R2Mamba** *(2026 Mamba / SSM)* | **$\mathbf{97.59 \pm 0.60}$** | **$\mathbf{97.95 \pm 0.53}$** | **$\mathbf{96.86 \pm 0.77}$** | **$\mathbf{98.03\%}$** | **~40s / run** |
| **Performance Delta** | **+5.14%** | **+2.05%** | **+6.42** | **+1.23%** | **~3.5x Faster** |

- **Key Observations:**
  - `R2Mamba` reaches **$98.03\%$ peak accuracy** with an ultra-consistent variance of only **$\pm 0.60\%$**.
  - `MMFormer` experiences higher variance ($\pm 5.98\%$) due to the quadratic sensitivity of multi-head self-attention on urban edges.

---

# 10. Qualitative Visual Analysis: Classification Maps

- **Arranged Best-Run Maps (Pavia University & Indian Pines):**
  - **Noise Reduction:** `R2Mamba` produces cohesive, smooth regions with minimal salt-and-pepper artifacts.
  - **Boundary Precision:** Route-reliability gating preserves sharp boundaries between adjacent fields and road corridors.
  - **Minority Class Recognition:** High Average Accuracy ($>91\%$ and $>97\%$) confirms balanced recognition across rare classes.

*(Full high-resolution maps exported in `results/classification_maps/`)*

---

# 11. Computational Complexity & Efficiency

| Architectural Aspect | MMFormer (Vision Transformer) | R2Mamba (State Space Model) |
|---|---|---|
| **Theoretical Complexity** | **Quadratic $O(N^2)$** | **Linear $O(N)$** |
| **GPU Memory Footprint** | Scales quadratically with patch/tokens | Constant memory per recurrent step |
| **Convergence Speed** | Requires more epochs to stabilize attention | Converges rapidly within 25–40 epochs |
| **Training Latency** | ~75s – 140s per run | **~26s – 40s per run (3–3.5x faster)** |
| **Inference Scalability** | Heavy full-image test inference | Ultra-fast sequential scan across large scenes |

---

# 12. Summary & Key Contributions

- **1. Systematic Benchmark:** Established the first rigorous comparison between 2026 SOTA Transformer and Mamba architectures under identical protocols.
- **2. Mamba Superiority:** Demonstrated that selective State Space Models (`R2Mamba`) deliver **higher accuracy (+5% to +7.5% OA)**, **greater stability**, and **3.5x faster execution**.
- **3. Fully Reproducible:** Open-source GitHub repository with automated YAML configs, seeded runs, and complete LaTeX/map artifacts.
- **4. Modular Design:** Readily extensible to additional datasets, patch scales, and multimodal fusion.

---

# 13. Future Work & Extensions

- **Scalability to UAV Datasets:** Evaluating performance on ultra-high-resolution drone datasets (WHU-Hi series).
- **Multimodal Remote Sensing:** Extending `R2Mamba` to fuse Hyperspectral cubes with LiDAR height elevation maps.
- **Model Compression:** Exploring knowledge distillation to deploy lightweight State Space Models directly on satellite edge hardware.

---

# Q & A

### Thank You!

**Project Repository:**  
[github.com/lPrincel/HSI-Classification-Minor-Project](https://github.com/lPrincel/HSI-Classification-Minor-Project)

*Questions and Feedback are welcome!*

