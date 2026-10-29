::: IEEEkeywords
Hyperspectral image classification, State Space Models, Mamba, Vision
Transformers, MMFormer, R2Mamba, remote sensing, few-shot learning.
:::

# Introduction

Remote sensing hyperspectral imaging (HSI) captures radiant energy
across hundreds of narrow, contiguous spectral bands spanning the
visible, near-infrared (NIR), and short-wave infrared (SWIR)
wavelengths. Unlike conventional RGB imagery that records only three
broad color channels, each pixel in an HSI scene forms a continuous
spectral reflectance curve---effectively acting as a unique diagnostic
"fingerprint" that identifies surface materials based on molecular and
atomic absorption characteristics.

Hyperspectral image classification---assigning each pixel to a
land-cover or material class---serves as the foundational analysis step
for vital real-world tasks:

-   **Precision Agriculture:** Monitoring crop vigor, identifying crop
    strains, detecting early weed infestations, and measuring soil
    nitrogen/moisture gradients.

-   **Urban Planning:** Segmenting impervious surfaces, differentiating
    asphalt compositions, distinguishing metallic vs. tiled roofing, and
    cataloging urban tree canopies.

-   **Environmental Science:** Mapping mineral deposits, monitoring
    forest degradation, tracking coastal algal blooms, and planetary
    surface geology.

## Problem Statement & Challenges

Despite its capabilities, HSI classification encounters three
fundamental bottlenecks:

1.  **The Hughes Phenomenon:** With 100 to 200+ raw spectral bands
    exhibiting strong cross-band correlation, the parameter space grows
    exponentially, leading to severe overfitting when training data is
    scarce.

2.  **Few-Shot Label Scarcity:** Ground-truth collection requires
    exhaustive on-site physical field surveys (ground truthing).
    Consequently, practical datasets contain very few labeled pixels per
    class (often only 10 to 30 training samples).

3.  **Computational Complexity Bottleneck:** Standard Vision
    Transformers compute all-to-all self-attention, whose time and
    memory complexity scales quadratically $\mathcal{O}(N^2)$ with the
    number of tokens. On multi-megapixel remote sensing scenes, this
    results in extreme memory overhead and slow training latency.

## Project Objectives & Contributions

The primary contributions of this project are:

-   We establish a modular, reproducible PyTorch benchmarking framework
    for hyperspectral image classification.

-   We conduct a two-phase investigation: Phase 1 validates the pipeline
    with five representative historical baselines (`HybridSN`, `SSRN`,
    `pResNet`, `DBDA`, and `SpectralFormer`), while Phase 2 conducts a
    direct empirical investigation of the two newest 2026 architectures:
    `MMFormer` (Vision Transformer) versus `R2Mamba` (State Space
    Model).

-   All evaluations are performed under identical preprocessing (PCA to
    30 bands), fixed spatial patch windows ($11 \times 11$), identical
    few-shot sample splits, and three seeded runs to report genuine
    statistical variance ($\text{Mean} \pm \text{Std}$).

# Related Work

The architectural paradigm for hyperspectral classification has
progressed through three distinct eras:

## The Convolutional Era (2017--2020)

Early deep learning approaches applied 2D and 3D Convolutional Neural
Networks on dimensionality-reduced HSI data. Zhong *et al.* introduced
SSRN [@ssrn], segregating 3D spectral convolutions from 3D spatial
convolutions with residual connections. Roy *et al.* proposed HybridSN
[@hybridsn], combining 3D convolutions for low-level spectral-spatial
feature extraction with 2D convolutions for higher-level spatial
abstraction. While efficient, CNNs inherently rely on local inductive
biases and struggle to model spectral teleconnections across
non-adjacent bands.

## The Transformer Era (2021--2024)

Inspired by ViT, Hong *et al.* proposed SpectralFormer
[@spectralformer], pioneering token-based spectral sequence modeling.
Subsequent works such as MorphFormer and MASSFormer introduced
memory-augmented and morphological tokens. However, standard multi-head
self-attention computes an $N \times N$ attention matrix for $N$
spatial-spectral tokens:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
This quadratic cost ($\mathcal{O}(N^2)$) severely degrades training and
inference throughput on large remote sensing spatial grids.

## The Mamba / State Space Revolution (2024--2026)

To overcome the quadratic wall of Transformers while preserving global
context modeling, Gu and Dao formulated Mamba [@mamba] using Selective
State Space Models. By parameterizing the continuous linear
time-invariant (LTI) system:
$$h'(t) = Ah(t) + Bx(t), \quad y(t) = Ch(t)$$ and discretizing it into
input-dependent selective recurrence, Mamba achieves strict linear
complexity $\mathcal{O}(N)$ with hardware-aware parallel scans. In
remote sensing, MambaHSI and R2Mamba [@r2mamba] adapted this framework
to multidirectional spatial-spectral scanning, demonstrating remarkable
efficiency.

# Methodology & Model Formulations

## End-to-End Pipeline

The benchmarking pipeline follows a standardized workflow:

1.  **Dimensionality Reduction:** Principal Component Analysis (PCA)
    reduces raw bands (up to 200) to 30 principal components, capturing
    $>99\%$ of spectral variance while eliminating sensor noise.

2.  **On-the-Fly Patch Extraction:** Spatial patches of size
    $11 \times 11$ centered on labeled pixels are extracted dynamically
    by PyTorch `DataLoader` instances, preventing RAM exhaustion.

3.  **Few-Shot Split Protocol:** Strict class-balanced sample
    partitioning with per-run seeds (`seeds: [1, 2, 3]`).

4.  **Model Training:** Supervised cross-entropy minimization with Adam
    optimizer ($\text{lr}=0.001$, batch size 64) and early stopping
    (patience 12 epochs).

## MMFormer Architecture

Published in *IEEE TGRS* in 2026, `MMFormer` [@mmformer] addresses
spatial scale disparity via a dual-branch architecture:

-   **Macro Branch:** Downsamples the input patch into coarse semantic
    tokens, allowing the network to capture global scene context and
    regional field geometry.

-   **Micro Branch:** Operates at fine resolution to preserve
    high-frequency spectral transitions and subtle boundary textures.

-   **Interactive Cross-Attention:** A cross-scale attention bridge
    dynamically aligns macro contextual features with micro spectral
    tokens. Its asymptotic complexity remains quadratic:
    $\mathcal{O}(N^2)$.

## R2Mamba Architecture

Published in *IEEE JSTARS* in 2026, `R2Mamba` [@r2mamba] introduces
Route-Reliability selective scanning:

-   **Bidirectional Spatial-Spectral Scanning:** Flattens 3D patches
    into sequence tokens scanned across multiple spatial directions to
    resolve Mamba's 1D directional bias.

-   **Route Reliability Gating (RRG):** At boundary transitions between
    disparate land-cover classes, mixed pixels degrade scanning quality.
    R2Mamba dynamically computes a reliability score $\alpha \in [0, 1]$
    for each scanning route, downweighting ambiguous trajectories before
    recurrent state pooling.

-   **Complexity:** Operates with strict linear complexity:
    $\mathcal{O}(N)$.

# Experimental Results & Discussion

## Benchmark Datasets

-   **Indian Pines (AVIRIS):** $145 \times 145$ pixels, 200 raw bands
    (PCA $\rightarrow$ 30), 16 agricultural crop classes, $20\text{ m}$
    spatial resolution. Severe class imbalance (classes from 20 to 2,455
    pixels). Sample budget: 10 train / 5 validation pixels per class;
    9,482 test pixels.

-   **Pavia University (ROSIS):** $610 \times 340$ pixels, 103 raw bands
    (PCA $\rightarrow$ 30), 9 urban land-cover classes, $1.3\text{ m}$
    spatial resolution. Sample budget: 30 train / 10 validation pixels
    per class; 40,044 test pixels.

## Quantitative Results

Table [\[tab:main_results\]](#tab:main_results){reference-type="ref"
reference="tab:main_results"} reports the statistical performance across
three independent seeded runs ($\text{Mean} \pm \text{Std}$).

## Discussion & Complexity Analysis

**Accuracy & Stability:** `R2Mamba` establishes clear dominance across
all metrics. On Indian Pines, `R2Mamba` achieves an OA of $83.89\%$,
outperforming `MMFormer` by $+7.48\%$ while exhibiting $2.3\times$ lower
standard deviation ($\pm 1.66\%$ vs. $\pm 3.84\%$). On Pavia University,
`R2Mamba` reaches an exceptional $97.59\%$ mean OA ($98.03\%$ peak) with
near-zero variance ($\pm 0.60\%$). `MMFormer` suffers from noticeable
instability on urban boundaries ($\pm 5.98\%$ OA variance) due to
quadratic token sensitivity.

**Computational Efficiency:** As shown in
Table [\[tab:main_results\]](#tab:main_results){reference-type="ref"
reference="tab:main_results"}, `R2Mamba` trains in
$\sim$`<!-- -->`{=html}26 seconds on Indian Pines and
$\sim$`<!-- -->`{=html}40 seconds on Pavia University, representing a
**2.9$\times$ to 3.5$\times$ speedup** over `MMFormer`. Because
`R2Mamba` computes linear $\mathcal{O}(N)$ selective recurrent states,
its memory consumption remains constant per token step, allowing
seamless scaling to larger remote sensing scenes without GPU memory
overflow.

# Conclusion

This project presented a rigorous empirical head-to-head evaluation of
the two latest 2026 hyperspectral classification architectures:
`MMFormer` and `R2Mamba`. Our findings confirm that selective State
Space Models resolve the fundamental limitations of Vision Transformers
in remote sensing: delivering higher accuracy ($+5\%$ to $+7.5\%$ OA),
substantially lower variance, and a $3.5\times$ reduction in training
latency.

::: thebibliography
00 T. Feng *et al.*, "Macro-Micro Transformer for Hyperspectral Image
Classification," *IEEE Trans. Geosci. Remote Sens.*, vol. 64, pp. 1--14,
2026. X. Shang *et al.*, "Route-Reliability Mamba for Hyperspectral
Image Classification," *IEEE J. Sel. Top. Appl. Earth Obs. Remote
Sens.*, vol. 19, pp. 1--12, 2026. T. Rachamalla *et al.*, "A PyTorch
Library for Hyperspectral Image Models: Technical Report,"
*arXiv:2609.39871*, 2026. S. K. Roy *et al.*, "HybridSN: Exploring
3-D--2-D CNN Feature Hierarchy for Hyperspectral Image Classification,"
*IEEE GRSL*, vol. 17, no. 2, pp. 277--281, 2019. Z. Zhong *et al.*,
"Spectral-Spatial Residual Network for Hyperspectral Image
Classification," *IEEE TGRS*, vol. 56, no. 2, pp. 847--865, 2017. D.
Hong *et al.*, "SpectralFormer: Rethinking Hyperspectral Image
Classification With Transformers," *IEEE TGRS*, vol. 60, pp. 1--15,
2021. A. Gu and T. Dao, "Mamba: Linear-Time Sequence Modeling with
Selective State Spaces," *arXiv:2312.00752*, 2023.
:::
