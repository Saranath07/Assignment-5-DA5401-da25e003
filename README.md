# Visualizing Data Veracity Challenges in Multi-Label Classification

## DA5401 Assignment 5

This project explores the challenges of real-world machine learning, specifically focusing on data veracity issues in a multi-label classification context. Using the **Yeast dataset**, we employ non-linear dimensionality reduction techniques—**t-SNE** and **Isomap**—to visually inspect the data for noisy labels, outliers, and hard-to-learn samples. The final output is a Jupyter Notebook that documents the entire process, from data preprocessing to visualization and insightful analysis.

---

## Details

Name : Saranath P
Roll No : DA25E003

---

## Table of Contents
*   [Project Objective](#project-objective)
*   [Dataset](#dataset)
*   [Methodology](#methodology)
*   [Key Findings](#key-findings)
*   [Setup and Installation](#setup-and-installation)
*   [How to Run](#how-to-run)
*   [File Structure](#file-structure)

---

# Visualizing Data Veracity in Multi-Label Classification: A Deep Dive into the Yeast Dataset

## Project Objective

This project provides a comprehensive analysis of the Yeast dataset, aiming to uncover and visualize the inherent challenges of data veracity in a real-world, multi-label biological context. Using a combination of exploratory data analysis, advanced non-linear dimensionality reduction techniques (**t-SNE** and **Isomap**), and quantitative model evaluation, this investigation tells a complete story—from initial visual hunches to concrete, predictive insights.

The core goals were to:
1.  Visually identify data veracity issues like **noisy labels**, **outliers**, and **hard-to-learn samples**.
2.  Understand the underlying geometric structure (**data manifold**) of the gene expression data.
3.  Quantitatively measure how these visual insights translate into the performance of a machine learning classifier.

---

## The Dataset: Yeast Gene Expression

*   **Source:** MULAN Repository
*   **Description:** The dataset contains **2,417 samples** (yeast genes), each described by **103 features** (gene expression levels). The classification task is multi-label, with each gene potentially belonging to **14 different functional categories**.

---

## Methodology: A Multi-Stage Investigative Workflow

Our analysis followed a rigorous, multi-stage process to build a compelling data story.

### 1. Initial Data Exploration & Preprocessing

Before any visualization, we first sought to understand the nature of the classification problem itself.
*   **Label Frequency Analysis:** We plotted the distribution of the 14 labels and discovered a **severe class imbalance**. Some functional categories are extremely common, while others are very rare, posing a significant challenge for any predictive model.
*   **Label Co-occurrence Analysis:** A heatmap of label co-occurrences revealed a "tangled web" of relationships. Many functional categories frequently appear together, confirming the high degree of ambiguity and overlap inherent in the problem.
*   **Simplification for Visualization:** To create clear and interpretable plots, we simplified the 14 labels into four distinct categories for coloring: the two most frequent single-label classes, the most frequent multi-label combination, and a catch-all "Other" category.
*   **Feature Scaling:** All 103 features were standardized (mean=0, std=1) to ensure that distance-based algorithms like t-SNE and Isomap would function correctly without bias from feature scale.

### 2. Visualizing Local Structure with t-SNE

We used t-SNE to create a 2D map that emphasizes the local neighborhood structure of the data.
*   **Hyperparameter Tuning:** The `perplexity` parameter was tuned by comparing plots for values of 5, 30, and 50. A value of **30** was chosen as it provided the optimal balance, revealing clear, coherent clusters without the noisy fragmentation of lower values or the over-smoothing of higher values.
*   **Enhanced Visualization with Density Clouds:** To better visualize cluster density and overlap, the final scatter plots were enhanced by overlaying **Kernel Density Estimation (KDE) "clouds"**. This technique highlighted the dense cores of our key clusters and made regions of class overlap visually explicit.

### 3. Uncovering Global Structure with Isomap

To understand the large-scale geometry of the data, we used Isomap.
*   **Manifold Unrolling:** Isomap was applied to "unroll" the high-dimensional data manifold, creating a 2D representation that preserves the global geodesic distances between points.
*   **Comparison to t-SNE:** The resulting plot was contrasted with the t-SNE visualization to highlight the fundamental difference between preserving local cluster separation (t-SNE) versus revealing the overall shape and curvature of the data (Isomap).

### 4. The Ultimate Test: A Predictive Showdown

The final, and most crucial, step was to bridge our visual findings with quantitative performance.
*   **Optimal Dimensionality:** We recognized that the 2D limitation for visualization is not optimal for prediction. We systematically determined a better number of components for each method:
    *   **PCA:** Chosen to capture 80% of the original variance (39 dimensions).
    *   **t-SNE & Isomap:** Tuned by testing a range of dimensions (`[2, 5, 10, 15, 20]`) and selecting the one that yielded the highest k-NN classifier accuracy.
*   **Fair Comparison:** A k-Nearest Neighbors (k-NN) classifier was trained and evaluated on four different feature sets: the **Baseline** (103D), optimal **PCA** (39D), optimal **t-SNE** (10D), and optimal **Isomap** (20D).

---

## Key Results and Findings

### 1. The Dataset is Fundamentally Difficult
The initial EDA proved that this is an inherently challenging problem due to severe class imbalance and a high degree of label co-occurrence. The vast majority of samples fall into an "Other" category, representing a long tail of rare label combinations.

### 2. t-SNE Successfully Reveals Data Veracity Issues
The t-SNE plots, enhanced with KDE clouds, successfully visualized the key challenges:
*   **Noisy Labels:** Points of one color located deep within the density cloud of another.
*   **Outliers:** Isolated points far from any dense cloud.
*   **Hard-to-Learn Samples:** Regions where the transparent density clouds of different colors clearly overlapped, representing areas of high class ambiguity.

### 3. Isomap Confirms a Complex, Non-Linear Manifold
The Isomap visualization revealed that the data does not form simple, separable blobs. Instead, it lies on a single, continuous, and highly **curved manifold**. This explains *why* the data is so challenging: the classes are intertwined along a complex geometric surface, making them impossible to separate with simple linear models.

### 4. The Predictive Showdown: A Story of Subtle Victories
The final classification experiment yielded a nuanced and insightful result:
*   **All methods performed similarly, achieving around 20% exact match accuracy**, confirming the dataset's high degree of fundamental difficulty.
*   **PCA did not fail**, performing on par with the baseline while using fewer than half the dimensions. It successfully compressed the data but could not improve upon it.
*   **Isomap and t-SNE provided a subtle but significant performance lift**, edging out both the baseline and PCA. **Isomap (20D)** emerged as the narrow victor at **20.11% accuracy**.

## Final Conclusion

This investigation demonstrates the profound connection between visualization and quantitative modeling. While advanced techniques like t-SNE and Isomap create powerful visualizations that reveal the *nature* of data veracity challenges—local noise, global complexity, and ambiguous boundaries—they are not a magic bullet. For a fundamentally difficult dataset like Yeast, their benefit translated not to a knockout victory, but to a **subtle and hard-won predictive advantage**.

The project successfully shows that dimensionality reduction techniques are powerful feature engineering tools. They can untangle complex data structures to provide a measurable performance gain. However, it also proves that overcoming the most severe data veracity issues requires more than just better features; it necessitates sophisticated models designed to handle the inherent noise and ambiguity of the problem domain.

---
**Tools Used:** Python, Jupyter, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
## Setup and Installation

To run the analysis, you will need Python 3 and the following libraries.

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment. The required packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    If you do not have a `requirements.txt` file, you can install them manually:
    ```bash
    pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyterlab
    ```

## How to Run

1.  **Download the Dataset:** Download the `yeast.arff` file from the [MULAN Repository](http://mulan.sourceforge.net/datasets-mlc.html) and place it in the root directory of this project.
2.  **Launch Jupyter:** Open a terminal in the project directory and run:
    ```bash
    jupyter lab
    ```
    or
    ```bash
    jupyter notebook
    ```
3.  **Open and Run the Notebook:** Open the `DA5401_A5_Notebook.ipynb` file and run the cells sequentially from top to bottom.

## File Structure

```
.
├── DA5401_A5_Notebook.ipynb    # The main Jupyter Notebook with all code and analysis.
├── yeast.arff                  # The dataset file (must be downloaded).
├── README.md                   # This file.
└── .gitgnore         
```