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

## Project Objective

The primary goal is to analyze gene expression data from the Yeast dataset to uncover inherent data quality challenges. By reducing the 103-dimensional feature space to a 2D plane, we aim to:
1.  **Visualize the data's structure** using t-SNE and Isomap.
2.  **Identify data veracity issues**, such as:
    *   **Noisy/Ambiguous Labels:** Genes whose functions are misclassified or span multiple categories.
    *   **Outliers:** Experiments with highly unusual gene expression profiles.
    *   **Hard-to-Learn Samples:** Data points lying in regions where functional categories are thoroughly mixed.
3.  **Understand the data manifold** and explain how its complexity impacts the difficulty of a classification task.

## Dataset

*   **Name:** Yeast Dataset
*   **Source:** [MULAN Repository - Yeast Data](http://mulan.sourceforge.net/datasets-mlc.html)
*   **Description:** The dataset consists of 2,417 yeast genes (samples). Each gene is described by 103 features representing gene expression levels. The target is a multi-label binary matrix indicating membership in 14 possible functional classes.
*   **File Format:** `yeast.arff`

## Methodology

The analysis is broken down into three main parts:

1.  **Preprocessing and Initial Setup:**
    *   **Data Loading:** The `yeast.arff` file is loaded and split into a feature matrix `X` (103 features) and a target matrix `Y` (14 labels).
    *   **Label Simplification for Visualization:** To avoid a cluttered plot with 14 colors, a new target variable is created. It highlights the **two most frequent single-label classes**, the **most frequent multi-label combination**, and groups all other samples into an "Other" category.
    *   **Feature Scaling:** `StandardScaler` is applied to the feature matrix to standardize it (mean=0, variance=1). This is a crucial step for distance-based algorithms like t-SNE and Isomap to ensure all features contribute equally.

2.  **t-SNE for Local Structure Analysis:**
    *   **t-Distributed Stochastic Neighbor Embedding (t-SNE)** is used to visualize the local neighborhood structure of the data.
    *   The `perplexity` hyperparameter was tuned (tested values: 5, 30, 50), with **30** being chosen as the optimal value for providing a balance between revealing local clusters and maintaining a sensible global arrangement.
    *   The resulting plot is analyzed to pinpoint outliers, noisy labels, and regions of high class overlap.

3.  **Isomap for Global Manifold Analysis:**
    *   **Isometric Mapping (Isomap)** is applied to visualize the global geodesic structure of the data, assuming it lies on a non-linear manifold.
    *   The Isomap plot is compared to the t-SNE plot to highlight their fundamental differences (global vs. local structure preservation).
    *   The visualization is used to discuss the concept of the data manifold, its complexity, and how its curved nature makes the classification task inherently difficult for simple linear models.

## Key Findings

1.  **t-SNE Analysis:**
    *   The t-SNE visualization successfully revealed distinct local clusters but also highlighted significant data quality issues.
    *   **Noisy labels** were identified as points located deep within clusters of a different color.
    *   **Outliers** were visible as isolated points on the periphery of the main data cloud.
    *   **Hard-to-learn samples** were concentrated in a large, central region where all categories were heavily mixed, indicating fuzzy decision boundaries.

2.  **Isomap Analysis:**
    *   Isomap provided a much clearer view of the **global data structure**, revealing that the data lies on a single, continuous, and highly **curved manifold**.
    *   This non-linear structure explains why classification is so challenging for this dataset: the functional categories are intertwined along this curved surface, not cleanly separated in space.

3.  **Conclusion:**
    The combination of t-SNE and Isomap tells a compelling story. The Yeast dataset is challenging not only due to local data quality problems (noise, outliers) but also because of its complex global geometry. Any successful classifier for this data must be a non-linear model capable of learning the intricate, curved decision boundaries of the underlying data manifold.

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