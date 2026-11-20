# Movie Recommender Systems: User-Based Collaborative Filtering (k-NN) & Matrix Factorization (SVD)

This project implements, compares, and evaluates two distinct paradigms of **Recommender Systems** using the **MovieLens 100k Dataset** (`ml-100k`). It covers both **Memory-Based Collaborative Filtering (k-NN)**—built from scratch and benchmarked with `scikit-surprise`—and **Model-Based Matrix Factorization (SVD)**.

---

## Key Technologies & Competencies

| Category | Tools / Competencies |
| :--- | :--- |
| **Language** | Python |
| **Libraries** | Pandas, NumPy, Scikit-Surprise (`1.1.5`) |
| **Algorithms** | Custom User-Based k-NN (`CF_userbased`), `KNNWithMeans`, Singular Value Decomposition (`SVD`) |
| **Skills** | Collaborative Filtering, Matrix Factorization, Pearson & Cosine Similarity, 5-Fold Cross-Validation, RMSE & MAE Evaluation |

---

## Repository Structure

```text
├── Recommender Systems_svd.ipynb   # Main Colab/Jupyter notebook with all implementations & evaluations
├── ml-100k.zip                     # MovieLens 100k dataset archive[cite: 1, 3]
├── utilitymatrix.csv               # Preprocessed user-item utility matrix (943 users × 603 movies)[cite: 1, 3]
└── README.md                       # Project documentation[cite: 3]
