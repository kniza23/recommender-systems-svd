# Movie Recommender Systems: User-Based Collaborative Filtering   (k-NN) & Matrix Factorization (SVD)

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

## System Approaches

### 1. Model-Based Approach: Singular Value Decomposition (SVD)
- **File:** `Recommender Systems_svd.ipynb
- **Concept:** Uses Matrix Factorization (`SVD` via `scikit-surprise`) on the **MovieLens 100k** dataset (`100,000` ratings, `943` users, `1,682` movies) to learn latent factors capturing underlying user preferences and item characteristics[cite: 1].
- **Evaluation:** Evaluates prediction accuracy using **5-Fold Cross-Validation** (`cross_validate`) with **RMSE** (`0.9365` mean) and **MAE** (`0.7381` mean), along with fit and test execution times[cite: 1].

### 2. Memory-Based Approach: k-Nearest Neighbors (k-NN) Collaborative Filtering
- **File:** `Recommender Systems_svd.ipynb
- **Concept:** Implements **User-Based Collaborative Filtering**, both from scratch (`CF_userbased`) and with `scikit-surprise` (`KNNWithMeans`, `user_based=True`). Predictions rely on a mean-centered weighted average of ratings from the top-$K$ most similar users[cite: 1].
- **Methodology:** Computes user similarity over co-rated items using **Pearson correlation** or **Cosine similarity**, handles missing ratings / matrix sparsity (`utilitymatrix.csv` and `utility_example`), masks 50% of known ratings for testing (`create_u_test`), and evaluates performance via 5-fold cross-validation (`RMSE`, `MAE`, fit/test times)[cite: 1, 3].

  ---

## Repository Structure

```text
├── Recommender Systems_svd.ipynb   # Main Colab/Jupyter notebook with all implementations & evaluations
├── ml-100k.zip                     # MovieLens 100k dataset archive[cite: 1, 3]
├── utilitymatrix.csv               # Preprocessed user-item utility matrix (943 users × 603 movies)[cite: 1, 3]
└── README.md                       # Project documentation[cite: 3]
