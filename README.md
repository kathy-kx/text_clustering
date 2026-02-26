# Assignment 2 – Text Clustering

Unsupervised clustering of research paper abstracts across five semantically similar cancer categories, comparing BOW, TF-IDF, LDA, and Word2Vec feature representations with K-Means, EM (Gaussian Mixture), and Hierarchical clustering algorithms.

---

## Categories

| Label | Category |
|-------|----------|
| a | Colon Cancer |
| b | Liver Cancer |
| c | Lung Cancer |
| d | Stomach Cancer |
| e | Thyroid Cancer |

200 papers per category (1 000 total), sourced from [lens.org](https://www.lens.org).

---

## Repository Structure

```
assignment2-clustering/
├── data/                        # Raw CSV files downloaded from lens.org
│   ├── Colon_Cancer.csv
│   ├── Liver_Cancer.csv
│   ├── Lung_Cancer.csv
│   ├── Stomach_Cancer.csv
│   └── Thyroid_Cancer.csv
│
├── features/                    # Pre-built feature matrices (output of data_processing.ipynb)
│   ├── X_bow.npz                # BOW sparse matrix       (1000 × ~4000)
│   ├── X_tfidf.npz              # TF-IDF sparse matrix    (1000 × ~4000)
│   ├── X_lda.npy                # LDA topic distribution  (1000 × 5)
│   ├── X_w2v.npy                # Word2Vec mean vectors   (1000 × 100)
│   ├── labels.npy               # True labels array       (1000,)
│   ├── bow_vectorizer.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── lda_model.pkl
│   └── word2vec.model
│
├── data_processing.ipynb        # Data prep, cleaning & feature engineering
├── clustering.ipynb             # Clustering, evaluation & error analysis
├── processed_data.csv           # Full cleaned dataset (1000 rows)
├── train.csv                    # 80% split (800 rows, stratified)
└── test.csv                     # 20% split (200 rows, stratified)
```

---

## Notebooks

| Notebook | Covers | Owner |
|----------|--------|-------|
| `data_processing.ipynb` | Data loading · Mild & deep cleaning · Feature engineering (BOW, TF-IDF, LDA, Word2Vec) 
| `clustering.ipynb` | K-Means · EM · Hierarchical · Evaluation (Kappa, Silhouette, Coherence) · Error Analysis 

> **Run `data_processing.ipynb` first** — it generates the `features/` folder and `processed_data.csv` that `clustering.ipynb` depends on.

---

## Pipeline Overview

```
lens.org CSVs
    └─► data_processing.ipynb
            ├── Mild cleaning  (lowercase, strip HTML/DOI, truncate to 150 words)
            ├── Deep cleaning  (remove digits/punctuation, lemmatize, remove stopwords)
            ├── Dataset summary table + visualizations
            ├── Train/test split (80/20, stratified)
            └── Feature engineering
                    ├── BOW         → features/X_bow.npz
                    ├── TF-IDF      → features/X_tfidf.npz
                    ├── LDA         → features/X_lda.npy
                    └── Word2Vec    → features/X_w2v.npy

    └─► clustering.ipynb
            ├── K-Means, EM (GMM), Hierarchical clustering
            ├── Evaluation: Kappa, Silhouette, Coherence
            ├── Comparison across feature representations
            └── Error Analysis (top frequent words, collocations)
```

---

## How to Run on Google Colab

### Recommended Colab Environment

To ensure reproducible results, pin the following package versions at the top of the notebook:

**`clustering.ipynb`**
```python
!pip install -q \
    numpy==2.0.2 \
    pandas==2.2.2 \
    scikit-learn==1.6.1 \
    scipy==1.16.3 \
    gensim==4.4.0 \
    sentence-transformers==5.2.3 \
    torch==2.10.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

> Clustering results (especially K-Means) are sensitive to library versions due to differences in random state handling and numerical precision. Using the same versions across the team ensures the "champion model" stays consistent.

---

### Step 1 – Run `data_processing.ipynb`

1. Open [Google Colab](https://colab.research.google.com) and upload `data_processing.ipynb`
2. Upload the `data/` folder contents via the Colab file browser (left sidebar → Files → Upload):
   - `Colon_Cancer.csv`, `Liver_Cancer.csv`, `Lung_Cancer.csv`, `Stomach_Cancer.csv`, `Thyroid_Cancer.csv`
3. Run the environment setup cell, then `Runtime → Run all`
4. Download the outputs and share with the team:
   - `processed_data.csv`, `train.csv`, `test.csv`
   - The entire `features/` folder

### Step 2 – Run `clustering.ipynb`

1. Upload `clustering.ipynb` to Colab
2. Upload the `features/` folder and `processed_data.csv`
3. Run the environment setup cell, then `Runtime → Run all`

> **Note on train/test split:** Since clustering is unsupervised, algorithms are applied to the **full 1 000-document dataset**. True labels (`features/labels.npy`) are used only for post-hoc evaluation (Kappa, error analysis). `train.csv` / `test.csv` are provided for reference.

---

## Loading Feature Matrices (in `clustering.ipynb`)

```python
import numpy as np
import scipy.sparse

X_bow   = scipy.sparse.load_npz('features/X_bow.npz')
X_tfidf = scipy.sparse.load_npz('features/X_tfidf.npz')
X_lda   = np.load('features/X_lda.npy')
X_w2v   = np.load('features/X_w2v.npy')
labels  = np.load('features/labels.npy')   # true labels: a/b/c/d/e
```

---

## Dependencies Summary

| Package | Used in | Purpose |
|---------|---------|---------|
| `numpy`, `pandas` | both | Data handling |
| `nltk` | `data_processing` | Tokenization, POS tagging, lemmatization, stopwords |
| `scikit-learn` | both | Vectorizers, LDA, clustering algorithms, evaluation metrics |
| `scipy` | both | Sparse matrix I/O |
| `gensim` | both | Word2Vec |
| `matplotlib` | both | Visualizations |
| `sentence-transformers` | `clustering` | Sentence-level embeddings |
| `torch` | `clustering` | Backend for sentence-transformers |
