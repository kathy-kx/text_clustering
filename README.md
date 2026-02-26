
## Deliverables

| File | Contents |
|------|----------|
| `processed_data.csv` | 1000 rows — `text_raw` / `text_raw_150` / `text_clean` / `label` / `category` |
| `train.csv` | 800 rows (80%, stratified) |
| `test.csv` | 200 rows (20%, stratified) |
| `viz_tfidf_top_terms.png` | Top-10 TF-IDF terms per category |
| `viz_lda_heatmap.png` | LDA topic-word weight heatmap |
| `viz_w2v_pca.png` | Word2Vec embeddings projected to 2D via PCA |
| `features/X_bow.npz` | BOW sparse matrix (1000 x 5000) |
| `features/X_tfidf.npz` | TF-IDF sparse matrix (1000 x 5000) |
| `features/X_lda.npy` | LDA topic-distribution matrix (1000 x 5) |
| `features/X_w2v.npy` | Word2Vec mean-pooled vectors (1000 x 100) |
| `features/labels.npy` | Label array (1000,) |
| `features/*.pkl` / `word2vec.model` | Fitted vectorisers and models |