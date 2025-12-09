#   Beyond LDA: A Comparative Analysis of Topic Modeling on Amazon Reviews
### **Team Members:** Caroline Delva · Amina Nsanza  
---

##  Project Overview
This project will explores topic modeling on Amazon Beauty product reviews using three different modeling approaches:

- **Latent Dirichlet Allocation (LDA)**
- **Latent Semantic Analysis (LSA)**
- **BERTopic (Transformer-Based Topic Modeling)**


### Task

Discover meaningful topics from large amounts of unstructured text and evaluate how classical statistical models compare to modern transformer-based approaches in terms of **topic quality**, coherence, and interpretability. 

### Why It Matters

- Topic modeling is widely used in the industry to analyze text and better understand human behavior
-  Different models → different fundamental approaches 
    - Understand how each works and compare them. one-size ≠ fits-all solution. 
- By comparing → we can identify which ones work best on customer feedback. 
-  Assist businesses in improving their products and services. Additionally, it can show how transformer-based models can improve modeling and help individuals assess which methods work better.

### Desired Outcome
A successful outcome is a fully implemented end-to-end pipeline:

- LDA 
- LSA
- BERTopic 

Applied to Amazon Reviews → Produces a quantitative comparison using metrics such as:

- Topic coherence
- Topic diversity
- Clustering performance


---

## Data   (You can change this it was just a filler)
**Source:** Amazon Reviews (Beauty Category)  
**Total Reviews:** *701,528 (full dataset)*  
**Subset for BERTopic:** *5,000 sampled reviews*  

**Available fields include:**
- `rating`
- `title`
- `text`
- `images`
- `asin`
- `parent_asin`
- `user_id`
- `timestamp`
- `helpful_vote`
- `verified_purchase`

---

### Data Preprocessing

### Data Transformation 

### LSA 

### LDA 

### BERTopic

Model Initialization

- Constructed a BERTopic instance with:  SentenceTransformer embeddings
- Dimensionality Reduction (UMAP) 
- Clustering (HDBSCAN)
    - Identify coherent groups of documents. 
- Topic Extraction (c-TF-IDF)
- Generated interpretable topic representations →  BERTopic’s representation model.


## EDA 

data/eda_results/plotly_rating_dist.html
data/eda_results/plotly_text_dist.html
data/eda_results/plotly_verified_purchase.html
data/eda_results/interactive_reviews_ratings_2023.html

<p align="center">
  <iframe src="../data/eda_results/plotly_rating_dist.html" width="800" height="500"></iframe>
</p>

<p align="center">
  <iframe src="../data/eda_results/plotly_text_dist.html" width="800" height="500"></iframe>
</p>

<p align="center">
  <iframe src="../data/eda_results/plotly_verified_purchase.html" width="800" height="500"></iframe>
</p>

<p align="center">
  <iframe src="../data/eda_results/interactive_reviews_ratings_2023.html" width="800" height="500"></iframe>
</p>

