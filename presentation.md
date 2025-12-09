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

- Widely used method in the industry today to tackle various challenges and to  better understand human behavior. 
-  Different models -> different fundamental approaches 
-  (tab) Understand how each works and compare them. one-size =/ fits-all solution. 
- By comparing ->  we can identify which ones work best on customer feedback. 
-  Assist businesses in improving their products and services. Additionally, it can show how transformer-based models can improve modeling and help individuals assess which methods work better.

### Desired Outcome
A successful outcome is a fully implemented end-to-end pipeline:
    LDA + LSA, +  a BERT-based topic model on the Amazon Reviews dataset ->  produces a quantitative comparison using metrics such as:

    Topic coherence
    Topic diversity
    Clustering performance


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

### BERT 

Model Initialization

    Constructed a BERTopic instance with:  SentenceTransformer embeddings from the steps above
        So embedding_model=None

Dimensionality Reduction (UMAP) 

Clustering (HDBSCAN)
    Identify coherent groups of documents. 

Topic Extraction (c-TF-IDF) to extract topwords 

Generated interpretable topic representations ->  BERTopic’s representation model.


## EDA 



