# Beyond LDA: A Comparative Analysis of Topic Modeling on Amazon Reviews


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

## Data  
**Source:** Amazon Reviews (Beauty Category)  
**Total Reviews:** *701,528 (full dataset)*  
**All Beauty subset:** 632K users; 112K items;701K ratings
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

*Raw json files streamed, decompressed, and stored in S3* 

---

# Data Preprocessing

### Data Cleaning

- Lowercased text, removed URLs, usernames, punctuation, extra white space, and non ASCII characters

### NER Redaction

- spaCy `en_core_web_sm` named entity detection  
- Replace people, organizations, locations with tags  
  - `[PERSON]`, `[ORG]`, `[LOCATION]`  
  
### Stopwords Removal

- Keep alphabetic tokens with length at least three  
- Remove extended stopword list  
  - scikit learn English  
  - NLTK stopwords  
  - HTML artifacts  
  - Specific words such as “great” and “product” 

---

# Data Transformation 

### Count Vectorization

- Unigrams and bigrams  
- At most 40K features  
- Frequency filters: `min_df=20`, `max_df=0.5`  



### TFIDF Vectorization

- Same text stream, token pattern, and stopwords as count model  
- TFIDF weights instead of raw counts  



### Sentence Embeddings

- SentenceTransformer model such as `all-MiniLM-L6-v2`  
- Batch encoding into dense embeddings  

---

# Models

### LSA 

- Load TFIDF vectors and vocab from S3  
- Fit ten topic Truncated SVD model with Normalizer  
- Extract top terms per component  

### LDA 

- Load count vectors and vocab from S3  
- Train ten topic LDA model in scikit learn  
- Extract top words per topic and build word clouds  

### BERTopic

Model Initialization

- Constructed a BERTopic instance with:  SentenceTransformer embeddings
- Dimensionality Reduction (UMAP) 
- Clustering (HDBSCAN)
    - Identify coherent groups of documents. 
- Topic Extraction (c-TF-IDF)
- Generated interpretable topic representations → BERTopic’s representation model.


## EDA 

<p align="center">
  <iframe src="./data/eda_results/plotly_rating_dist.html" width="800" height="500"></iframe>
</p>

<p align="center">
  <iframe src="./data/eda_results/plotly_text_dist.html" width="800" height="500"></iframe>
</p>

<p align="center">
  <iframe src="./data/eda_results/plotly_verified_purchase.html" width="800" height="500"></iframe>
</p>

<p align="center">
  <iframe src="./data/eda_results/interactive_reviews_ratings_2023.html" width="800" height="500"></iframe>
</p>


## Results 

### LDA 

<p align="center">
  <iframe src="./data/lda_results/ldatable.png" width="800" height="500"></iframe>
</p>

<p align="center">
  <iframe src="./data/lda_results/wordclouds/topic_0.png" width="800" height="500"></iframe>
</p>


### LSA 

<p align="center">
  <iframe src="./data/lsa_results/lsatable.png" width="800" height="500"></iframe>
</p>

<p align="center">
  <iframe src="./data/lsa_results/topicdistribution.png" width="800" height="500"></iframe>
</p>


### BERTopic 


<p align="center">
  <iframe src="./data/bert_results/visuals/vis_topics_overview.html"
          width="850"
          height="600"
          style="border:none;">
  </iframe>
</p>





## Model Performance & Comparison 

Topic Coherence: 

  - Assesses how meaningful & interpretable each topic is. 
  - Ensures that the top words within a topic have a logically connected theme.
  
Topic Diversity: 

  - Measures how distinct the topics are from one another.
  - Ensures that a model captures a wide range of themes instead of redundant or overlapping topics.

Silhouette Score  (Clustering Performance): 

- Evaluates how well the model groups similar reviews together,
- Validates that topics reflect real structure in customer feedback.


#### Topic Model Evaluation Comparison

| Model     | Topic Coherence (c_v) | Topic Diversity | Silhouette Score (Cluster Performance) |
|-----------|------------------------|-----------------|----------------------------------------|
| **LDA**        | 0.3000                 | **0.83**           | **0.4563**                               |
| **LSA**        | 0.3058                 | 0.47            | -0.0265                                 |
| **BERTopic**   | **0.4204**             | 0.6384         | 0.0502                                  |


Across all three BERTopic performs best on topic coherence. 

- Discovered themes that are generally more semantically meaningful and internally consistent 
- aligns with expectations → relies on dense sentence embeddings instead of raw word counts.

On topic diversity, LDA comes out strongest →  a wider spread of unique top words across topics. 

- Better at separating themes distinctly, even if the topics themselves aren’t as coherent. 
- LSA struggled → overlapping, less interpretable topics.

For cluster performance, that ws measured by silhouette score, LDA again performs the best. 
- BERTopic’s silhouette score was weak 
- LSA's poor score → clusters are weakly formed but directionally sensible.

**BERTopic**  was the strongest model for **interpretability** and semantic quality, while **LDA** is the strongest for **structural separation** and topic distinctiveness. LSA consistently underperforms across metrics.

#### WHAT TO MODEL TO USE.....

- If you want meaningful, human-like topic themes →  **BERTopic**.
- If you need sharply separated clusters → **LDA** is better.

Overall BERTopic did outperform the baseline models on topic coherence so texts such as reviews, the model would be better and capturing real semantic structure. 


## Conclusion 


<p align="center">
  <img src="./data/bert_results/visuals/conclusion_flowchart.png" width="700">
</p>
