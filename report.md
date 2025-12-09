# Beyond LDA: A Comparative Analysis of Topic Modeling on Amazon Reviews


### **Team Members:** Caroline Delva · Amina Nsanza  

### Date: December, 12 2025 

### Class: Advanced NLP DSAN 5800 · Professor Chris Larson
---

##  Introduction 

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


# Literature Review 


---

## Methods 

### Data Processing 

#### Data Collection 

Data collection was automated as the data was downloaded programmatically from the UCSD Amazon Reviews 2023 (UCSD Amazon Reviews 2023, 2023). The source index page was scraped to locate all the JSON.gz files containing the reviews. Each file was streamed and decompressed, and uploaded to an S3 bucket subfolder, amazon_raw/S3, so that no files were stored locally, as the files were incredibly large.  

#### Data Cleaning
The pipeline looped through each JSON file from the amazon_raw/ S3 directory and cleaned the reviewText, summary, and title fields. Cleaning included lowercasing, removing URLs, usernames, non-ASCII characters, punctuation, and collapsing whitespace. 

#### Named Entity Recognition Redaction

After cleaning, the pipeline applied spaCy’s en_core_web_sm model to detect named entities. Detected people, organizations, locations, and other entity types are replaced with tags like [PERSON], [ORG], and [LOCATION]. This step anonymized the text, preventing identity leakage and bias in the models. Every cleaned record is written to a temporary file and then uploaded to the amazon_clean/ directory in S3 with the same filename.

#### Text preprocessing
The clean text was then preprocessed as the AmazonPreprocessor class streams the clean records from this prefix, reads the chosen text field, and relies on scikit-learn’s token pattern to keep alphabetic tokens of length three or more. Stopwords combined scikit-learn’s English list, NLTK stopwords, simple HTML artifacts such as “br” and “nbsp”, and domain-specific review words such as “great” and “product”, which reduced plain language before vectorization.

#### Count vectorization

For count vectors, the build_count_vectors method of the AmazonPreprocessor class collected up to max_docs texts from amazon_clean/ and fits a CountVectorizer with unigrams and bigrams, a maximum of 40,000 features, and frequency filters (min_df=20, max_df=0.5). Tokens shorter than three characters were dropped, and the combined stopword list was removed from the corpus. The resulting sparse document–term matrix and vocabulary mapping are written to temporary files on disk and then uploaded to S3 under amazon_vectors/countvectorizer/ as vectors.npz and vocab.json.

#### TF–IDF vectorization

TF–IDF vectors were built using the build_tfidf_vectors method of the AmazonPreprocessor class, which used the same document stream, token pattern, stopwords, ngram_range, and frequency thresholds as the count model. The TfidfVectorizer produced a sparse matrix of TF–IDF weights instead of raw counts. As with the count vectors, the matrix and vocabulary were then saved as temporary files and uploaded to S3 under amazon_vectors/tfidf/ for later analysis.

#### Sentence embeddings

Dense sentence embeddings were created with SentenceTransformer through the build_embeddings method of the AmazonPreprocessor class. The pipeline loaded a specified model, such as all-MiniLM-L6-v2, streamed up to max_docs texts from amazon_clean/, and encoded them in batches into a fixed-size NumPy array of embeddings. The array was saved as embeddings. npy in a temporary directory and then uploaded to S3 under amazon_vectors/embeddings/, providing a parallel dense representation of the same review corpus.

### EDA 


<p align="center">
  <iframe src="./data/eda_results/plotly_rating_dist.html" width="800" height="500"></iframe>
</p>

Figure 1


<p align="center">
  <iframe src="./data/eda_results/plotly_text_dist.html" width="800" height="500"></iframe>
</p>

Figure 2 

<p align="center">
  <iframe src="./data/eda_results/plotly_verified_purchase.html" width="800" height="500"></iframe>
</p>

Figure 3

<p align="center">
  <iframe src="./data/eda_results/interactive_reviews_ratings_2023.html" width="800" height="500"></iframe>
</p>

Figure 4 

### LDA 
Latent Dirichlet Allocation, LDA, identified ten topics from the All Beauty Amazon reviews. LDA treats each review as a mixture of topics. Each topic is defined by words that often appear together (Blei, Ng, and Jordan, 2003). The workflow queried the CountVectorizer matrix and vocabulary from S3, then loaded both locally. This ensured the model used the same token space created during preprocessing.

Training used scikit-learn’s batch LDA with a fixed random seed for consistency. After training, the code extracted the highest-weight words for each topic from the component matrix. The pipeline also produced word clouds from these weights to support direct visual inspection and faster interpretation.

All outputs were saved under data/lda_results. The directory includes count_vectors.npz, vocab.json, lda_topics.txt, and a wordclouds folder. These files store the trained topic structure, vocabulary mapping, top topic terms, and visual summaries used to assign clear, human-readable labels.

### LSA 

Latent Semantic Analysis (LSA), identified lower-dimensional topics in the All Beauty Amazon reviews. LSA applied Truncated SVD to the TFIDF matrix, which reduced sparse text features to dense semantic components (Deerwester et al, 1990). The workflow downloaded the TFIDF vectors and vocabulary from S3 and loaded the sparse matrix locally to keep the feature space aligned with the preprocessing step.

The model used ten topics using scikit learn TruncatedSVD with a fixed random seed. The pipeline applied a Normalizer after SVD to produce stable document embeddings. After training, the code extracted the highest weight terms for each component from the SVD loadings. These terms highlighted the words that define each LSA dimension.


### BERTopic 

## Analysis & Results 

### LDA 

Topic terms and themes for All Beauty reviews appear in the table. Each topic lists the highest-weight words from the LDA model. These words show the strongest signals in the count vector space. The table also includes a three-word theme for each topic. This theme summarizes the pattern suggested by the top words. The topics cover product areas such as hair tools, skin care, nails, grooming, and makeup. This view helps you link the numeric output of LDA to clear product themes in the review set.


<p align="center">
  <iframe src="./data/lda_results/ldatable.png" width="800" height="500"></iframe>
</p>

Figure 5 

### LSA 

LSA topic terms and three word themes for All Beauty reviews appear in the table. Each topic lists the highest weight terms from the Truncated SVD model. These terms show the strongest signals in the TFIDF space. The themes point to clear patterns in the reviews. Several topics focus on hair care. Others focus on skin treatment, scented body products, or color cosmetics. A few topics reflect product quality or price concerns. These patterns show how LSA groups related words into broader semantic areas, which helps you read the main product themes present in the review set.


<p align="center">
  <iframe src="./data/lsa_results/lsatable.png" width="800" height="500"></iframe>
</p>

Figure 6


### BERTopic 


<p align="center">
  <iframe src="./data/bert_results/visuals/vis_topics_overview.html"
          width="850"
          height="600"
          style="border:none;">
  </iframe>
</p>

Figure 7




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
- Aligns with expectations → relies on dense sentence embeddings instead of raw word counts.

On topic diversity, LDA comes out strongest →  a wider spread of unique top words across topics. 

- Better at separating themes distinctly, even if the topics themselves aren’t as coherent. 
- LSA struggled → overlapping, less interpretable topics.

For cluster performance, that was measured by silhouette score, LDA again performs the best. 
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
