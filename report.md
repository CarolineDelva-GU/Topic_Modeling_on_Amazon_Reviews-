# Beyond LDA: A Comparative Analysis of Topic Modeling on Amazon Reviews


### **Team Members:** Caroline Delva · Amina Nsanza  

**Advanced NLP DSAN 5800 · Professor Chris Larson · December, 12 2025** 

---
##  Introduction 

Topic modeling is one of the most widely used techniques in the industry today, from technology to health and marketing; it provides powerful tools that assist in organizing and understanding themes in large unstructured text data. This type of modeling automatically discovers meaningful subjects within documents, and enables businesses to better understand conversational structures, uncover common themes, and gain insights into human behavior. Although topic modeling is the most popular approach to understanding human behavior, there are fundamental differences in the multiple methods of topic modeling, which range from probabilistic and dimensional reduction techniques, embedding-based methods, indicating that there is no single solution that is a one-size-fits-all. 
This project explores a comparative analysis of three of the main topic modeling methods today: Latent Dirichlet Allocation (LDA), Latent Semantic Analysis (LSA), and a BERT-based transformer model (BERTopic). This comparison is critical because all of these models use fundamentally different methodologies for processing text and uncovering the themes in it.  The main objective is to evaluate and compare how well these models perform in discovering meaningful topics from an Amazon Reviews dataset. The performance metrics that will be used to measure success are: Topic coherence, Topic diversity, and Clustering performance through Silhouette score. These performance metrics will assess the models’ ability to capture how meaningful and interpretable each topic is and whether top words within a topic have a logically connected theme. Additionally, they measure how distinct the topics are from one another and if a wide range of themes is captured, rather than redundant and overlapping topics. Finally, the metrics will evaluate how well a model groups similar reviews together and validate that topics reflect real structure in customer feedback.

---

# Literature Review 

Classic topic modeling, like LDA, represent documents through a "bag-of-words" approach. This framework models each document as a finite mix over latent topics, and then each topic is later on extracted and defined by a distribution over words (1). Although LDA is a foundational model, it is limited because, it disregards semantic relationships and the natural sequential order of words. This classic probabilistic approach will be the baseline model in our project.

​
Attempts were made to bridge LDA’s gap between the order of words and the semantic relationship by introducing methods such as LSA. Where LDA would fall short because of its unreliability of simple keyword matching, LSA uses matrix decomposition to discover the latent structure that was disregarded by LDA in the term-document relationships (2). The LSA model does this by using the Singular Value Decomposition (SVD) method. This matrix decomposition method is used on a term-document matrix that is usually derived from TF-IDF counts, it then  uncovers the underlying latent semantic structure in a reduced-dimensional space (2). LSA’s ability to overcome issues such as synonymy, many ways to express a concept, and polysemy, words having multiple meanings, highlights why classical models may not be suited to interpret diverse and nuanced corpuses.

Presently, there are modern contextual topic modeling approaches, such as BERTopic, this model addresses the challenges that were faced by LDA and LSA by completely moving away from the  word-frequency statistics method to instead using rich document embeddings based representations (3). The BERTopic model is able to achieve this by using other pretrained language models like the baseline the Sentence-BERT. The language based transformer converts documents into rich dense embeddings. These embeddings from Sentence-Bert are then later used to uncover semantics, this is done through the encoding of semantic meaning. Additionally, BERTopic tops LDA and LSA, because it is able to use the encodings and generate natural clusters where, similar general ideas/themes are pulled together. These idea clusters (topics) are created in a reduced vector space, that is made by UMAP dimensionality reduction, and the clusters are formed through HDBSCAN. To incorporate semantic meaning and topic interpratation, the model uses a class based procedure ( c-TF-IDF) that interprets topic descriptions as well as balances semantics with interpretability (3). 

Therefore, through this literature review, we can see that contextual approaches capture more coherent topics; however, they do this by relying on baseline models and offer hybrid structures that differ from single source models. The difference in the  methodologies of these models make direct comparison critical. Classic models offer simplicity and efficiency, whereas modern contextual models provide stronger semantic modeling in return; however, they may come at a computational cost. This is why conducting a side by side evaluation is meaningful, so we can obtain a deeper understanding of how each of the models’ approach affect topic quality, coherence, and practical usefulness.


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
