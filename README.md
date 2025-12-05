
# **Project Proposal**

## **1. Team**

**Project Title:** *Beyond LDA: A Comparative Analysis of Topic Modeling on Amazon Reviews*
**Team Members:** Caroline Delva (cd1338), Amina Nsanza (abn41)
**Preferred Track:** (E) Student-defined

---

## **2. Problem Statement & Motivation**

### **Task**

This project is a comparative analysis of the topic models **LDA**, **LSA**, and a **BERT-based** transformer model. The task is to automatically discover meaningful topics from large amounts of unstructured text and evaluate how classical statistical models compare to modern transformer-based approaches in terms of topic quality, coherence, and interpretability. These models were chosen because they use different text representations, including **bag-of-words**, **TF-IDF**, and embeddings. The goal is to determine which model provides the best topic distribution.


### **Why It Matters**

This topic matters because it is one of the most widely used methods in tech today to tackle issues that deal with text and to better understand human behavior. Since there are multiple models that have different fundamental approaches, it is important to understand how each works and compare them, as there is never a one-size-fits-all solution. By comparing the different models’ approaches, from distributions to embeddings, we can identify which ones work best on customer feedback. This can, in return, assist businesses in improving their products and services. Additionally, it can show how transformer-based models can improve modeling and help individuals assess which methods work better. 

### **Desired Outcome (3 Weeks)**

A successful outcome is a **fully implemented end-to-end pipeline** that runs LDA, LSA, and a BERT-based topic model on the Amazon Reviews dataset, and produces a **quantitative comparison** using metrics such as:

* Topic coherence
* Topic diversity
* Clustering performance

---

## **3. Datasets**

### **Primary Dataset**

**Amazon Reviews Dataset**
Source: [https://www.cs.cornell.edu/~arb/data/amazon-reviews/](https://www.cs.cornell.edu/~arb/data/amazon-reviews/)
Includes: Amazon product reviews (text), large scale, English, consumer domain.

### **Preprocessing**

* Standard text cleaning
* Tokenization
* Optional stopword removal
* Data formatting for each model (e.g., BOW for LDA, TF-IDF for LSA, embeddings for BERT models)

The text will first be normalized by lowercasing all words and cleaning out punctuation, emojis, URLs, usernames, and other non-ASCII or irrelevant characters. Named Entity Recognition will be used to anonymize identifiable information so the topic model does not capture specific names as topics. After normalization, the text will be tokenized, stop words and extremely common or rare tokens will be removed, and very short or non-informative tokens will be filtered out. Multi-word expressions (such as common bigrams or trigrams) may be detected and preserved to keep meaningful phrases intact. Finally, the remaining tokens will be lemmatized so that different inflected forms of the same word are reduced to a shared base form.


### **Train/Val/Test Split**

Not applicable — all chosen methods are **unsupervised**, so no split is required.

---

## **4. Baseline**

### **Baseline Model**

The baseline model will be **Latent Dirichlet Allocation (LDA)**.

### **Baseline Metrics**

* Human evaluation
* Topic coherence
* Perplexity

---

## **5. Approach (Beyond Baseline)**

This project follows **Track E**, combining multiple course-related approaches by comparing classical techniques with transformer-based methods.

### **Planned Improvements / Additional Experiments**

1. **Implement and evaluate LSA** using TF-IDF representations to compare performance with LDA.
2. **Implement a BERT-based topic model** (e.g., BERTopic or similar embedding-driven method) to assess modern transformer-based topic discovery.

The primary goal is to determine which model produces the most coherent and interpretable topics on Amazon Reviews.

---

## **6. Compute & Resources**

**Using Jetstream2?** No

### **Compute Plan**

* Expected use of AWS cloud resources
* Model sizes: small (LDA, LSA) to medium (BERT-based topic model)
* Minimal training time for statistical models; transformer-based method may require GPU-based embedding generation

**Other Resources:** AWS Cloud

---

## **7. Risks & Scope**

### **Potential Risks**

* Inability to access the dataset if AWS S3 permissions are restricted
* Large transformer-based models may require more compute than expected
* Topic quality evaluation may be subjective without strong metrics

### **Plan B**

If full access to the Amazon Reviews dataset is not possible, switch to a publicly accessible review dataset or a smaller subset from an alternative source.

---

## **8. Milestones**

* **End of Week 1:** Finalized proposal
* **End of Week 2:** Initial EDA, baseline model (LDA), and comparative models (LSA, BERT-based) implemented
* **End of Week 3:** Polished models, final evaluation, and full analysis

