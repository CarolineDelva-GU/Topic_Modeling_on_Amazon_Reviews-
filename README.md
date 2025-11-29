
# **Project Proposal**

## **1. Team**

**Project Title:** *Beyond LDA: A Comparative Analysis of Topic Modeling on Amazon Reviews*
**Team Members:** Caroline Delva (cd1338), Amina Nsanza (abn41)
**Preferred Track:** (E) Student-defined

---

## **2. Problem Statement & Motivation**

### **Task**

This project conducts a comparative analysis of three topic modeling approaches: **LDA**, **LSA**, and a **BERT-based transformer model**. The goal is to automatically discover meaningful topics from large collections of unstructured text and evaluate how classical statistical methods compare to modern transformer-based approaches in terms of **topic quality, coherence, and interpretability**.
The models were selected because they rely on fundamentally different text representations (bag-of-words, TF-IDF, embeddings), allowing a diverse comparison across modeling paradigms.

### **Why It Matters**

Topic modeling is a widely used method in industry and research for understanding large amounts of text and inferring human behavior. Each model type brings different strengths and limitations, so there is no single universal solution.
By comparing statistical models and transformer-based methods, especially in the context of customer feedback, we can identify which approaches generate more coherent and actionable topics. These insights can help businesses improve products and services and guide practitioners in selecting effective modeling techniques.

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

