# Online Banking Fraud Detection using Natural Language Processing

**Team 9:** Latif Masud, Wesley Mitchell, Gerald Wagner
**Course:** AI 574 – Natural Language Processing (Spring 2025)

## Overview

This project detects fraudulent online banking transactions by analyzing the sequence of user actions (API calls, page visits) using NLP. It trains a DistilBERT model to classify transaction sequences as fraudulent or legitimate based on the patterns learned from the text representation of these actions.

This code and related information can be found on [GitHub](https://github.com/lmasud-psu/AI_574_project/tree/main)

## Data

* **Source:** Modified from [FraudNLP GitHub](https://github.com/pboulieris/FraudNLP/blob/master/Fraud%20Detection%20with%20Natural%20Language%20Processing.rar)
* **Files:**
    * `data/Fraud Detection with Natural Language Processing.pkl`: Main transaction data (~105k transactions). Contains sequence of action IDs, timings, amount, etc., and a `is_fraud` label (0 or 1).
    * `data/vocab.csv`: Maps action IDs to API endpoint/URL names.
* **Challenge:** Severe class imbalance (only ~100 fraud cases out of ~105k).

## Requirements

* Python 3.x
* Key Libraries: `pandas`, `numpy`, `scikit-learn`, `nltk`, `matplotlib`, `seaborn`, `wordcloud`, `wordninja`, `pytorch`, `pytorch-lightning`, `transformers`, `sentence-transformers`

## Installation

1.  Ensure Python and pip are installed.
2.  Create and activate a virtual environment (recommended).
3.  Install required packages:
    ```bash
    pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud wordninja torch torchvision torchaudio pytorch-lightning transformers sentence-transformers
    ```
4.  Download NLTK stopwords data:
    ```python
    import nltk
    nltk.download('stopwords')
    ```
5.  Place the `.pkl` and `.csv` data files into a `./data/` subdirectory.

## Workflow Summary

1.  **Load Data:** Read transaction data (`.pkl`) and vocabulary (`.csv`).
2.  **Transform Vocabulary:** Convert raw API/URL strings from `vocab.csv` into readable English phrases using regex and `wordninja`. Store these in `vocab['new_name']`.
3.  **Create Action Strings:** Map the sequence of action IDs in each transaction to the transformed phrases and concatenate them into a single string feature (`actions_str`). Calculate other statistical features (time means/stds, log amount).
4.  **Preprocess Text for Model:**
    * Analyze word count distribution of `actions_str`.
    * Remove stopwords from `actions_str`.
    * Handle DistilBERT's token limit (512): Keep the first 256 and last 256 words if the sequence is too long. Store the result in `no_stopwords`.
5.  **Handle Imbalance:** Create a balanced training subset (`train_df_slice`) by oversampling fraud cases (duplicating them) and undersampling non-fraud cases (taking a random sample, e.g., 9x the oversampled fraud count). Shuffle this subset.
6.  **Tokenize:** Use the `DistilBERTTokenizer` on the `no_stopwords` column of the balanced subset (`train_df_slice`) to create input IDs and attention masks. Split into train/validation sets.
7.  **Model Training:**
    * Use `pytorch-lightning` to manage training.
    * Fine-tune a pre-trained `distilbert-base-uncased` model (`AutoModelForSequenceClassification`) for binary classification.
    * Use `AdamW` optimizer and `CrossEntropyLoss`.
    * Train for a set number of epochs (e.g., 10).

## Running the Code

The primary code is a Jupyter Notebook (`.ipynb`). Execute the cells or script sequentially. The steps include data loading, preprocessing, model definition (`FraudClassifier`), training (`pl.Trainer`), and evaluation.

## Evaluation & Results

* The model's performance is evaluated on the validation set using:
    * Confusion Matrix
    * Accuracy
    * Precision
    * Recall
    * F1-Score
* **Key Result:** The model achieves high Recall (e.g., ~91% in sample runs), demonstrating its effectiveness in identifying the rare fraud cases, which is critical for this task. Precision (e.g., ~81%) indicates some false positives occur. Overall F1-score (e.g., ~86%) shows a good balance.

*(Note: Clustering analysis using Sentence Transformers and KMeans was also explored as presented in the code for potential future feature engineering or pattern discovery.)*