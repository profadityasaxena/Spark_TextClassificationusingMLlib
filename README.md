# Text Classification using Apache Spark MLlib

## 📘 Project Overview

This project implements a scalable text classification pipeline using Apache Spark MLlib. It classifies Yelp customer reviews into categories based on review content using machine learning techniques in a distributed environment.

## 🧠 Skills Built & Applied

- Data engineering with Apache Spark
- Machine learning with Spark MLlib
- Text preprocessing (Tokenization, TF-IDF)
- Model evaluation and tuning
- PySpark scripting and pipeline automation
- Multi-model experimentation and benchmarking

## 🧰 Tech Stack

| Layer            | Technology                  |
|------------------|-----------------------------|
| Language         | Python 3.8+                 |
| Distributed Engine | Apache Spark 3.0           |
| ML Library       | Spark MLlib                 |
| Dataset Source   | Yelp Review Full (sampled)  |
| OS Environment   | Ubuntu (CU_VM)              |
| File System      | HDFS                        |
| Tools            | PySpark, VirtualBox, Git    |

## 🚀 Implementation Plan

### Step 1: Environment Setup
- Launch Apache Spark 3.0 CU_VM
- Transfer data to VM: `/home/cu/data/`
- Upload to HDFS:
  ```bash
  hadoop fs -mkdir /user/cu/data
  hadoop fs -put /home/cu/data/yelp_review_train.csv /user/cu/data/
  hadoop fs -put /home/cu/data/yelp_review_test.csv /user/cu/data/
  ```

### Step 2: Data Loading
```python
train_df = spark.read.option("header", True).csv("hdfs:///user/cu/data/yelp_review_train.csv")
test_df = spark.read.option("header", True).csv("hdfs:///user/cu/data/yelp_review_test.csv")
```

### Step 3: Preprocessing Pipeline
- `Tokenizer`
- `StopWordsRemover`
- `HashingTF` + `IDF`
- `StringIndexer` (labels)
- Combine into a `Pipeline`

### Step 4: Train First Classifier
- Model: `LogisticRegression`
- Fit pipeline on training data
- Evaluate accuracy on train and test sets

### Step 5: Dataset Size Impact
- Sample 10K, 15K, 20K, 25K, 30K from train set
- Train and test model
- Record and visualize accuracy trends

### Step 6: Second Classifier Comparison
- Alternate model: `NaiveBayes` or `RandomForestClassifier`
- Train and evaluate
- Compare with first classifier

### Step 7: Reporting
- Tabulate classifier performance
- Plot accuracy vs training size
- Save results

### Step 8: Batch Script Automation (Optional)
- Convert code to `.py` script
- Run with:
  ```bash
  spark-submit yelp_text_classifier.py
  ```

## 📂 Folder Structure
```
├── README.md
├── data/
│   └── yelp_review_train.csv
│   └── yelp_review_test.csv
├── scripts/
│   └── yelp_text_classifier.py
├── outputs/
│   └── accuracy_results.csv
│   └── accuracy_plot.png
```

## ✅ Future Enhancements
- Hyperparameter tuning and cross-validation
- Deep learning models via TensorFlowOnSpark
- Sentiment explainability using SHAP or LIME
"# Spark_TextClassificationusingMLlib" 
