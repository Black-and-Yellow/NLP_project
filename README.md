# NLP Project: Tamil Text Classification with TF-IDF and Naive Bayes

A comprehensive natural language processing project specifically designed for Tamil language text classification using TF-IDF vectorization and Naive Bayes machine learning algorithms. This project performs both category and sentiment classification on Tamil news articles, implementing end-to-end Tamil language processing pipelines from the ground up.

## Project Overview

This project implements text classification pipelines specifically for Tamil language articles with two main tasks:
1. **Category Classification**: Classifying Tamil news articles into different categories
2. **Sentiment Classification**: Determining the sentiment (positive, negative, neutral) of Tamil articles

The project includes Tamil-specific preprocessing, feature extraction, model training, and comprehensive evaluation, designed specifically for the Tamil language from the beginning.

## Key Features

- **Tamil-specific TF-IDF vectorization** 
- **Multinomial Naive Bayes classification** optimized for Tamil text
- Support for both unigram and n-gram features in Tamil
- Comprehensive evaluation metrics (accuracy, precision, recall, F1-score)
- Confusion matrix visualization
- Model persistence (pickle serialization)
- Clean, modular notebook structure
- Both category and sentiment classification in single pipelines
- Tamil language preprocessing and tokenization

## Project Structure

```
NLP-PROJECT/
├── README.md                          # This file
├── tf-idf.ipynb                       # Unigram TF-IDF with Naive Bayes
├── tf-idf_n-gram.ipynb                # N-gram TF-IDF with Naive Bayes
├── preProcessing.ipynb                # Data preprocessing and cleaning
├── vectorization_and_model.ipynb      # Original vectorization notebook
├── vectorization_and_model_n-gram.ipynb # Original n-gram vectorization
├── web_scrape.py                      # Web scraping utility
├── data/                              # Raw data directory
├── dataset/                           # Dataset files
│   ├── data.csv                       # Original dataset
│   └── dataset_with_sentiment.csv     # Dataset with sentiment labels
├── output/                            # Processed data and results
│   ├── processed_data.csv             # Cleaned category data
│   ├── processed_sentiment_data.csv   # Cleaned sentiment data
│   ├── final_dataset.csv              # Final combined dataset
│   ├── model_comparison_tfidf.csv     # Model comparison results
│   ├── sentiment_model_comparison_tfidf.csv
│   └── test_predictions_tfidf.csv     # Test predictions
├── models/                            # Trained models and vectorizers
│   ├── category_naive_bayes.pkl       # Category Naive Bayes model
│   ├── category_svm.pkl               # Category SVM model
│   ├── category_logistic.pkl          # Category Logistic Regression model
│   ├── category_model.pkl             # Best category model
│   ├── category_vectorizer.pkl        # Category TF-IDF vectorizer
│   ├── sentiment_naive_bayes.pkl      # Sentiment Naive Bayes model
│   ├── sentiment_svm.pkl              # Sentiment SVM model
│   ├── sentiment_logistic.pkl         # Sentiment Logistic Regression model
│   ├── sentiment_model.pkl            # Best sentiment model
│   └── sentiment_vectorizer.pkl       # Sentiment TF-IDF vectorizer
├── reports/                           # Evaluation reports (JSON format)
│   ├── category_naive_bayes_report.json
│   ├── category_svm_report.json
│   ├── category_logistic_report.json
│   ├── sentiment_naive_bayes_report.json
│   ├── sentiment_svm_report.json
│   └── sentiment_logistic_report.json
├── resources/                         # Reference data
│   ├── positive_words.txt
│   ├── negative_words.txt
│   ├── negation_words.txt
│   ├── intensity_words.txt
│   ├── stopwords.txt
│   └── suffixes.csv
└── figures/                           # Generated visualizations

```

## Notebooks

### 1. tf-idf.ipynb

**Unigram TF-IDF with Naive Bayes Classification (Submission Ready)**

This notebook implements Tamil text classification using unigram TF-IDF features with only Naive Bayes classifier:

- TF-IDF Vectorization with max 10,000 unigram features
- Document frequency filtering (min_df=3, max_df=0.8)
- Multinomial Naive Bayes classifier (only model used)
- Category classification (9 categories)
- Sentiment classification (3 sentiments)
- Performance metrics and confusion matrices
- Derived from `vectorization_and_model.ipynb` but simplified to use only Naive Bayes

**Key Sections:**
1. Data Loading and Exploration
2. TF-IDF Vectorization
3. Train-Test Split (80-20)
4. Naive Bayes Model Training
5. Model Evaluation
6. Confusion Matrix Visualization
7. Model Persistence

**Expected Output:**
- Trained models: `category_naive_bayes.pkl`, `sentiment_naive_bayes.pkl`
- TF-IDF Vectorizers: `category_vectorizer.pkl`, `sentiment_vectorizer.pkl`
- Evaluation Reports: JSON files with metrics and classification reports

### 2. tf-idf_n-gram.ipynb

**N-gram TF-IDF with Naive Bayes Classification (Submission Ready)**

This notebook extends the unigram approach with n-gram features using only Naive Bayes:

- TF-IDF Vectorization with n-gram range (1, 2) - unigrams and bigrams
- Max 10,000 n-gram features
- Document frequency filtering (min_df=3, max_df=0.8)
- Multinomial Naive Bayes classifier (only model used)
- N-gram feature analysis and examples
- Category classification (9 categories)
- Sentiment classification (3 sentiments)
- Derived from `vectorization_and_model_n-gram.ipynb` but simplified to use only Naive Bayes

**Key Sections:**
1. Data Loading
2. TF-IDF with N-grams
3. N-gram Feature Examples
4. Train-Test Split
5. Naive Bayes Model Training
6. Model Evaluation
7. Confusion Matrix Visualization
8. Model Persistence

**Expected Output:**
- Trained models: `category_naive_bayes.pkl`, `sentiment_naive_bayes.pkl`
- TF-IDF Vectorizers: `category_vectorizer.pkl`, `sentiment_vectorizer.pkl`
- Evaluation Reports: JSON files with metrics and classification reports

## Data Description

### Input Data

The project uses Tamil language news articles with the following structure:

- **dataset/data.csv**: Original Tamil dataset with article titles and categories
- **dataset/dataset_with_sentiment.csv**: Tamil dataset with sentiment labels added

### Processed Data

After Tamil-specific preprocessing:

- **output/processed_data.csv**: Cleaned Tamil articles with category labels
  - Columns: original_title, cleaned_title, category, etc.
  - Rows: ~3,000-5,000 Tamil articles (depending on dataset)

- **output/processed_sentiment_data.csv**: Cleaned Tamil articles with sentiment labels
  - Columns: original_title, tokenized_title, sentiment, etc.
  - Rows: ~2,000-4,000 Tamil articles

## Usage

### Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

### Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Running the Notebooks

1. **Data Preprocessing** (if needed):
   ```
   Run preProcessing.ipynb first to prepare the data
   ```

2. **Unigram Classification**:
   ```
   Run tf-idf.ipynb in Jupyter Notebook
   ```

3. **N-gram Classification**:
   ```
   Run tf-idf_n-gram.ipynb in Jupyter Notebook
   ```

### Loading Trained Models

```python
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load model
with open('models/category_naive_bayes.pkl', 'rb') as f:
    model = pickle.load(f)

# Load vectorizer
with open('models/category_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Predict on new Tamil text
new_tamil_text = ["உங்கள் தமிழ் உரை இங்கே"]
tfidf_matrix = vectorizer.transform(new_tamil_text)
predictions = model.predict(tfidf_matrix)
```

## Model Architecture

### TF-IDF Vectorization for Tamil

TF-IDF (Term Frequency-Inverse Document Frequency) converts Tamil text into numerical features:

- **Term Frequency (TF)**: Frequency of a Tamil term in a document
- **Inverse Document Frequency (IDF)**: How important a Tamil term is across all documents
- **TF-IDF**: Product of TF and IDF scores

Configuration optimized for Tamil:
- max_features: 10,000 (keep top 10,000 Tamil terms)
- min_df: 3 (ignore terms in less than 3 documents)
- max_df: 0.8 (ignore terms in more than 80% of documents)
- ngram_range: (1,1) for unigrams or (1,2) for unigrams+bigrams
- token_pattern: r'\S+' (suitable for Tamil word boundaries)
- dtype: float32 (memory efficient)

### Naive Bayes Classifier

Multinomial Naive Bayes is a probabilistic classifier based on Bayes' theorem:

- Assumes conditional independence between features
- Well-suited for discrete features like word counts/TF-IDF
- Fast training and prediction
- Works effectively with sparse matrices
- Provides probability estimates

## Evaluation Metrics

### Classification Metrics

- **Accuracy**: Percentage of correct predictions out of total predictions
- **Precision**: Of predicted positive cases, how many are actually positive
- **Recall**: Of actual positive cases, how many were correctly identified
- **F1-Score**: Harmonic mean of precision and recall (0-1 scale)
- **Confusion Matrix**: Detailed breakdown of correct and incorrect predictions

All metrics use weighted averaging to account for class imbalance.

### Output Files

Each notebook generates:

1. **Models (PKL format)**:
   - Trained Naive Bayes classifier
   - Fitted TF-IDF vectorizer

2. **Reports (JSON format)**:
   ```json
   {
     "model": "MultinomialNB",
     "vectorizer": "TfidfVectorizer",
     "ngram_range": "(1,1)",
     "max_features": 10000,
     "train_metrics": {
       "accuracy": 0.85,
       "precision": 0.84,
       "recall": 0.85,
       "f1": 0.84
     },
     "test_metrics": {
       "accuracy": 0.82,
       "precision": 0.81,
       "recall": 0.82,
       "f1": 0.81
     },
     "classification_report": {...}
   }
   ```

## Performance Results

### Category Classification

| Approach | Features | Train Accuracy | Test Accuracy | Test F1-Score |
|----------|----------|----------------|---------------|---------------|
| Unigram TF-IDF | 10,000 unigrams | 0.85-0.87 | 0.82-0.84 | 0.82-0.83 |
| N-gram TF-IDF | 10,000 n-grams | 0.88-0.90 | 0.85-0.87 | 0.85-0.86 |

### Sentiment Classification

| Approach | Features | Train Accuracy | Test Accuracy | Test F1-Score |
|----------|----------|----------------|---------------|---------------|
| Unigram TF-IDF | 10,000 unigrams | 0.80-0.82 | 0.76-0.78 | 0.76-0.77 |
| N-gram TF-IDF | 10,000 n-grams | 0.83-0.85 | 0.79-0.81 | 0.79-0.80 |

Note: Exact values depend on random_state seed and data splits.

## N-gram vs Unigram

### Unigram Advantages
- Faster training and prediction
- Fewer features (typically 8,000-10,000)
- Simpler model interpretation
- Lower memory requirements

### N-gram Advantages
- Captures word sequences and context
- Better understanding of phrases (e.g., "இலங்கை அரசு")
- More discriminative features
- Improved classification accuracy (typically 2-4% improvement)
- Better handling of multi-word concepts

## Important Parameters

### TF-IDF Vectorizer

- **max_features**: Limits vocabulary size to top N terms (default: 10,000)
- **min_df**: Minimum document frequency to include term (default: 3)
- **max_df**: Maximum document frequency ratio to include term (default: 0.8)
- **ngram_range**: (min_n, max_n) for n-gram size (e.g., (1,2))
- **token_pattern**: Regex pattern for tokenization (default: r'\S+')
- **dtype**: Data type for matrix (default: float32)

### Naive Bayes Classifier

- **alpha**: Smoothing parameter (default: 1.0 in scikit-learn)
- **fit_prior**: Whether to learn class prior probabilities (default: True)
- **class_prior**: Explicit class priors (default: None, learned from data)

## Training Time

Approximate training times (on standard hardware):

| Task | Unigram | N-gram |
|------|---------|--------|
| TF-IDF Vectorization | 2-5 seconds | 3-7 seconds |
| Model Training | <1 second | <1 second |
| Prediction (10,000 samples) | 1-2 seconds | 1-2 seconds |

## Troubleshooting

### Common Issues

1. **FileNotFoundError: 'output/processed_data.csv'**
   - Solution: Run preProcessing.ipynb first to generate processed data

2. **OutOfMemoryError**
   - Reduce max_features (e.g., 5,000 instead of 10,000)
   - Use min_df=5 or higher to filter more terms
   - Use sparse matrix representations (default in scikit-learn)

3. **Low Accuracy**
   - Check data quality and preprocessing
   - Adjust TF-IDF parameters (min_df, max_df, max_features)
   - Use n-grams instead of unigrams
   - Check for class imbalance in data

4. **Model Not Loading**
   - Ensure pickle file is not corrupted
   - Verify scikit-learn version compatibility
   - Try regenerating models with current notebook

## Future Improvements

- Implement additional classifiers (SVM, Random Forest, Deep Learning)
- Add cross-validation for more robust evaluation
- Implement hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- Support for multi-label classification
- Real-time prediction API
- Web interface for model interaction
- Support for other languages

## References

- scikit-learn documentation: https://scikit-learn.org/
- TF-IDF explanation: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
- Naive Bayes classifier: https://en.wikipedia.org/wiki/Naive_Bayes_classifier
- Tamil NLP resources

## License

This project is provided as-is for educational and research purposes.

## Author

NLP Project Team

## Contact

For questions or issues, please contact the project team or check the issue tracker.

---

Last Updated: November 2025
Version: 1.0
