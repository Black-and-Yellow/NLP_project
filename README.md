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
├── README.md                          # This filesee 
├── tf-idf.ipynb                       # Unigram TF-IDF with Naive Bayes
├── tf-idf_n-gram.ipynb                # N-gram TF-IDF with Naive Bayes
├── preProcessing.ipynb                # Data preprocessing and cleaning
├── vectorization_and_model.ipynb      # TF-IDF from scratch (unigrams) + All models
├── vectorization_and_model_n-gram.ipynb # TF-IDF from scratch (n-grams) + All models
├── predict_headline.ipynb             # 🆕 Prediction system for new headlines
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

### 3. predict_headline.ipynb

**🆕 Multi-Model Prediction System for New Headlines**

This notebook provides a complete prediction system for classifying new Tamil news headlines:

- Loads ALL trained models (Naive Bayes, SVM, Logistic Regression)
- Tamil text preprocessing pipeline
- TF-IDF vectorization using saved vectorizers
- Dual classification: Category AND Sentiment
- Shows actual model accuracies from training
- Single and batch prediction modes

**Key Features:**
1. **Single Prediction**: Classify one headline at a time
2. **Batch Prediction**: Process multiple headlines and save to CSV
3. **Model Performance Display**: Shows accuracy and F1-score for each model
4. **Complete Preprocessing**: Same pipeline used during training
5. **All 6 Models**: Uses all trained models for comprehensive predictions

**Usage Examples:**
```python
# Single prediction
predict_headline("உங்கள் தமிழ் செய்தி தலைப்பு")

# Batch prediction
headlines = ["தலைப்பு 1", "தலைப்பு 2", "தலைப்பு 3"]
results = predict_batch(headlines, save_to_csv=True)
```

**Output Format:**
- Category predictions from 3 models (Naive Bayes, SVM, Logistic Regression)
- Sentiment predictions from 3 models (Naive Bayes, SVM, Logistic Regression)
- Model accuracy and F1-score for each prediction
- CSV export for batch predictions

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

2. **Model Training - Choose one approach**:
   ```
   Option A: Run tf-idf.ipynb (Naive Bayes only)
   Option B: Run vectorization_and_model.ipynb (All models: NB, SVM, LR)
   ```

3. **Making Predictions on New Headlines**:
   ```
   Run predict_headline.ipynb to classify new Tamil news headlines
   ```
   - Loads all trained models automatically
   - Supports single and batch predictions
   - Shows model accuracies alongside predictions

### Making Predictions on New Headlines

**Method 1: Use the Prediction Notebook (Recommended)**

```python
# Open and run predict_headline.ipynb
# Then use:
predict_headline("உங்கள் தமிழ் செய்தி தலைப்பு")
```

**Method 2: Load Models Manually**

```python
import pickle

# Load category model (choose one)
with open('models/category_naive_bayes.pkl', 'rb') as f:
    cat_model = pickle.load(f)
# Or: category_svm.pkl, category_logistic.pkl

# Load sentiment model (choose one)
with open('models/sentiment_naive_bayes.pkl', 'rb') as f:
    sent_model = pickle.load(f)
# Or: sentiment_svm.pkl, sentiment_logistic.pkl

# Load vectorizers
with open('models/category_vectorizer.pkl', 'rb') as f:
    cat_vectorizer_data = pickle.load(f)
with open('models/sentiment_vectorizer.pkl', 'rb') as f:
    sent_vectorizer_data = pickle.load(f)

# Note: Vectorizers are dictionaries with vocabulary, word2idx, and idf_dict
# You'll need to implement TF-IDF transformation (see predict_headline.ipynb)
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

### Category Classification (All Models)

| Model | Features | Train Accuracy | Test Accuracy | Test F1-Score |
|-------|----------|----------------|---------------|---------------|
| **Naive Bayes** | 10,000 unigrams | 83.40% | **64.55%** | **0.6390** |
| **Linear SVM** | 10,000 unigrams | 92.26% | **66.21%** | **0.6570** |
| **Logistic Regression** | 10,000 unigrams | 92.78% | **65.20%** | **0.6512** |

**Best Model for Category**: Linear SVM (66.21% accuracy)

### Sentiment Classification (All Models)

| Model | Features | Train Accuracy | Test Accuracy | Test F1-Score |
|-------|----------|----------------|---------------|---------------|
| **Naive Bayes** | 10,000 unigrams | 96.85% | **65.25%** | **0.6507** |
| **Linear SVM** | 10,000 unigrams | 100.00% | **67.19%** | **0.6613** |
| **Logistic Regression** | 10,000 unigrams | 99.70% | **68.40%** | **0.6571** |

**Best Model for Sentiment**: Logistic Regression (68.40% accuracy)

### Model Comparison Notes

- **Linear SVM**: Best for category classification, strong generalization
- **Logistic Regression**: Best for sentiment classification, excellent performance
- **Naive Bayes**: Fast training, good baseline performance
- All models use TF-IDF vectorization with 10,000 features
- Train-test split: 80-20 with stratification
- Dataset: Tamil news articles with preprocessing

## Model Selection Guide

### Which Model to Use?

**For Category Classification:**
- **Best Overall**: Linear SVM (66.21% accuracy, F1=0.6570)
- **Fastest**: Naive Bayes (64.55% accuracy, F1=0.6390)
- **Balanced**: Logistic Regression (65.20% accuracy, F1=0.6512)

**For Sentiment Classification:**
- **Best Overall**: Logistic Regression (68.40% accuracy, F1=0.6571)
- **Runner-up**: Linear SVM (67.19% accuracy, F1=0.6613)
- **Fastest**: Naive Bayes (65.25% accuracy, F1=0.6507)

### Model Characteristics

**Naive Bayes:**
- ✅ Fastest training and prediction
- ✅ Works well with small datasets
- ✅ Probabilistic predictions
- ⚠️ Assumes feature independence

**Linear SVM:**
- ✅ Excellent for high-dimensional data
- ✅ Good generalization
- ✅ Robust to overfitting
- ⚠️ Slower than Naive Bayes

**Logistic Regression:**
- ✅ Balanced performance and speed
- ✅ Interpretable coefficients
- ✅ Probabilistic predictions
- ✅ Works well with regularization

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

- ✅ **Implemented**: Multiple classifiers (Naive Bayes, SVM, Logistic Regression)
- ✅ **Implemented**: Prediction system for new headlines
- Add cross-validation for more robust evaluation
- Implement hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- Add N-gram models to prediction system
- Support for multi-label classification
- Real-time prediction API
- Web interface for model interaction
- Deep learning models (LSTM, BERT for Tamil)
- Support for other languages
- Model explainability (feature importance, LIME/SHAP)

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
