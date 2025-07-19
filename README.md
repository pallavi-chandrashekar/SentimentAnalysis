
# SentimentAnalysis

A Python project for sentiment analysis of movie reviews using data from IMDB.com. The classifier determines whether a review expresses positive or negative sentiment. The project explores different preprocessing and feature extraction techniques, and evaluates their effectiveness using K-Fold cross-validation.

## Features
- Tokenizes reviews with options for handling punctuation
- Removes repeated words
- Extracts features: token counts, token pairs, and sentiment lexicon features
- Supports TF-IDF and n-gram features (unigrams, bigrams, etc.)
- Supports multiple feature combinations and minimum frequency thresholds
- Supports multiple classifiers: Logistic Regression, SVM, Random Forest, Naive Bayes
- Evaluates models with K-Fold cross-validation
- Computes accuracy, precision, recall, F1, and confusion matrix
- Plots accuracy results for different settings
- Analyzes top features and misclassified examples

## Project Structure
- `SentimentAnalysis.py`: Main script containing all logic for data processing, feature extraction, model training, and evaluation
- `data/`: Directory for data files
    - `test/`: Test data (with subfolders for negative reviews)
- `imdb.tgz`: Compressed IMDB dataset (downloaded automatically if not present)
- `accuracies.png`: Plot of model accuracies for different settings

## Usage
1. **Install dependencies:**
   ```bash
   pip install numpy scipy scikit-learn matplotlib
   ```
2. **Run the script (classic features):**
   ```bash
   python SentimentAnalysis.py --mode classic --model logreg
   ```
   Or, to use TF-IDF and n-gram features with a different model:
   ```bash
   python SentimentAnalysis.py --mode tfidf --ngram_range 1,2 --model svm
   ```
   Available models: `logreg` (Logistic Regression), `svm` (Support Vector Machine), `rf` (Random Forest), `nb` (Naive Bayes)
   (The above uses unigrams and bigrams. You can set any n-gram range, e.g., 1,3 for up to trigrams.)
   The script will download the IMDB dataset if not already present, preprocess the data, train and evaluate models, and print results to the console. It will also generate `accuracies.png` (for classic mode).

## Output
- Prints best and worst cross-validation results
- Shows mean accuracy per setting
- Displays top coefficients (features) for each class (if supported)
- Reports test set accuracy, precision, recall, F1, and confusion matrix
- Prints the most misclassified test documents

## Customization & Improvements
- You can modify feature extraction or add new features in `SentimentAnalysis.py`
- Adjust the list of features, minimum frequency, or cross-validation folds as needed
- Try different models and n-gram ranges
- Add more datasets or improve data cleaning
- Add more visualizations (feature importance, ROC curves)
- Add unit tests or notebook support for interactive exploration

## License
This project is for educational purposes.

