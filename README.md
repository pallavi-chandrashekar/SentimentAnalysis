

# SentimentAnalysis

SentimentAnalysis is a Python project for classifying IMDB movie reviews as positive or negative. It supports multiple feature extraction techniques, advanced data cleaning, several classifiers, detailed evaluation metrics, and both script and notebook-based workflows.

## Features
- **Advanced Data Cleaning:** Removes HTML tags, normalizes whitespace, and strips emojis for robust text preprocessing.
- **Flexible Tokenization:** Options for handling punctuation and repeated words.
- **Feature Extraction:**
  - Classic features: token counts, token pairs, sentiment lexicon features
  - TF-IDF and n-gram features (unigrams, bigrams, trigrams, etc.)
- **Multiple Classifiers:** Logistic Regression, SVM, Random Forest, Naive Bayes
- **Combinatorial Feature Search:** Tests all combinations of feature functions and minimum frequency thresholds
- **Evaluation:**
  - K-Fold cross-validation
  - Accuracy, precision, recall, F1, confusion matrix
  - Top misclassified examples
- **Visualization:**
  - Plots sorted accuracy results for all settings (`accuracies.png`)
  - Feature importance bar plots (Logistic Regression, Random Forest)
  - ROC curve for supported models (`roc_curve.png`)
- **Interactive Notebook:** Jupyter notebook template for step-by-step exploration and demonstration
- **Unit Testing:** Basic unit tests for core functions

## Project Structure
- `SentimentAnalysis.py`: Main script for data processing, feature extraction, model training, evaluation, and visualization
- `SentimentAnalysis_Demo.ipynb`: Jupyter notebook for interactive analysis and demonstration
- `test_sentiment_analysis.py`: Unit tests for core functions
- `data/`: Directory for data files
  - `test/`: Test data (with subfolders for negative reviews)
- `imdb.tgz`: Compressed IMDB dataset (downloaded automatically if not present)
- `accuracies.png`: Plot of model accuracies for different settings (classic mode)
- `roc_curve.png`: ROC curve plot (if supported by model)

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
   - Available models: `logreg` (Logistic Regression), `svm` (Support Vector Machine), `rf` (Random Forest), `nb` (Naive Bayes)
   - The above uses unigrams and bigrams. You can set any n-gram range, e.g., `1,3` for up to trigrams.
   - The script will download the IMDB dataset if not already present, preprocess the data, train and evaluate models, and print results to the console. It will also generate `accuracies.png` (for classic mode) and `roc_curve.png` (if supported).

3. **Interactive exploration (Jupyter notebook):**
   Open `SentimentAnalysis_Demo.ipynb` in Jupyter or VS Code to run and modify each step interactively, including data loading, feature engineering, model training, and visualization.

4. **Run unit tests:**
   ```bash
   python -m unittest test_sentiment_analysis.py
   ```

## Output
- Prints best and worst cross-validation results
- Shows mean accuracy per setting
- Displays top coefficients (features) for each class (if supported)
- Reports test set accuracy, precision, recall, F1, and confusion matrix
- Prints the most misclassified test documents
- Saves accuracy and ROC plots as PNG files
- Shows feature importance bar plots for supported models

## Customization & Improvements
- Modify or add feature extraction functions in `SentimentAnalysis.py`
- Adjust the list of features, minimum frequency, or cross-validation folds as needed
- Try different models and n-gram ranges
- Improve data cleaning or add new preprocessing steps
- Add more visualizations (e.g., confusion matrix heatmaps)
- Expand unit tests for more robust validation
- Use the Jupyter notebook for rapid prototyping and demonstration

## License
This project is for educational purposes.

