
# SentimentAnalysis

A Python project for sentiment analysis of movie reviews using data from IMDB.com. The classifier determines whether a review expresses positive or negative sentiment. The project explores different preprocessing and feature extraction techniques, and evaluates their effectiveness using K-Fold cross-validation.

## Features
- Tokenizes reviews with options for handling punctuation
- Removes repeated words
- Extracts features: token counts, token pairs, and sentiment lexicon features
- Supports multiple feature combinations and minimum frequency thresholds
- Uses Logistic Regression for classification
- Evaluates models with K-Fold cross-validation
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
2. **Run the script:**
   ```bash
   python SentimentAnalysis.py
   ```
   The script will download the IMDB dataset if not already present, preprocess the data, train and evaluate models, and print results to the console. It will also generate `accuracies.png`.

## Output
- Prints best and worst cross-validation results
- Shows mean accuracy per setting
- Displays top coefficients (features) for each class
- Reports test set accuracy
- Prints the most misclassified test documents

## Customization
- You can modify feature extraction or add new features in `SentimentAnalysis.py`
- Adjust the list of features, minimum frequency, or cross-validation folds as needed

## License
This project is for educational purposes.

