"""
A text classifier to determine whether a
movie review is expressing positive or negative sentiment. The data come from
the website IMDB.com.

will write code to preprocess the data in different ways (creating different
features), then compare the cross-validation accuracy of each approach. Then,
we will compute accuracy on a test set and do some analysis of the errors.

"""


from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

import string
import tarfile
import urllib.request
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def download_data():
    """ Download and unzip data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/8oehplrobcgi9cq/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


def read_data(path):
    """
    Walks all subdirectories of this path and reads all
    the text files and labels.
    DONE ALREADY.

    Params:
      path....path to files
    Returns:
      docs.....list of strings, one per document
      labels...list of ints, 1=positive, 0=negative label.
               Inferred from file path (i.e., if it contains
               'pos', it is 1, else 0)
    """
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def tokenize(doc, keep_internal_punct=False):
    """
    Tokenize a string.
    The string should be converted to lowercase.
    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).
    If keep_internal_punct is True, then also retain punctuation that
    is inside of a word. E.g., in the example below, the token "isn't"
    is maintained when keep_internal_punct=True; otherwise, it is
    split into "isn" and "t" tokens.

    Params:
      doc....a string.
      keep_internal_punct...see above
    Returns:
      a numpy array containing the resulting tokens.

    >>> tokenize(" Hi there! Isn't this fun?", keep_internal_punct=False)
    array(['hi', 'there', 'isn', 't', 'this', 'fun'], dtype='<U5')
    >>> tokenize("Hi there! Isn't this fun? ", keep_internal_punct=True)
    array(['hi', 'there', "isn't", 'this', 'fun'], dtype='<U5')
    """
     
    lowerCaseInput = doc.lower()

    if keep_internal_punct == False:
        finalResult = re.sub('\W+', ' ', lowerCaseInput).split() 
    else:
        finalResult = [re.sub('^\W+|\W+$', '', x) for x in lowerCaseInput.split()]

    return np.array(finalResult)

def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    Note that the feats dict is modified in place,
    so there is no return value.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_features(['hi', 'there', 'hi'], feats)
    >>> sorted(feats.items())
    [('token=hi', 2), ('token=there', 1)]
    """
    ###TODO
    for i in tokens:
        feats["token="+i] += 1


def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.

    For example [a, b, c, d] with k=3 will consider the
    windows: [a,b,c], [b,c,d]. In the first window,
    a_b, a_c, and b_c appear; in the second window,
    b_c, c_d, and b_d appear. This example is in the
    doctest below.
    Note that the order of the tokens in the feature name
    matches the order in which they appear in the document.
    (e.g., a__b, not b__a)

    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)
    >>> sorted(feats.items())
    [('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1), ('token_pair=c__d', 1)]
    """
    
    windows = []
    combList = []
    for i in range(0, len(tokens) - (k - 1)):
        windows.append(tokens[i : i + k])
    for ii in windows:
        for jj in list(combinations(ii, 2)):
            combList.append(jj[0] + '__' + jj[1])
    counterList = Counter(['token_pair=' + s for s in combList])
    for ii, jj in counterList.items():
        feats[ii] = jj


neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])

def lexicon_features(tokens, feats):
    """
    Add features indicating how many time a token appears that matches either
    the neg_words or pos_words (defined above). The matching should ignore
    case.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    In this example, 'LOVE' and 'great' match the pos_words,
    and 'boring' matches the neg_words list.
    >>> feats = defaultdict(lambda: 0)
    >>> lexicon_features(np.array(['i', 'LOVE', 'this', 'great', 'boring', 'movie']), feats)
    >>> sorted(feats.items())
    [('neg_words', 1), ('pos_words', 2)]
    """
    
    feats['neg_words'] = 0
    feats['pos_words'] = 0
    for t in tokens:
        if t.lower() in neg_words:
            feats['neg_words'] += 1
        if t.lower() in pos_words:
            feats['pos_words'] += 1


def featurize(tokens, feature_fns):
    """
    Compute all features for a list of tokens from
    a single document.

    Params:
      tokens........array of token strings from a document.
      feature_fns...a list of functions, one per feature
    Returns:
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.

    >>> feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie']), [token_features, lexicon_features])
    >>> feats
    [('neg_words', 0), ('pos_words', 2), ('token=LOVE', 1), ('token=great', 1), ('token=i', 1), ('token=movie', 1), ('token=this', 1)]
    """
    
    featsValue = defaultdict(lambda: 0)
    for f in feature_fns:
        f(tokens, featsValue)
    return sorted(featsValue.items(),key=lambda x: x[0])


def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """
    Given the tokens for a set of documents, create a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.

    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix: See https://goo.gl/f5TiF1 for documentation.
      This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index. NOTE
      that the columns are sorted alphabetically (so, the feature
      "token=great" is column 0 and "token=horrible" is column 1
      because "great" < "horrible" alphabetically),

    When vocab is None, we build a new vocabulary from the given data.
    when vocab is not None, we do not build a new vocab, and we do not
    add any new terms to the vocabulary. This setting is to be used
    at test time.

    >>> docs = ["Isn't this movie great?", "Horrible, horrible movie"]
    >>> tokens_list = [tokenize(d) for d in docs]
    >>> feature_fns = [token_features]
    >>> X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)
    >>> type(X)
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> X.toarray()
    array([[1, 0, 1, 1, 1, 1],
           [0, 2, 0, 1, 0, 0]], dtype=int64)
    >>> sorted(vocab.items(), key=lambda x: x[1])
    [('token=great', 0), ('token=horrible', 1), ('token=isn', 2), ('token=movie', 3), ('token=t', 4), ('token=this', 5)]
    """
    
    featsCount = defaultdict(lambda: 0)
    fet = []
    featureList = []
    for t in tokens_list:
        feats = featurize(t, feature_fns)
        fet.append(feats)
        for f, c in feats:
            featsCount[f] += 1

    if vocab == None:
        featureList = []
        for f in featsCount:
            if featsCount[f] >= min_freq:
                featureList.append(f)
        sortedFeats = sorted(featureList)
        vocab = {}
        for i, f in enumerate(sortedFeats):
            vocab[f] = i

    document = []
    features = []
    data = []
    for i in range(len(fet)):
        for j in range(len(fet[i])):
            if fet[i][j][0] in vocab:
                document.extend([i])
                features.extend([vocab[fet[i][j][0]]])
                data.extend([fet[i][j][1]])
    matrixValue = csr_matrix((data, (document, features)),shape=(len(tokens_list), len(vocab)), dtype= np.int64)
    return matrixValue, vocab


def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation. You
    can use sklearn's KFold class here (no random seed, and no shuffling
    needed).

    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.

    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    
    value = KFold(n_splits = k, shuffle = False, random_state = 42)
    accValue = []
    for trainIndex, testIndex in value.split(X):
        clf.fit(X[trainIndex], labels[trainIndex])
        preValue = clf.predict(X[testIndex])
        accValue.append(accuracy_score(labels[testIndex], preValue))
    return np.mean(accValue)


def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    """
    Enumerate all possible classifier settings and compute the
    cross validation accuracy for each setting. We will use this
    to determine which setting has the best accuracy.

    For each setting, construct a LogisticRegression classifier
    and compute its cross-validation accuracy for that setting.

    In addition to looping over possible assignments to
    keep_internal_punct and min_freqs, we will enumerate all
    possible combinations of feature functions. So, if
    feature_fns = [token_features, token_pair_features, lexicon_features],
    then we will consider all 7 combinations of features (see Log.txt
    for more examples).

    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])

    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.

      This list should be SORTED in descending order of accuracy.

      This function will take a bit longer to run (~20s for me).
    """
    
    listOfDicts = []
    allFeatures = []
    for i in range(len(feature_fns)):
        for j in combinations(feature_fns, i + 1):
            allFeatures.append(list(j))
    for p in punct_vals:
        tokens_list = [tokenize(d, p) for d in docs]
        for freqs in min_freqs:
            for features in allFeatures:
                    X, vocab = vectorize(tokens_list, features, freqs, None)
                    accuracies = cross_validation_accuracy(LogisticRegression(), X, labels, 5)
                    dic ={}
                    dic['punct']= p
                    dic['features']= features
                    dic['min_freq']= freqs
                    dic['accuracy']= accuracies
                    listOfDicts.append(dic)
    return sorted(listOfDicts, key = lambda x : x['accuracy'], reverse = True)


def plot_sorted_accuracies(results):
    """
    Plot all accuracies from the result of eval_all_combinations
    in ascending order of accuracy.
    Save to "accuracies.png".
    """
   
    Accuracy = []
    for i in results:
        Accuracy.append(i['accuracy'])
    sortedAccuracy = sorted(Accuracy)
    plt.xlabel('setting')
    plt.ylabel('accuracy')
    plt.plot(sortedAccuracy)
    plt.savefig('accuracies.png')


def mean_accuracy_per_setting(results):
    """
    To determine how important each model setting is to overall accuracy,
    we'll compute the mean accuracy of all combinations with a particular
    setting. For example, compute the mean accuracy of all runs with
    min_freq=2.

    Params:
      results...The output of eval_all_combinations
    Returns:
      A list of (accuracy, setting) tuples, SORTED in
      descending order of accuracy.
    """
    
    listOfMeanValues = []
    count = defaultdict(list)
    for i in results:
        featureName = 'features=' + ' '.join(i.__name__ for i in i['features'])
        p_name = 'punct=' + str(i['punct'])
        minFrequencyValue = 'min_freq=' + str(i['min_freq'])
        count.setdefault(featureName,[])
        count[featureName].append(i['accuracy'])
        count.setdefault(p_name,[])
        count[p_name].append(i['accuracy'])
        count.setdefault(minFrequencyValue,[])
        count[minFrequencyValue].append(i['accuracy'])
    for k, v in count.items():
        accMean = np.mean(v)
        listOfMeanValues.append((accMean, k))
    return sorted(listOfMeanValues, key=lambda x: x[0], reverse = True)


def fit_best_classifier(docs, labels, best_result):
    """
    Using the best setting from eval_all_combinations,
    re-vectorize all the training data and fit a
    LogisticRegression classifier to all training data.
    (i.e., no cross-validation done here)

    Params:
      docs..........List of training document strings.
      labels........The true labels for each training document (0 or 1)
      best_result...Element of eval_all_combinations
                    with highest accuracy
    Returns:
      clf.....A LogisticRegression classifier fit to all
            training data.
      vocab...The dict from feature name to column index.
    """
    
    punct = best_result['punct']
    featureValue = best_result['features']
    minFrequency = best_result['min_freq']
    listOfTokens = []
    for i in docs:
        listOfTokens.append(tokenize(i, punct))
    val, vocab = vectorize(listOfTokens, featureValue, minFrequency, vocab = None)
    classifier= LogisticRegression().fit(val,labels)
    return (classifier, vocab)


def top_coefs(clf, label, n, vocab):
    """
    Find the n features with the highest coefficients in
    this classifier for this label.
    See the .coef_ attribute of LogisticRegression.

    Params:
      clf.....LogisticRegression classifier
      label...1 or 0; if 1, return the top coefficients
              for the positive class; else for negative.
      n.......The number of coefficients to return.
      vocab...Dict from feature name to column index.
    Returns:
      List of (feature_name, coefficient) tuples, SORTED
      in descending order of the coefficient for the
      given class label.
    """
    
    listOfCoefficient = []
    coef = clf.coef_[0]
    idx_Word = dict((v,k) for k,v in vocab.items())
    if label == 1:
        topPosIdx = np.argsort(coef)[::-1][:n]
        for i in topPosIdx:
            featureName = idx_Word[i]
            coefficient = coef[i]
            listOfCoefficient.append((featureName, coefficient))
    if label == 0:
        negativeIndex = np.argsort(coef)[:n]
        for i in negativeIndex:
            featureName = idx_Word[i]
            coefficient = abs(coef[i])
            listOfCoefficient.append((featureName, coefficient))
    return listOfCoefficient


def parse_test_data(best_result, vocab):
    """
    Using the vocabulary fit to the training data, read
    and vectorize the testing data. Note that vocab should
    be passed to the vectorize function to ensure the feature
    mapping is consistent from training to testing.

    Note: use read_data function defined above to read the
    test data.

    Params:
      best_result...Element of eval_all_combinations
                    with highest accuracy
      vocab.........dict from feature name to column index,
                    built from the training data.
    Returns:
      test_docs.....List of strings, one per testing document,
                    containing the raw.
      test_labels...List of ints, one per testing document,
                    1 for positive, 0 for negative.
      X_test........A csr_matrix representing the features
                    in the test data. Each row is a document,
                    each column is a feature.
    """
    
    punct = best_result['punct']
    featureValue = best_result['features']
    minFrequency = best_result['min_freq']
    testDocs, testLabels = read_data(os.path.join('data', 'test'))
    value=[tokenize(d,punct) for d in testDocs]
    Xtest, vocab=vectorize(value, featureValue, minFrequency, vocab)
    return (testDocs, testLabels, Xtest)



def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    """
    Print the n testing documents that are misclassified by the
    largest margin. By using the .predict_proba function of
    LogisticRegression <https://goo.gl/4WXbYA>, we can get the
    predicted probabilities of each class for each instance.
    We will first identify all incorrectly classified documents,
    then sort them in descending order of the predicted probability
    for the incorrect class.
    E.g., if document i is misclassified as positive, we will
    consider the probability of the positive class when sorting.

    Params:
      test_docs.....List of strings, one per test document
      test_labels...Array of true testing labels
      X_test........csr_matrix for test data
      clf...........LogisticRegression classifier fit on all training
                    data.
      n.............The number of documents to print.

    Returns:
      Nothing; see Log.txt for example printed output.
    """
    
    probaValue = clf.predict_proba(X_test)
    predictValue = clf.predict(X_test)
    finalValue = []
    for i in range(len(test_labels)):
        if predictValue[i] != test_labels[i]:
            if predictValue[i] == 0:
                finalValue.append((test_labels[i],predictValue[i],probaValue[i][0],test_docs[i]))
            else:
                finalValue.append((test_labels[i],predictValue[i],probaValue[i][1],test_docs[i]))
    sortedfinalValue = sorted(finalValue,key=lambda x: -x[2])
    for i in sortedfinalValue[:n]:
        print("\nTruth=%d Predicted=%d Proba=%.6f\n%s" % (i[0], i[1], i[2], i[3]))



def main():
    """
    Main entry point. Now supports classic and TF-IDF/n-gram feature extraction.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Sentiment Analysis IMDB")
    parser.add_argument('--mode', choices=['classic', 'tfidf'], default='classic',
                        help='Feature extraction mode: classic (default) or tfidf')
    parser.add_argument('--ngram_range', type=str, default='1,1',
                        help='n-gram range for tfidf mode, e.g., "1,2" for unigrams and bigrams')
    args = parser.parse_args()

    # Download and read data.
    download_data()
    docs, labels = read_data(os.path.join('data', 'train'))

    if args.mode == 'classic':
        feature_fns = [token_features, token_pair_features, lexicon_features]
        results = eval_all_combinations(docs, labels,
                                        [True, False],
                                        feature_fns,
                                        [2,5,10])
        best_result = results[0]
        worst_result = results[-1]
        print('best cross-validation result:\n%s' % str(best_result))
        print('worst cross-validation result:\n%s' % str(worst_result))
        plot_sorted_accuracies(results)
        print('\nMean Accuracies per Setting:')
        print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))

        # Fit best classifier.
        clf, vocab = fit_best_classifier(docs, labels, results[0])

        # Print top coefficients per class.
        print('\nTOP COEFFICIENTS PER CLASS:')
        print('negative words:')
        print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
        print('\npositive words:')
        print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))

        # Parse test data
        test_docs, test_labels, X_test = parse_test_data(best_result, vocab)
        predictions = clf.predict(X_test)
        print('testing accuracy=%f' % accuracy_score(test_labels, predictions))
        print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
        print_top_misclassified(test_docs, test_labels, X_test, clf, 5)

    elif args.mode == 'tfidf':
        ngram_range = tuple(int(x) for x in args.ngram_range.split(','))
        print(f"Using TfidfVectorizer with ngram_range={ngram_range}")
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=ngram_range, stop_words='english')
        X = vectorizer.fit_transform(docs)
        clf = LogisticRegression(max_iter=200)
        kf = KFold(n_splits=5, shuffle=False, random_state=42)
        accs = []
        for train_idx, test_idx in kf.split(X):
            clf.fit(X[train_idx], labels[train_idx])
            preds = clf.predict(X[test_idx])
            accs.append(accuracy_score(labels[test_idx], preds))
        print(f"Mean cross-validation accuracy: {np.mean(accs):.4f}")

        # Fit on all data
        clf.fit(X, labels)
        feature_names = np.array(vectorizer.get_feature_names_out())
        coefs = clf.coef_[0]
        top_pos = np.argsort(coefs)[-5:][::-1]
        top_neg = np.argsort(coefs)[:5]
        print('\nTOP COEFFICIENTS PER CLASS:')
        print('negative words:')
        for idx in top_neg:
            print(f'{feature_names[idx]}: {abs(coefs[idx]):.5f}')
        print('\npositive words:')
        for idx in top_pos:
            print(f'{feature_names[idx]}: {coefs[idx]:.5f}')

        # Test set
        test_docs, test_labels = read_data(os.path.join('data', 'test'))
        X_test = vectorizer.transform(test_docs)
        preds = clf.predict(X_test)
        print('testing accuracy=%f' % accuracy_score(test_labels, preds))
        # Print top misclassified
        probs = clf.predict_proba(X_test)
        misclassified = []
        for i in range(len(test_labels)):
            if preds[i] != test_labels[i]:
                prob = probs[i][preds[i]]
                misclassified.append((test_labels[i], preds[i], prob, test_docs[i]))
        misclassified = sorted(misclassified, key=lambda x: -x[2])
        print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
        for i in misclassified[:5]:
            print(f"\nTruth={i[0]} Predicted={i[1]} Proba={i[2]:.6f}\n{i[3]}")


if __name__ == '__main__':
    main()
