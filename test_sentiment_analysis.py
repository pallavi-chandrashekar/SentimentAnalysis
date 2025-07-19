import unittest
import numpy as np
from SentimentAnalysis import clean_text, tokenize, accuracy_score

class TestSentimentAnalysis(unittest.TestCase):
    def test_clean_text_html(self):
        self.assertEqual(clean_text('This is <b>great</b>!'), 'This is great!')

    def test_clean_text_whitespace(self):
        self.assertEqual(clean_text('This   is\n\tgood.'), 'This is good.')

    def test_tokenize_basic(self):
        tokens = tokenize('Hello, world!', keep_internal_punct=False)
        self.assertTrue('hello' in tokens and 'world' in tokens)

    def test_accuracy_score(self):
        y_true = np.array([1, 0, 1, 1])
        y_pred = np.array([1, 0, 0, 1])
        self.assertAlmostEqual(accuracy_score(y_true, y_pred), 0.75)

if __name__ == '__main__':
    unittest.main()
