from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

import numpy as np

# Load the pre-classified training data as "training-fanfic/clean" and "training-fanfic/dirty/"
train = load_files("training-fanfic", encoding="utf-8", load_content=True)

# Set up a pipeline to vectorize the words with a weighted frequency, and then
# use the classifing algorithm suggested by the scikit-learn documentation for this problem
text_clf = Pipeline([('vect', TfidfVectorizer(analyzer='word', encoding="utf-8", stop_words="english")),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=0.001, n_iter=5, random_state=42)),
                     ])

# Generate a model that "fits" the training data
text_clf = text_clf.fit(train.data, train.target)

# Now load the data we'll test again, same layout as before
test = load_files("testing-fanfic", encoding="utf-8", load_content=True)

# Magic happens here
predicted = text_clf.predict(test.data)

# Generate an overall average performance score (but break this down carefully
# to check our work!)
print("Test mean: " + str(np.mean(predicted == test.target)))
