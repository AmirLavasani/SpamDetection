# SpamDetection
This is a repositor for sms spam detection in Persian language. 

# Data
we gathered data from more than 8,000 SMS. Labeled them manually accordingly in to two classes: spam and not-spam. spam class has 4342 samples and not-spam class has 2982 samples. I normalized the data in terms of unicodes, ommiting punchuations and truncated the length of the sentences to 30 words. Data samples are in the Data folder.

# Feature Extraction
I used a 64 dimension word embedding. word embeddings obtained using python Polyglot NLP tool.

# Architecture
Two models trained. One with scikitlearn tool and using svm and the other using tensorflow and using a multilayer lstm architecture. 

# Accuracy
svm reached an average of 89 percent and the deep lstm model reached around 96 percent. 

# Pre-Trained models
In the model folder you can find a pretrained model for tensorflow architecture. svm model will be uploaded soon. 

# usage
Tensorflow model for training:
`python tf_spam_classifier.py --spam data/spam_norm.txt --not-spam data/not_spam_norm.txt --train-flag True`
Tensorflow model for prediction:
`python tf_spam_classifier.py`
SVM model train then prediction:
`python svm.py`

# Requirements
* polyglot
* numpy
* scikitlearn
* tqdm
* click
* tensorflow > 1.2

