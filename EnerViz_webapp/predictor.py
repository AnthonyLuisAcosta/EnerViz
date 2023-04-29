import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import sys
import csv
import os

print('\n\n Classifiying Sentiment...')

# Load the saved SVM model from file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
   vectorizer = pickle.load(f)
   
# Read preprocessed data from a CSV file into a Pandas DataFrame and drop NaN values
df = pd.read_csv('Datasets/preprocessed_data.csv').dropna()

# Apply trained model to test data
test_tfidf =  vectorizer.transform(df['preprocessed_comments'])
pred = model.predict(test_tfidf)
df['sentiment'] = pred 

# Save the DataFrame to a CSV file and force-save the file
# Save the DataFrame to a CSV file and force-save the file
with open('Datasets/classified_data.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(df.columns)
    for _, row in df.iterrows():
        writer.writerow(row)
    file.flush()
    os.fsync(file.fileno())

print('Done!')