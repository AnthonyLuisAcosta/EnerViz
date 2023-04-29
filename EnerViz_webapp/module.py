# print execution time
# import the required libraries
import sys
import pandas as pd
import numpy as np
import re
import time
import dask.dataframe as dd
from collections import Counter
from dask.distributed import Client
import multiprocessing
from nltk.stem import PorterStemmer
import json
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import MWETokenizer
import warnings
import logging
logging.getLogger('distributed').setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="Port 8787 is already in use.")

start_time = time.time()

def spelling(df):
    
    df = pd.DataFrame(df)

    dataset = df
    # Reset index
    dataset = dataset.reset_index(drop=True)

    # remove the extra spaces
    dataset['preprocessed_comments'] = dataset['raw_comments'].str.strip()

    # remove url
    # define a regular expression pattern to match URLs
    pattern = r'(https?://\S+|www\.\S+)'

    # function to remove URLs from text
    def remove_urls(text):
        return re.sub(pattern, '', text)

    # apply the function to the 'text' column of the dataframe
    dataset['preprocessed_comments'] = dataset['preprocessed_comments'].astype(
        str)
    dataset['preprocessed_comments'] = dataset['preprocessed_comments'].apply(
        remove_urls)

    # remove punctuation
    dataset['preprocessed_comments'] = dataset['preprocessed_comments'].str.replace(
        '[^\w\s]', ' ', flags=re.UNICODE, regex=True)

    # to lowercase
    dataset['preprocessed_comments'] = dataset['preprocessed_comments'].str.lower()

    # Define a regular expression to match font styles
    font_style_regex = re.compile(
        r'\b(?:bold|italic|oblique)\b', flags=re.IGNORECASE)

    # Remove font styles from all string columns
    dataset['preprocessed_comments'] = dataset['preprocessed_comments'].apply(
        lambda x: font_style_regex.sub('', x) if isinstance(x, str) else x)


    # read multi word tokenizer

    with open("Bicol Corpus/multi_word_tokenizer.txt", "r") as f:
        mwe_list = json.load(f)

    # Create an instance of the MWETokenizer
    tokenizer = MWETokenizer(mwe_list)
    dataset['preprocessed_comments'] = dataset['preprocessed_comments'].astype(
        str)
    # Define a function that applies the tokenizer to a single row

    def tokenize_row(row):
        return tokenizer.tokenize(row.split())

    # Apply the function to the text column in the dataframe
    dataset['tokenized_preprocessed_comment'] = dataset['preprocessed_comments'].apply(
        tokenize_row)

    ###Spelling Corrector###

    # get the number of CPU cores on your system

    
    # dataset['tokenized_preprocessed_comment'] = dd.from_pandas((dataset['tokenized_preprocessed_comment']), npartitions=6)


    def words(text):
        return re.findall(r'\w+', text.lower())

    WORDS = Counter(words(open('Bicol Corpus/spelling.txt').read()))

    def P(word, N=sum(WORDS.values())):
        return WORDS[word] / N

    def correction(word):
        return max(candidates(word), key=P)

    def candidates(word):
        return (known([word]) or known(edits1(word)) or [word])

    def known(words):
        return set(w for w in words if w in WORDS)

    def edits1(word):
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:]
            for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    dataset['tokenized_preprocessed_comment'] = dataset['tokenized_preprocessed_comment'].apply(lambda x: [correction(y) for y in x])
   
    # Stemming

    # Create an instance of the PorterStemmer
    stemmer = PorterStemmer()

    # Create a custom dictionary to store the words and their stemmed form
    custom_dictionary = {}

    # Add words to the custom dictionary
    # reading the data from the file
    with open('Bicol Corpus/stemmer.txt') as f:
        data = f.read()

    # reconstructing the data as a dictionary
    new_words = json.loads(data)

    # new_words = {'pwd': 'pwede', 'dgd': 'digdi', 'added': 'add'}
    custom_dictionary.update(new_words)

    # Define a function to stem words using the custom dictionary
    def custom_stem(word):
        if word in custom_dictionary:
            return custom_dictionary[word]
        else:
            return word

    dataset['tokenized_preprocessed_comment'] = dataset['tokenized_preprocessed_comment'].apply(lambda x: [custom_stem(y) for y in x]) # Stem every word.
   

    # Export preprocessed dataset
    dataset['raw_comments'] = dataset['tokenized_preprocessed_comment'].apply(lambda x: ' '.join([y for y in x]))
    dataset =  dataset[['raw_comments', 'sentiment']]
    
    return dataset



def remove_stopword(df):
    
    df = pd.DataFrame(df)

    dataset = df
    # Reset index
    dataset = dataset.reset_index(drop=True)

    # remove the extra spaces
    dataset['preprocessed_comments'] = dataset['raw_comments'].str.strip()

    # remove url
    # define a regular expression pattern to match URLs
    pattern = r'(https?://\S+|www\.\S+)'

    # function to remove URLs from text
    def remove_urls(text):
        return re.sub(pattern, '', text)

    # apply the function to the 'text' column of the dataframe
    dataset['preprocessed_comments'] = dataset['preprocessed_comments'].astype(
        str)
    dataset['preprocessed_comments'] = dataset['preprocessed_comments'].apply(
        remove_urls)

    # remove punctuation
    dataset['preprocessed_comments'] = dataset['preprocessed_comments'].str.replace(
        '[^\w\s]', ' ', flags=re.UNICODE, regex=True)

    # to lowercase
    dataset['preprocessed_comments'] = dataset['preprocessed_comments'].str.lower()

    # Define a regular expression to match font styles
    font_style_regex = re.compile(
        r'\b(?:bold|italic|oblique)\b', flags=re.IGNORECASE)

    # Remove font styles from all string columns
    dataset['preprocessed_comments'] = dataset['preprocessed_comments'].apply(
        lambda x: font_style_regex.sub('', x) if isinstance(x, str) else x)

 
    # Create an instance of the MWETokenizer
    tokenizer = MWETokenizer()
    dataset['preprocessed_comments'] = dataset['preprocessed_comments'].astype(
        str)
    # Define a function that applies the tokenizer to a single row

    def tokenize_row(row):
        return tokenizer.tokenize(row.split())

    # Apply the function to the text column in the dataframe
    dataset['tokenized_preprocessed_comment'] = dataset['preprocessed_comments'].apply(
        tokenize_row)
        # Add new stop words

    # opening the file in read mode
    my_file = open("Bicol Corpus/bicol_stopwords.txt", "r")

    # reading the file
    words = my_file.read()
    new_stopwords = words.split("\n")
    my_file.close()

    # add the bicol stop words
    stpwrd = nltk.corpus.stopwords.words('english')
    stpwrd.extend(new_stopwords)

    # remove stopwords
    dataset['preprocessed_comments'] = dataset['tokenized_preprocessed_comment'].apply(lambda x: [y for y in x if y not in (stpwrd)]) # Stem every word.
    
    
    dataset['raw_comments']= dataset['preprocessed_comments'].apply(lambda x: ' '.join([y for y in x]))
    dataset =  dataset[['raw_comments', 'sentiment']]
    return dataset
