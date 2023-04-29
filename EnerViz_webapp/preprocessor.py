# import the required libraries
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
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import MWETokenizer
import warnings
import logging
logging.getLogger('distributed').setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="Port 8787 is already in use.")
from datetime import datetime

start_time = time.time()


if __name__ == '__main__':
   
        multiprocessing.freeze_support()
        # Load ds file
        print(' Reading Dataset...')
        df = pd.read_csv('Datasets/raw_data.csv')
        df = pd.DataFrame(df)
 
        print('Done!')

        dataset = df
        
        print('Cleaning Dataset...')
        # dropping ALL duplicate values
        dataset.drop_duplicates(
            subset=['raw_comments', 'created_time'], keep=False, inplace=True)
      

        # Reset index
        dataset = dataset.reset_index(drop=True)
      

        # remove the extra spaces
        dataset['preprocessed_comments'] = dataset['raw_comments'].str.strip()
      

        # remove punctuation
        dataset['preprocessed_comments'] = dataset['preprocessed_comments'].str.replace(
            '[^\w\s]', '', flags=re.UNICODE, regex=True)
      

        # to lowercase
        dataset['preprocessed_comments'] = dataset['preprocessed_comments'].str.lower()
      
        print('Done!')
        

        # #Add new stop words
        # import nltk
        # nltk.download('stopwords')
        # from nltk.corpus import stopwords

        # # opening the file in read mode
        # my_file = open("bicol_stopwords.txt", "r")

        # # reading the file
        # words = my_file.read()
        # new_stopwords = words.split("\n")
        # my_file.close()

        # # add the bicol stop words
        # stpwrd = nltk.corpus.stopwords.words('english')
        # stpwrd.extend(new_stopwords)

        # #remove stopwords
        # dataset['preprocessed_comments'] = dataset['preprocessed_comments'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stpwrd)]))

        print('Tokenizing Dataset...')
        # tokenize string


        # read multi word tokenizer
        
        with open("Bicol Corpus/multi_word_tokenizer.txt", "r") as f:
            mwe_list = json.load(f)

        # Create an instance of the MWETokenizer
        tokenizer = MWETokenizer(mwe_list)
        dataset['preprocessed_comments'] = dataset['preprocessed_comments'].astype(str)
        # Define a function that applies the tokenizer to a single row

        def tokenize_row(row):
            return tokenizer.tokenize(row.split())

        # Apply the function to the text column in the dataframe
        dataset['tokenized_preprocessed_comment'] = dataset['preprocessed_comments'].apply(
            tokenize_row)
    
        print('Done!')

        print('Spelling Correction...')
        ###Spelling Corrector###
        

        # get the number of CPU cores on your system
        num_cores = multiprocessing.cpu_count()
        ds = pd.DataFrame()

        ds['comment'] = dataset['tokenized_preprocessed_comment']
        #dataset['tokenized_preprocessed_comment'] = dd.from_pandas((dataset['tokenized_preprocessed_comment']), npartitions=6)
        ds = dd.from_pandas(ds, npartitions=num_cores)
        # # Add custom words to the dictionary

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

        #dataset['tokenized_preprocessed_comment'] = dataset['tokenized_preprocessed_comment'].apply(lambda x: [correction(y) for y in x])
        ds['comment'] = ds['comment'].apply(
            lambda x: [correction(y) for y in x], meta=('comment', 'object'))

       
        print('Done!')

        print('Stemming Dataset...')
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

        #new_words = {'pwd': 'pwede', 'dgd': 'digdi', 'added': 'add'}
        custom_dictionary.update(new_words)

        # Define a function to stem words using the custom dictionary
        def custom_stem(word):
            if word in custom_dictionary:
                return custom_dictionary[word]
            else:
                return word

        # dataset['tokenized_preprocessed_comment'] = dataset['tokenized_preprocessed_comment'].apply(lambda x: [custom_stem(y) for y in x]) # Stem every word.
        ds['comment'] = ds['comment'].apply(
            lambda x: [custom_stem(y) for y in x], meta=('comment', 'object'))
       
        print('Done!')

        print('Removing Stopwords...')
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
        # dataset['tokenized_preprocessed_comment'] = dataset['tokenized_preprocessed_comment'].apply(lambda x: [y for y in x if y not in (stpwrd)]) # Stem every word.
        ds['comment'] = ds['comment'].apply(lambda x: [y for y in x if y not in (
            stpwrd)], meta=('comment', 'object'))  # Stem every word.
        #dataset['preprocessed_comments'] = dataset['preprocessed_comments'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stpwrd)]))
       
        print('Done!')

        #dataset['tokenized_preprocessed_comment'] = dataset['tokenized_preprocessed_comment'].astype(str)
        #dataset['tokenized_preprocessed_comment'] = ds['comment']

        # Convert each partition of the Dask DataFrame to a Pandas DataFrame

        from dask.distributed import Client
        client = Client(n_workers=4)
        print('Converting Dataset...')
        print('This may take a few moments...')
        dataset['tokenized_preprocessed_comment'] = ds.compute()
      
        print('Done!')
          # print execution time
          
        print("   Execution Time   ")
        print("--- %.2s seconds ---" % (time.time() - start_time))
        
        # Define the conversion function
        def convert_date(date_str):
            try:
                # Parse the date string using the datetime module
                date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S.%f")
                
                # Convert the date object back to a string in the desired format
                formatted_date = date_obj.strftime("%Y-%m-%d")
            except ValueError:
                # If the date string is already in the desired format, return it unchanged
                formatted_date = date_str
            
            return formatted_date

        # Apply the conversion function to the 'date' column
        df['created_time'] = df['created_time'].apply(convert_date)
        #Export preprocessed dataset
        dataset['preprocessed_comments'] = dataset['tokenized_preprocessed_comment'].apply(lambda x: ' '.join([y for y in x]))
        dataset =  dataset[['created_time', 'raw_comments', 'preprocessed_comments']]
        dataset.to_csv('Datasets/preprocessed_data.csv', index = False)
        print('***Preprocessing Complete***')
    