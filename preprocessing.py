import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from konlpy.tag import Okt
import re

okt = Okt()
tokenizer = Tokenizer()

file_path = './beta_data.csv'
docs = []  
crimes = []

data = pd.read_csv(file_path)

for text in data['text']: #text열에 있는 데이터들을 하나 씩 가져옴
    new_text = str(text).replace("\n",'')
    pattern = r'[^a-zA-Z가-힣]'
    new_text = re.sub(pattern=pattern, repl=' ', string=text)

    docs.append(okt.nouns(new_text))

tokenizer.fit_on_texts(docs) 

print(tokenizer.word_index)

docs = tokenizer.texts_to_sequences(docs) 

max_len = max(len(item) for item in docs)
voca_size = len(tokenizer.word_index) + 1

print(max_len)
print(voca_size)

