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
SEED = 1

file_path = './beta_data.csv'
docs = []
crimes = []

data = pd.read_csv(file_path)

# 종속 변수 y 생성
y = data[['murder', 'fraud', 'assault', 'sexaul_misconduct']]

#print(data)
#print(data.columns)

for text in data['text'] :
  new_text = str(text)
  docs.append(new_text)

# for crime in data['murder', 'fraud', 'assault','sexaul_misconduct'] :
#   new_crime = int(crime)
#   crimes.append(int(crime))

tokenizer.fit_on_texts(docs)

print(y)

# print(docs)
# print(tokenizer.word_index)
# print(tokenizer.word_counts)
# docs = tokenizer.texts_to_sequences(docs) # 입력된 문장을 각 단어의 인덱스로 이루어진 순서형 데이터로 변환. 

# max_len = max(len(item) for item in docs)
# voca_size = len(tokenizer.word_index) + 1

# print(max_len)
# print(voca_size)

# X_data = pad_sequences(docs, maxlen=max_len)  #패딩

