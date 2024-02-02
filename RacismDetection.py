#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.layers import Embedding, GRU, Bidirectional
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1_l2
import snowballstemmer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from tensorflow.keras.metrics import Precision, Recall
from sklearn.model_selection import GridSearchCV



# Gerekli kütüphaneleri yükleme
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense
import networkx as nx
import torch
from torch_geometric.nn import GCNConv
from torch.nn import LSTM, Linear, Module
from IPython.display import HTML


# Veri kümesini yükleme
file_path = '/Users/sulekaya/Downloads/2023/PeerJ/Dosya/Dataset.xlsx'
data = pd.read_excel(file_path)

# Stopwords yükleme
with open('/Users/sulekaya/Downloads/2023/TEZ/kod/turkce-stop-words.txt', 'r', encoding='utf-8') as file:
    turkish_stopwords = file.read().splitlines()

import re
import pandas as pd

def clean_text(text):
    # NaN değerler için kontrol
    if pd.isna(text):
        return ""

    # URL'leri, kullanıcı adlarını ve özel karakterleri temizleme
    text = re.sub(r'http\S+', '', text)  # URL'leri temizle
    text = re.sub(r'@\w+', '', text)     # Kullanıcı adlarını temizle
    text = re.sub(r'\W', ' ', text)      # Özel karakterleri temizle

    # Emojileri temizleme
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # Metni küçük harflere çevirme
    text = text.lower()

    # Durak kelimeleri kaldırma
    text = ' '.join([word for word in text.split() if word not in turkish_stopwords])

    # Fazla boşlukları kaldırma
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Metinleri temizleme ve durak kelimeleri kaldırma
data['processed_text'] = data['content'].apply(clean_text)


# In[21]:


# Tokenizer'ı veri kümeniz üzerinde uygulayın
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['processed_text'])

# Kelime dağarcığı boyutunu hesaplayın
vocab_size = len(tokenizer.word_index) + 1  # Kelime dağarcığı boyutu

# Maksimum sıra uzunluğunu belirleme
# Tüm dizilerin uzunluklarını hesaplayın ve en uzununu bulun
sequence_lengths = [len(seq) for seq in tokenizer.texts_to_sequences(data['processed_text'])]
max_sequence_length = max(sequence_lengths)  # En uzun dizi uzunluğu

# Gömme boyutu ve LSTM üniteleri sabit olarak belirlenebilir
embedding_dim = 100  # Gömme boyutu
lstm_units = 64  # LSTM ünitelerinin sayısı

# Parametreleri Yazdır
print("Vocabulary Size:", vocab_size)
print("Max Sequence Length:", max_sequence_length)
print("Embedding Dimension:", embedding_dim)
print("LSTM Units:", lstm_units)


# In[25]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.layers import GlobalAveragePooling1D, Concatenate
from tensorflow.keras import Input, Model

# Model Parametreleri
max_sequence_length = 45  # Dizin uzunluğu
vocab_size = len(tokenizer.word_index) + 1  # Kelime dağarcığı boyutu
embedding_dim = 100  # Gömme boyutu
lstm_units = 64  # LSTM ünitelerinin sayısı

# Model Mimarisi
input_layer = Input(shape=(max_sequence_length,))
embedding_layer = Embedding(vocab_size, embedding_dim, input_length=max_sequence_length)(input_layer)

# İlk LSTM Katmanı
lstm_layer1 = Bidirectional(LSTM(lstm_units, return_sequences=True))(embedding_layer)

# İkinci LSTM Katmanı
lstm_layer2 = Bidirectional(LSTM(lstm_units, return_sequences=True))(lstm_layer1)

# Global Ortalama Havuzlama
global_avg_pool = GlobalAveragePooling1D()(lstm_layer2)

# Tam Bağlantılı Katman
dense_layer = Dense(32, activation='relu')(global_avg_pool)
dropout_layer = Dropout(0.5)(dense_layer)
output_layer = Dense(1, activation='sigmoid')(dropout_layer)

# Modeli Oluşturma
model = Model(inputs=input_layer, outputs=output_layer)

# Modeli Derleme
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model Özeti
model.summary()


# Eğitim ve test verilerini max_sequence_length'a göre kırpma veya doldurma
X_train_padded = pad_sequences(X_train, maxlen=max_sequence_length, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test, maxlen=max_sequence_length, padding='post', truncating='post')

# Model Eğitimi
history = model.fit(X_train_padded, y_train, epochs=10, batch_size=64, validation_data=(X_test_padded, y_test))

# Model Değerlendirmesi
loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=1)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")


# In[ ]:




