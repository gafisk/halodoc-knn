import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_confusion_matrix

nltk.download('punkt')
nltk.download('stopwords')
from sklearn.model_selection import train_test_split

def read_data():
    dataset = pd.read_csv('main_data.csv')
    data_lower = pd.read_csv('data_lower.csv')
    data_cleaning = pd.read_csv('data_cleaning.csv')
    data_normalization = pd.read_csv('data_normalization.csv')
    data_stopword = pd.read_csv('data_stopword.csv')
    data_stemming = pd.read_csv('data_stemming.csv')
    return dataset, data_lower, data_cleaning, data_normalization, data_stopword, data_stemming

def label():
    dataset = read_data()[0]
    label = dataset['Label'].unique()
    return label

def tf_idf(data):
    label_list = ['Operability', 'Simplicity', 'Execution Efficiency', 'Error Tolerance', 'Completeness']
    filtered_data = []    
    for label in label_list:
        label_data = data[data['Label'] == label]
        label_data_sample = label_data.sample(n=min(1509, len(label_data)), random_state=42)
        filtered_data.append(label_data_sample)
    selected_data = pd.concat(filtered_data)
    data = selected_data
    data['Ulasan'] = data['Ulasan'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    data['Label'] = data['Label'].apply(lambda x: x[0] if isinstance(x, list) else x)
    data = data
    vect = TfidfVectorizer()
    TF_IDF_vector = vect.fit_transform(data['Ulasan'])
    df_TF_IDF_vector = pd.DataFrame(TF_IDF_vector.toarray(), columns=vect.get_feature_names_out())
    return TF_IDF_vector, df_TF_IDF_vector, data

def split_data(data, n):
    X_train, X_test, y_train, y_test = train_test_split(data[1], data[2]['Label'], test_size=float(f"0.{n}"), random_state=1221)
    return X_train, X_test, y_train, y_test

def knn(hasil, k):
    X_train, X_test, y_train, y_test = hasil
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    model_knn = KNeighborsClassifier(n_neighbors=k)
    knn_train = model_knn.fit(X_train, y_train_encoded)
    y_pred_encoded = knn_train.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    accuracy = accuracy_score(y_test, y_pred)
    class_names = label_encoder.classes_
    c_matrix = confusion_matrix(y_test, y_pred)
    return accuracy*100, c_matrix, class_names

def grafik(hasil):
    list_akurasi = []
    for k in range(3, 10, 2):
        model = knn(hasil, k)
        list_akurasi.append(model[0])
    return list_akurasi

    

