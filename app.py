import streamlit as st
import streamlit_antd_components as sac
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from assets import *

st.set_page_config(
    page_title="Halodoc",
    page_icon="computer",
    layout="centered",
    initial_sidebar_state="expanded",
)

if 'n' not in st.session_state:
    st.session_state.n = None

with st.sidebar:
    selected = sac.menu([
        sac.MenuItem('Home', icon="house"),
        sac.MenuItem('Dataset', icon="book"),
        sac.MenuItem('Preprocessing', icon='yin-yang',
                     children=[
                         sac.MenuItem('Case Folding', icon='alphabet'),
                         sac.MenuItem('Cleaning', icon='fan'),
                         sac.MenuItem('Normalization', icon='amd'),
                         sac.MenuItem('Stopwords', icon='sign-stop'),
                         sac.MenuItem('Stemming', icon='hammer'),
                    ]),
        sac.MenuItem('TF-IDF', icon="bezier2"),
        sac.MenuItem('Split Data', icon="hourglass-split"),
        sac.MenuItem('KNN', icon="activity"),
        sac.MenuItem('Grafik', icon="bar-chart-line"),
    ], open_all=False)

if selected == "Home":
    st.title("Halaman Home")
    st.write("Klasifikasi Komentar Applikasi Halodoc")
if selected == "Dataset":
    st.title("Halaman Dataset")
    label = label()
    st.subheader("Label Data")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.subheader(label[0])
    with col2:
        st.subheader(label[1])
    with col3:
        st.subheader(label[2])
    with col4:
        st.subheader(label[3])
    with col5:
        st.subheader(label[4])
    data = read_data()[0]
    st.dataframe(data)
    # uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
    # if uploaded_file is not None:
    #     df = pd.read_csv(uploaded_file)
    #     st.subheader("Data yang dibaca dari file CSV:")
    #     st.write(df)
            
if selected == "Case Folding":
    st.title("Proses Case Folding")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Awal")
        dataset = read_data()[0]
        st.dataframe(dataset)
    with col2:
        st.subheader("Hasil Case Folding")
        data_lower = read_data()[1]
        st.dataframe(data_lower)

if selected == "Cleaning":
    st.title("Proses Cleaning")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Awal")
        dataset = read_data()[0]
        st.dataframe(dataset)
    with col2:
        st.subheader("Hasil Cleaning")
        data_lower = read_data()[2]
        st.dataframe(data_lower)
if selected == "Normalization":
    st.title("Proses Normalization")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Awal")
        dataset = read_data()[0]
        st.dataframe(dataset)
    with col2:
        st.subheader("Hasil Normalisasi")
        data_lower = read_data()[3]
        st.dataframe(data_lower)
if selected == "Stopwords":
    st.title("Proses Stopwords")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Awal")
        dataset = read_data()[0]
        st.dataframe(dataset)
    with col2:
        st.subheader("Hasil Stopwords")
        data_lower = read_data()[4]
        st.dataframe(data_lower)
if selected == "Stemming":
    st.title("Proses Stemming")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Awal")
        dataset = read_data()[0]
        st.dataframe(dataset)
    with col2:
        st.subheader("Hasil Stemming")
        data_lower = read_data()[5]
        st.dataframe(data_lower)
if selected == "TF-IDF":
    st.title("Halaman TF-IDF")
    st.subheader("Hasil TF-IDF")
    data = read_data()[5]
    tf_idf = tf_idf(data)
    st.dataframe(tf_idf[1])

if selected == "Split Data":
    st.title("Split Data")
    st.subheader("Pilih Ukuran Data Testing")
    n = st.slider('', min_value=2, max_value=3, value=2, step=1)
    data = read_data()[5]
    tf_idf = tf_idf(data)
    hasil = split_data(tf_idf, n)
    st.subheader(f"Total Data = {(len(hasil[0])+len(hasil[1]))}")
    # st.subheader("Data Train")
    col1, col2 = st.columns(2)
    st.session_state.n = n
    with col1:
        st.subheader("X Train")
        col11, col12 = st.columns(2)
        with col11:
            st.subheader("Jumlah = ")
        with col12:
            st.subheader(len(hasil[0]))
        st.dataframe(hasil[0])
    with col2:
        st.subheader("X Test")
        col21, col22 = st.columns(2)
        with col21:
            st.subheader("Jumlah = ")
        with col22:
            st.subheader(len(hasil[1]))
        st.dataframe(hasil[1])
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("y Train")
        col31, col32 = st.columns(2)
        with col31:
            st.subheader("Jumlah = ")
        with col32:
            st.subheader(len(hasil[2]))
        st.dataframe(hasil[2])
    with col4:
        st.subheader("y Test")
        col41, col42 = st.columns(2)
        with col41:
            st.subheader("Jumlah = ")
        with col42:
            st.subheader(len(hasil[3]))
        st.dataframe(hasil[3])

if selected == "KNN":
    if st.session_state.n is None:
        st.title("Halaman Klasifikasi")
        data = read_data()[5]
        tf_idf = tf_idf(data)
        hasil = split_data(tf_idf, 2)
        k = st.slider('', min_value=1, max_value=5, value=2, step=1)
        knn = knn(hasil, k)
        plt.figure(figsize=(8, 8))
        sns.heatmap(knn[1], annot=True, fmt="d", cmap="Blues", xticklabels=knn[2], yticklabels=knn[2])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        st.subheader("Akurasi Yang Diperoleh")
        st.subheader(f"{knn[0]}%")
        st.pyplot(plt.gcf())
    else:
        st.title("Halaman KNN")
        data = read_data()[5]
        tf_idf = tf_idf(data)
        hasil = split_data(tf_idf, st.session_state.n)
        st.subheader("Pilih Nilai K")
        k = st.slider('', min_value=3, max_value=9, value=3, step=2)
        knn = knn(hasil, k)
        plt.figure(figsize=(8, 8))
        sns.heatmap(knn[1], annot=True, fmt="d", cmap="Blues", xticklabels=knn[2], yticklabels=knn[2])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        st.subheader("Akurasi Yang Diperoleh")
        st.subheader(f"{knn[0]}%")
        st.pyplot(plt.gcf())

if selected == "Grafik":
    if st.session_state.n is None:
        st.title("Halaman Grafik")
        data = read_data()[5]
        tf_idf = tf_idf(data)
        hasil = split_data(tf_idf, 2)
        graph = grafik(hasil)
        labels = ['K 3', 'K 5', 'K 7', 'K 9']
        bars = plt.bar(labels, graph, color='blue')
        plt.title(f'Akurasi Split Data Testing 20%')
        plt.xlabel('Data Points')
        plt.ylabel('Accuracy (%)')
        for bar, value in zip(bars, graph):
            plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 1, f'{round(value,2)}%', ha='center', va='bottom', color='black', fontweight='bold')
        st.pyplot(plt.gcf())
    else:
        st.title("Halaman Grafik")
        data = read_data()[5]
        tf_idf = tf_idf(data)
        hasil = split_data(tf_idf, st.session_state.n)
        graph = grafik(hasil)
        labels = ['K 3', 'K 5', 'K 7', 'K 9']
        bars = plt.bar(labels, graph, color='blue')
        plt.title(f'Akurasi Split Data Testing {st.session_state.n}0%')
        plt.xlabel('Data Points')
        plt.ylabel('Accuracy (%)')
        for bar, value in zip(bars, graph):
            plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 1, f'{round(value,2)}%', ha='center', va='bottom', color='black', fontweight='bold')
        st.pyplot(plt.gcf())