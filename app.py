import string
import requests
import streamlit as st
import re
import csv
import base64
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')
import time
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, classification_report
from streamlit_option_menu import option_menu
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# --- SET PAGE CONFIG ---
st.set_page_config(
    page_title="ANALISIS SENTIMEN RANGKA E-SAF",
    page_icon="twitter.jpg",
    layout="wide"
)

# --- SIDEBAR MENU ---
with st.sidebar:
    # Menampilkan gambar dengan ukuran lebih kecil
    st.image(Image.open('twitter.png'), width=150)  # Sesuaikan ukuran lebar sesuai kebutuhan
    
    # Navigasi menu setelah gambar
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Dataset", "Preprocessing", "Modeling", "Visualization", "Prediction"],
        icons=["house", "book", "search", "gear", "image-alt", "chat-dots"],
        menu_icon="cast",
        default_index=0
    )

    # Social media links
    st.markdown("<h1 style='text-align: center; font-size:20px;'>Let's Connect With Me</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown("[![LinkedIn](https://skillicons.dev/icons?i=linkedin)](https://www.linkedin.com/in/kinanthi-putri/)")
    with col2:
        st.markdown("[![Instagram](https://skillicons.dev/icons?i=instagram)](https://www.instagram.com/kinanthipa/)")
    with col3:
        st.markdown("[![Github](https://skillicons.dev/icons?i=github)](https://github.com/KinanthiPutriAriyani)")

    st.caption("<h1 style='text-align: center; font-size:15px;'>© 2025 by Kinanthi Putri Ariyani</h1>", unsafe_allow_html=True)

# Hide default Streamlit menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.css-1aumxhk {display: none;}  /* Sidebar */
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Untuk Melakukan Prediksi
def load_positive_words():
    with open("positive.txt", "r") as file:
        positive_words = [line.strip() for line in file]
    return positive_words

def load_negative_words():
    with open("negative.txt", "r") as file:
        negative_words = [line.strip() for line in file]
    return negative_words

def clean_text(text, stopwords=None):
    # Remove emojis from the text using regular expression
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    text = re.sub(f"[{string.punctuation}]", " ", text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stopwords]
    cleaned_text = ' '.join(words)
    return cleaned_text

def load_stopwords():
    # Memuat stopwords bahasa Indonesia dari NLTK
    stopwords_l = stopwords.words('indonesian')

    # Menambahkan custom stopwords
    custom_stopwords = '''
        nik ais ih kuea ndes tk arghhhh wuakakak gtth wowww apeeee Aksjsjsk alaee koq wuakakak salengpraew 
        rukkhadevata gtth zeon vivienne yaam woyy ykwim auff ue hoek hayo chnmn hahahah haaaaaa din woy ndeer 
        lalalala wkwkwkwkwkkw woyyy dih den hehehew etdah beeeuh wahh heheee hhaaha waaaaa oakilah haaaahh huft ai 
        et acha ue hokyahokya hahahihi yl wihh hahahaa hhhhh def ayom ser duh heuheueheu huwaaaaaa yalah mww cekabia 
        dikatara angganara krtsk woee ndi ohh www aee huaaaa gn hahahah nd ema ceratops pasuk ygy repp gais 
        hadehhhhhhh walah hahah paa awkwkwk wkwkk wkwkw wkwkwkwkwkwah wkwkwkw baceprot sksksk heheh brooo dbd aeee 
        weeeh wehh milta hsnah swsg hemm xda yara ohh heh kle acy hayooo hahahahaha balablablabla lai loj itine 
        heehehe kwkwk kwkwkwkwwkwk waaa demending pali eeh dlsb cooooy hehehehe adjem aih syar wkwkk aowkwkwk walah 
        euy der hahaa hesteg hmmmmtar gtideologi ab owkwkwkwk dncw sloga jo jengjenggg anuanu caw ehehheheh hlaa 
        hahahihi ckckckck sich pakin mmarkpkk ponponpon kyary pamyu laaahhh cp duhhh napen lise bi ieu poho boga 
        imah keur ulin kwkwkw ehheh gryli oalah prekk hehh cere ekekekek chco nganu wkwkkwkwkwkwkw pfft 
        awowkwkwkwk kinyis pus yng yg wkwoswkwo wkwkwkwkwkwk ahahha weeeeh hah fir hong jay haikyuu nderrr 
        omtanteuwaksodara ahsajkakaka kwkwkwk derrr wwkwkwkw hadehh aaaaa heeh dem ocaaa wo prenup dihhh 
        cokk imho chenle jsdieksisnisawikwok hahahahahahaha bam yowohh lau boiiiii gih beuhhh wkw wkwkwkw dooong 
        oalaaaa sinoeng wkekwk nyai cai anw tjuyyy hanss mh widihh cy eeeee gi luat laaaaa cam lancau tuch kun 
        uhhhh chuakssss oiyaa hadeuhhhh wkwkwkwwk hehehee nk lak qwq oneesan eeehmmm am wkwk nih hahh bulan2an 
        rek noh set or BYD oyi kah yaaa hehe lah bhahaha tuh wkwkwkwkkk wekekekekek duhhhh hahaha kah boss sih 
        rob pooooll si ahh wkwk dew laah yaa lhoo min mah banh pil wkwkwk ta deh wkwwk xx hahahahaha hon ah yah 
        esa xixixi yekann hemzz si ya atuhh kwkwkwk mah doong id amp wkw kirian an who ehhhh ny ee lho lah loh an 
        tuh cuy hrakiri mah nya tuh ke ah an kah no haha yeeee wt dunk lembiru lho iyain sak neh ny id shi kok to 
        pak dafi all kn hmmm hm ri tak siiihhh gress hmmmm lhoo wkwkwk hehe wassalam owh mberr ngab wgwg wkkwwk 
        salampuput omm wkwwk tha lahh hhhaaa oh ealah hayoo ahaha waduhh macam wah tag acm wkwkw ayooo part lohhhh 
        solehudin ser tok kes lachhhh keba lha hadeeeh blass sing kok icikiwir pro gl gan waaaaah the ahh anti 
        yeayyyy uwooohh behhh euy wuih awww ahhhh wihh woy parahh uwooohh nya waaaaah btw awww pula waduhh widihhh 
        uhuiiii no kwkw nyahoo uu ckckc nih id istil hahahahahaha dipast wasallam khan wkwkw loh deh ukhti 
        skywave in yahh bahahaha mb ii kaaaaan abieeesz lurrr awkwowkwowkwo wahhh sd we do oohh wkwkwkwkwk plant 
        tag si lhaaaaa jiah sallaamm hayo thank wkwkwkkwkwkwk wkwkwkw yhaaaa wkwkwk yahahahaha cem woaaah uhui 
        wkwkwkw bhahaha
    '''

    # Menggabungkan stopwords dari NLTK dan custom stopwords
    return set(stopwords_l + custom_stopwords.split())

# Menu Home
if selected == "Home":
   # Define custom CSS style to center the text
    st.markdown(
        f"""
        <style>
            .centered-text {{
                display: flex;
                justify-content: center;
                align-items: center;
                font-size: 30px;
                margin-bottom:30px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h1 style='text-align: center; color: black;'>Analisis Sentimen Rangka e-SAF</h1>", unsafe_allow_html=True)

    # Gambar yang ingin Anda tampilkan
    esaf = "rangka esaf.jpeg"

    # Membuat tata letak dengan kolom agar gambar berada di tengah
    col1, col2, col3 = st.columns([1, 2, 1])  # Kolom tengah lebih besar untuk gambar

    with col2:
        st.image(esaf, width=500)  # Menyesuaikan ukuran gambar

    # Menampilkan penjelasan di bawah gambar
    st.markdown(
        '''
        <div style="text-align: justify; text-justify: inter-word; max-width: 800px; margin: 0 auto;">
        Rangka eSAF (Enhanced Smart Architecture Frame) adalah rangka motor yang dikembangkan oleh Honda. 
        Struktur bagian depan rangka ini terhubung langsung ke tempat duduk pengendara. 
        eSAF merupakan hasil dari riset mendalam yang dilakukan oleh tim teknik Honda, dengan fokus pada peningkatan keamanan, kenyamanan, dan kinerja berkendara. 
        Namun, pada Agustus 2023 kasus patahnya rangka sepeda motor matic Honda mencuri perhatian publik. 
        Masalah ini menimbulkan kekhawatiran karena berpotensi menyebabkan kecelakaan dan membahayakan keselamatan konsumen. 
        Oleh karena itu, diperlukan analisis sentimen untuk memahami respons publik terhadap isu rangka eSAF.
        </div>
        ''',
        unsafe_allow_html=True)

# Menu Dataset
if selected == "Dataset":
    st.title("Dataset Analisis Sentimen Rangka eSAF Pada Sosial Media X")

    def load_data(file_path):
        try:
            # Menambahkan parameter encoding saat membaca file CSV
            data = pd.read_csv(file_path, encoding='ISO-8859-1')  # Coba dengan encoding lain jika diperlukan
            return data
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memuat data: {e}")
            return None

    # Tampilkan tombol untuk memuat data
    load_button = st.button("Muat Data")

    # Pengecekan apakah tombol ditekan
    if load_button:
        file_path = "rangka_esaf.csv"  # Pastikan file CSV sudah diunduh ke folder proyek
        
        # Jika file_path telah ditentukan, muat dan tampilkan data
        st.write(f"Menampilkan data dari {file_path}")
        data = load_data(file_path)
        if data is not None:
            st.dataframe(data)

            # Create the download link for CSV
            csv_data = data.to_csv(index=False, quoting=csv.QUOTE_NONNUMERIC)
            b64 = base64.b64encode(csv_data.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="rangka_esaf.csv">Download Data CSV</a>'
            st.markdown(href, unsafe_allow_html=True)


if selected == "Preprocessing":
    st.title("Preprocessing and Labeling")

    stop_words = load_stopwords()
    
    # Membaca file positive.txt dan negative.txt untuk pelabelan sentimen
    with open("positive.txt", "r", encoding="utf-8") as f:
        positive_words = set(f.read().splitlines())

    with open("negative.txt", "r", encoding="utf-8") as f:
        negative_words = set(f.read().splitlines())

    # Fungsi untuk menganalisis sentimen berdasarkan daftar kata positif dan negatif
    def analyze_sentiment(comment):
        words = comment.split()  # Tokenisasi sederhana berdasarkan spasi
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)

        if pos_count >= neg_count:
            return 'Positive'
        else:
            return 'Negative'

    def clean_comment(comment):
        if isinstance(comment, float):
            comment = str(comment)  # Konversi float ke string jika diperlukan
    
        # Remove tab, newline, backslash
        comment = comment.replace('\\t', " ").replace('\\n', " ").replace('\\u', " ").replace('\\', "")
        # Remove non-ASCII (emoticons, Chinese characters, etc.)
        comment = comment.encode('ascii', 'replace').decode('ascii')
        # Remove mentions, links, and hashtags
        comment = ' '.join(re.sub(r"([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", comment).split())
        # Remove numbers
        comment = re.sub(r"\d+", "", comment)
        # Remove punctuation
        comment = comment.translate(str.maketrans("", "", string.punctuation))
        # Remove leading and trailing whitespace
        comment = comment.strip()
        # Remove multiple whitespaces into a single whitespace
        comment = re.sub(r'\s+', ' ', comment)
        # Remove single characters (isolated)
        comment = re.sub(r"\b[a-zA-Z]\b", "", comment)
        return comment

    def case_folding(comment):
        # Mengubah teks menjadi lowercase
        return comment.lower()

    def tokenizing(comment):
        # Melakukan tokenisasi pada teks
        return word_tokenize(comment)

    # Membaca file normalisasi dan membuat dictionary
    normalized_word = pd.read_excel("normalisasi.xlsx")

    normalized_word_dict = {}
    for index, row in normalized_word.iterrows():
        if row[0] not in normalized_word_dict:
            normalized_word_dict[row[0]] = row[1]

    # Fungsi normalisasi menggunakan dictionary
    def normalization(tokens):
        return [normalized_word_dict[token] if token in normalized_word_dict else token for token in tokens]

    def removal_stopwords(tokens):
        # Menghapus stopwords dari teks
        stop_words = load_stopwords()  # Call the load_stopwords function to get stopwords
        tokens_without_stopwords = [
            token for token in tokens if token not in stop_words]
        return tokens_without_stopwords

    # Membuat objek Stemmer dari Sastrawi
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def stemming(tokens):
        # Melakukan stemming pada kata-kata
        stemmed_tokens = []
        for token in tokens:
            stemmed_token = stemmer.stem(token)
            stemmed_tokens.append(stemmed_token)
        return stemmed_tokens

    def fit_stopwords(tokens):
        # Menggabungkan elemen array menjadi satu string
        return ' '.join(tokens)

    uploaded_file = st.file_uploader(
        "Upload Dataset", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Info File:")
        st.write("- Nama File: ", uploaded_file.name)
        st.write("Data Awal:")
        st.write(df)

        def preprocessing():
            st.write("Start Preprocessing")
            st.write("| data cleaning...")
            time.sleep(1)  # Simulasi waktu pemrosesan
            df['data cleansing'] = df['full_text'].apply(clean_comment)

            st.write("| case folding...")
            time.sleep(1)  # Simulasi waktu pemrosesan
            df['case folding'] = df['data cleansing'].apply(case_folding)

            st.write("| tokenizing...")
            time.sleep(1)  # Simulasi waktu pemrosesan
            df['tokenization'] = df['case folding'].apply(tokenizing)

            st.write("| normalization...")
            time.sleep(1)  # Simulasi waktu pemrosesan
            df['normalisasi'] = df['tokenization'].apply(normalization)

            st.write("| removal stopwords...")
            time.sleep(1)  # Simulasi waktu pemrosesan
            df['stopwords'] = df['normalisasi'].apply(removal_stopwords)

            st.write("| stemming...")
            time.sleep(1)  # Simulasi waktu pemrosesan
            df['stemmed'] = df['stopwords'].apply(stemming)

            st.write("| combining words...")
            time.sleep(1)
            df['final_text'] = df['stemmed'].apply(fit_stopwords)

            st.write("| labeling data...")
            time.sleep(1)
            df['label'] = df['final_text'].apply(analyze_sentiment)

            st.write("Finish Preprocessing")

        if st.button("Mulai Preprocessing"):
            # Menyisakan hanya kolom 'full_text'
            if 'full_text' in df.columns:
                df = df[['full_text']]
                
                # Menghapus baris dengan nilai NaN di kolom full_text
                df = df.dropna(subset=['full_text'])
                
                # Mengecek apakah setelah penghapusan masih ada data
                if df.empty:
                    st.error("Semua data di kolom 'full_text' kosong atau NaN. Tidak dapat melanjutkan preprocessing.")
                    st.stop()  # Menghentikan eksekusi jika tidak ada data tersisa
            else:
                st.error("Kolom 'full_text' tidak ditemukan dalam dataset!")
                st.stop()  # Menghentikan eksekusi jika kolom tidak ditemukan
            
            preprocessing()

            # Mengurutkan label ke paling akhir
            cols = df.columns.tolist()
            cols.remove('label')
            cols.append('label')
            df = df[cols]

            # Menghapus baris jika 'final_text' kosong atau hanya berisi spasi
            if 'final_text' in df.columns:
                df['final_text'] = df['final_text'].astype(str).str.strip()  # Pastikan kolom berupa string dan hapus spasi berlebih
                df = df[df['final_text'] != ""]  # Hapus baris yang kosong
                
                # Mengecek apakah setelah penghapusan masih ada data
                if df.empty:
                    st.error("Semua data di kolom 'final_text' kosong setelah preprocessing. Tidak dapat melanjutkan.")
                    st.stop()
            else:
                st.error("Kolom 'final_text' tidak ditemukan setelah preprocessing!")
                st.stop()

            st.write("Hasil Preprocessing:")
            st.write(df)

            temp_file = df.to_csv(index=False)
            b64 = base64.b64encode(temp_file.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="hasil_preprocessing.csv">Download Hasil Preprocessing</a>'
            st.markdown(href, unsafe_allow_html=True)


if selected == "Modeling":
    st.title("Modeling")
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload file CSV yang Telah Dipreprocessing", type=["csv"])
    
    # If CSV file is selected
    if uploaded_file is not None:
        # Read CSV file into DataFrame
        df = pd.read_csv(uploaded_file)
        # Display file information
        st.write("Info File:")
        st.write("- Nama File: ", uploaded_file.name)
        st.write("Data Awal:")
        st.write(df)
        
        # Create two columns for buttons
        col1, col2 = st.columns(2)
        
        # Modeling Button
        if col1.button("Mulai Pemodelan"):
            def feature_extraction(df):
                vectorizer = TfidfVectorizer(max_features=1000)
                features = vectorizer.fit_transform(df['final_text'])
                return features, vectorizer

            def select_features(X, y, k=100):
                selector = SelectKBest(score_func=mutual_info_classif, k=k)
                X_selected = selector.fit_transform(X, y)
                selected_features = selector.get_support(indices=True)
                feature_names = vectorizer.get_feature_names_out()
                top_features = [feature_names[i] for i in selected_features]
                return X_selected, top_features
            
            # Separate features and labels
            X = df['final_text']
            y = df['label']
            
            # Vectorize text
            vectorizer = CountVectorizer()
            X_vectorized = vectorizer.fit_transform(X)
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=72)
            
            # SVM models with updated parameters
            svm_models = {
                "Linear": SVC(kernel='linear', probability=True, C=10),
                "Polynomial": SVC(kernel='poly', degree=1, probability=True, C=100),
                "RBF": SVC(kernel='rbf', gamma=1, probability=True, C=20)
            }
            
            for name, model in svm_models.items():
                st.write(f"### Model Support Vector Machine dengan Kernel: {name}")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate evaluation metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                cm = confusion_matrix(y_test, y_pred)
                
                st.write(f"**Kernel: {name}**")
                st.write(f"- Accuracy: {accuracy * 100:.2f}%")
                st.write(f"- Precision: {precision * 100:.2f}%")
                st.write(f"- Recall: {recall * 100:.2f}%")
                st.write(f"- F1-score: {f1 * 100:.2f}%")
                
                # Display classification report
                classification_rep = classification_report(y_test, y_pred, output_dict=True)
                classification_df = pd.DataFrame(classification_rep).transpose()
                st.write("### Classification Report")
                st.dataframe(classification_df.style.format({
                    "precision": "{:.2f}",
                    "recall": "{:.2f}",
                    "f1-score": "{:.2f}",
                    "support": "{:.0f}"
                }))
                
                # Display Confusion Matrix
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=model.classes_, 
                           yticklabels=model.classes_)
                ax.set_title(f'Confusion Matrix - Kernel: {name}')
                plt.tight_layout()
                st.pyplot(fig)

if selected == "Visualization":
    st.title("Visualisasi Sentimen")
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload file CSV yang Telah Dipreprocessing", type=["csv"])
    
    # If CSV file is selected
    if uploaded_file is not None:
        # Read CSV file into DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Display file information
        st.write("Info File:")
        st.write("- Nama File: ", uploaded_file.name)
        st.write("Data Awal:")
        st.write(df)
        
        # Button for starting visualization
        if st.button("Mulai Visualisasi"):
            try:
                # Combine all text
                all_text = ' '.join(df['stemmed'].values)
                
                # Count frequent words
                words_list = all_text.split()
                word_count = Counter(words_list)
                common_words = word_count.most_common(10)
                words = [word for word, count in common_words]
                count = [count for word, count in common_words]
                
                # Combine positive and negative tweets
                all_positive_comment = ' '.join(df[df['label'] == 'Positive']['final_text'])
                all_negative_comment = ' '.join(df[df['label'] == 'Negative']['final_text'])
                
                # Create tabs for different visualizations
                tab1, tab2, tab3, tab4 = st.tabs(["Distribusi Sentimen", "Kata Frekuensi", "Persentase", "WordCloud"])
                
                with tab1:
                    st.subheader("Distribusi Sentimen")
                    # Bar chart for sentiment distribution
                    a = len(df[df["label"] == "Positive"])
                    b = len(df[df["label"] == "Negative"])
                    colors = sns.color_palette("husl", 2)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(["Positive", "Negative"], [a, b], color=colors)
                    ax.set_title('Jumlah Data untuk Setiap Sentimen')
                    ax.set_xlabel('Sentimen')
                    ax.set_ylabel('Jumlah Data')
                    st.pyplot(fig)
                    plt.close()
                
                with tab2:
                    st.subheader("Frekuensi Kata")
                    # Bar chart for frequent words
                    colors = np.random.rand(len(words), 3)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(words, count, color=colors)
                    ax.set_title('Kata yang Sering Muncul')
                    ax.set_xlabel('Kata')
                    ax.set_ylabel('Jumlah Kemunculan')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                with tab3:
                    st.subheader("Persentase Sentimen")
                    # Pie chart
                    labels = ['Positive', 'Negative']
                    sizes = [a, b]
                    colors = ['#66b3ff', '#ff9999']
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    ax1.pie(sizes, colors=colors, labels=labels,
                           autopct='%1.1f%%', startangle=90)
                    
                    # Draw circle
                    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
                    fig = plt.gcf()
                    fig.gca().add_artist(centre_circle)
                    ax1.axis('equal')
                    plt.title("Persentase Sentimen")
                    plt.tight_layout()
                    st.pyplot(fig1)
                    plt.close()
                
                with tab4:
                    st.subheader("WordCloud Visualisasi")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("WordCloud Semua Text")
                        # WordCloud - All text
                        wordcloud = WordCloud(width=800, height=400, max_words=150,
                                            background_color='white').generate(all_text)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.set_title('WordCloud - Kata yang Sering Muncul')
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close()
                    
                    with col2:
                        st.write("WordCloud Sentimen Positif")
                        # WordCloud - Positive sentiment
                        wordcloud_positive = WordCloud(
                            width=800, height=400, background_color='white'
                        ).generate(all_positive_comment)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.imshow(wordcloud_positive, interpolation='bilinear')
                        ax.set_title('WordCloud Sentimen Positif')
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close()
                    
                    with col3:
                        st.write("WordCloud Sentimen Negatif")
                        # WordCloud - Negative sentiment
                        if 'Negative' in df['label'].values:
                            wordcloud_negative = WordCloud(
                                width=800, height=400, background_color='white'
                            ).generate(all_negative_comment)
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.imshow(wordcloud_negative, interpolation='bilinear')
                            ax.set_title('WordCloud Sentimen Negatif')
                            ax.axis('off')
                            st.pyplot(fig)
                            plt.close()
                            
            except Exception as e:
                st.error(f"Terjadi kesalahan dalam pemrosesan data: {str(e)}")

if selected == "Prediction":
    st.title("Sentiment Prediction")
    # Kode untuk menampilkan konten halaman "Prediksi"
    uploaded_file = st.file_uploader(
        "Upload File Hasil Preprocessing", type=["csv"])

    # Jika file CSV dipilih
    if uploaded_file is not None:

        # Membaca file CSV menjadi DataFrame
        df = pd.read_csv(uploaded_file)

        # Menampilkan informasi tentang file yang diunggah
        st.write("Info File:")
        st.write("- Nama File: ", uploaded_file.name)

        # Menampilkan data awal dari DataFrame
        st.write("Data Awal:")
        st.write(df)

        # Mengatasi nilai NaN atau null pada kolom full_text
        df['full_text'].fillna('', inplace=True)

        # Muat stopwords
        stopwords = load_stopwords()

        # Fitur kata negatif
        negative_words = load_negative_words()
        positive_words = load_positive_words()

        # Feature Extraction
        vectorizer = TfidfVectorizer(max_features=1000)
        features = vectorizer.fit_transform(df['full_text'])

        # Prepare data for modeling
        X = features
        y = df['label']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=72)

        # Latih SVM classifier 
        svm_classifier = SVC(kernel='linear', probability=True, C=10)
        svm_classifier.fit(X_train, y_train)

        # Make predictions
        y_pred = svm_classifier.predict(X_test)

        # Form untuk prediksi
        st.write("Form Prediksi:")
        input_text = st.text_input("Masukkan komentar:")

        if st.button("Prediksi"):
            try:
                # Clean the input text
                cleaned_input = clean_text(input_text, stopwords)
                words = cleaned_input.split()
                
                # Count positive and negative words
                pos_count = sum(1 for word in words if word in positive_words)
                neg_count = sum(1 for word in words if word in negative_words)
                
                # Determine sentiment based on word counts
                if pos_count >= neg_count:
                    prediction = 'Positive'
                else:
                    prediction = 'Negative'
                
                # Display results with more detail
                st.write(f"Sentimen yang Diprediksi: {prediction}")
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan dalam pemrosesan teks: {str(e)}")

