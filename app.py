import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import sys

# Nama file CSV Anda
NAMA_FILE = 'IT Support Ticket Data.csv'

def main():
    # 1. LOAD DATA
    print("⏳ Sedang membaca data...")
    try:
        # Membaca CSV (mengabaikan kolom Unnamed/Index jika ada)
        df = pd.read_csv(NAMA_FILE)
        
        # Bersihkan data (Hapus baris yang kosong isinya)
        df = df.dropna(subset=['Body', 'Department'])
        
        print(f"✅ Data berhasil diload! Total: {len(df)} tiket support.")
    except FileNotFoundError:
        print(f"❌ Error: File '{NAMA_FILE}' tidak ditemukan.")
        sys.exit()

    # 2. TRAINING MODEL (Melatih Komputer)
    print("⏳ Sedang melatih model kecerdasan buatan...")
    
    # X = Teks Keluhan, y = Departemen Tujuan
    X = df['Body']
    y = df['Department']

    # Membuat Pipeline: 
    # Langkah 1: TfidfVectorizer (Mengubah kata-kata menjadi angka statistik)
    # Langkah 2: MultinomialNB (Algoritma Naive Bayes yang jago menebak kategori teks)
    model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())
    
    # Proses belajar
    model.fit(X, y)
    print("✅ Model selesai dilatih dan siap digunakan!")

    # 3. DEPLOYMENT (Interface Prediksi)
    print("\n" + "="*50)
    print("   SISTEM PREDIKSI TIKET IT SUPPORT (AUTO-SORTING)")
    print("="*50)
    print("Ketik keluhan/masalah IT di bawah ini (Bahasa Inggris).")
    print("Ketik 'exit' untuk keluar.\n")

    while True:
        user_input = input("📝 Masukkan Isi Keluhan: ")
        
        if user_input.lower() == 'exit':
            print("Sampai jumpa!")
            break
        
        if len(user_input.strip()) < 5:
            print("⚠️ Mohon masukkan kalimat yang lebih lengkap.\n")
            continue

        # Prediksi
        prediksi_dept = model.predict([user_input])[0]
        probabilitas = model.predict_proba([user_input]).max() * 100
        
        print(f"➡️  Rekomendasi Departemen: [{prediksi_dept}]")
        print(f"📊 Tingkat Keyakinan Model: {probabilitas:.1f}%\n")

if __name__ == "__main__":
    main()