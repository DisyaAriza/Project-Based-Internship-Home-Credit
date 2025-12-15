Home Credit Default Risk – Project Based Learning Internship

Project ini bertujuan untuk memprediksi risiko gagal bayar (default) nasabah menggunakan dataset internal dari Home Credit yang diperoleh melalui program Project Based Learning (PBL) Internship melalui platform Rakamin dan bersifat terbatas serta tidak untuk disebarluaskan. Project ini disusun untuk menerapkan konsep data science dan machine learning pada studi kasus nyata di industri pembiayaan.

Analisis dimulai dengan exploratory data analysis (EDA) untuk memahami karakteristik data, termasuk distribusi target, profil demografi nasabah, serta hubungan antara fitur finansial dan risiko default. Hasil EDA digunakan sebagai dasar dalam proses feature engineering, khususnya dalam membangun fitur yang merepresentasikan riwayat kredit, pengajuan pinjaman sebelumnya, dan perilaku pembayaran cicilan nasabah.

Tahap data preprocessing meliputi penggabungan beberapa dataset menggunakan SK_ID_CURR sebagai key identifier, penanganan missing value dengan pendekatan median untuk data numerik dan kategori “Unknown” untuk data kategorikal, serta encoding variabel kategorikal. Selanjutnya dilakukan feature scaling menggunakan StandardScaler dan pembagian data menjadi data latih dan data uji dengan proporsi 80:20 untuk memastikan evaluasi model yang objektif.

Model machine learning yang digunakan dalam project ini adalah Logistic Regression, yang dipilih karena kemampuannya dalam menangani permasalahan klasifikasi biner serta interpretabilitasnya yang tinggi dalam konteks credit scoring. Evaluasi performa model dilakukan menggunakan AUC–ROC, confusion matrix, dan classification report untuk mengukur kemampuan model dalam membedakan nasabah berisiko dan tidak berisiko.

Hasil analisis menunjukkan bahwa model mampu memberikan prediksi risiko default yang stabil dan dapat digunakan sebagai decision support system dalam proses persetujuan kredit. Berdasarkan probabilitas risiko yang dihasilkan, perusahaan dapat menerapkan kebijakan berbasis risiko, seperti persetujuan otomatis untuk nasabah berisiko rendah, peninjauan manual untuk risiko menengah, serta penolakan atau pembatasan limit kredit untuk nasabah berisiko tinggi.

Note: Dataset yang digunakan dalam project ini bersifat internal dan confidential, serta tidak tersedia untuk publik.
