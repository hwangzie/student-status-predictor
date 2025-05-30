# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

**Nama:** Muhammad Za'im Shidqi  
**Email:** muhammad.zaim67@gmail.com  
**ID Dicoding:** muhammad_zaim_shidqi

## Business Understanding

Jaya Jaya Institut adalah institusi pendidikan tinggi yang berkomitmen untuk memberikan pendidikan berkualitas tinggi kepada mahasiswanya. Namun, seperti banyak institusi pendidikan lainnya, Jaya Jaya Institut menghadapi tantangan serius dalam hal tingkat dropout (putus kuliah) mahasiswa yang cukup tinggi.

Tingkat dropout yang tinggi tidak hanya berdampak pada reputasi institusi, tetapi juga menyebabkan kerugian finansial yang signifikan. Selain itu, dropout mahasiswa juga mencerminkan kegagalan dalam memberikan dukungan yang memadai kepada mahasiswa untuk menyelesaikan pendidikan mereka.

### Permasalahan Bisnis

1. **Tingkat Dropout Tinggi**: Jaya Jaya Institut mengalami tingkat dropout mahasiswa yang cukup tinggi, yang berdampak negatif pada reputasi dan finansial institusi.

2. **Kurangnya Sistem Early Warning**: Tidak adanya sistem peringatan dini untuk mengidentifikasi mahasiswa yang berisiko dropout sebelum mereka benar-benar keluar.

3. **Keterbatasan Monitoring**: Sulitnya memantau performa dan status mahasiswa secara real-time untuk mengambil tindakan preventif.

4. **Alokasi Resources yang Tidak Optimal**: Kesulitan dalam mengalokasikan sumber daya (konselor, tutor, beasiswa) kepada mahasiswa yang paling membutuhkan.

### Cakupan Proyek

Proyek ini akan mengembangkan solusi berbasis machine learning dan business intelligence untuk:

1. **Analisis Faktor Dropout**: Mengidentifikasi faktor-faktor utama yang mempengaruhi tingkat dropout mahasiswa
2. **Sistem Prediksi**: Membangun model machine learning untuk memprediksi status mahasiswa (Dropout, Enrolled, Graduate)
3. **Dashboard Monitoring**: Membuat dashboard untuk monitoring performa mahasiswa secara real-time
4. **Early Warning System**: Mengembangkan sistem peringatan dini untuk mahasiswa berisiko tinggi
5. **Rekomendasi Action Items**: Memberikan rekomendasi strategis untuk mengurangi tingkat dropout

### Persiapan

**Sumber data**: Dataset performa mahasiswa Jaya Jaya Institut yang berisi informasi demografis, akademis, dan ekonomi mahasiswa.

**Setup environment**:
```bash
# Clone repository
git clone <repository-url>
cd sub2

# Install dependencies
pip install -r requirements.txt

# Setup Jupyter notebook
jupyter notebook sub2_proyekakhir.ipynb
```

**Dataset Features**:
- Data demografis: umur, status pernikahan, gender
- Data akademis: nilai masuk, nilai semester, unit yang diambil
- Data ekonomi: status beasiswa, tunggakan biaya, tingkat pengangguran
- Target: Status (Dropout, Enrolled, Graduate)

## Business Dashboard

Dashboard telah dibuat menggunakan **Metabase** untuk membantu Jaya Jaya Institut dalam memahami data dan memonitor performa mahasiswa secara real-time.

**Dashboard Features**:
- **Overview Dashboard**: Distribusi status mahasiswa, KPI utama
- **Risk Analysis Dashboard**: Identifikasi mahasiswa berisiko tinggi
- **Academic Performance Dashboard**: Monitoring performa akademik
- **Prediction Results Dashboard**: Hasil prediksi model ML

**Akses Dashboard**:
- **Tool**: Metabase
- **Email**: root@mail.com
- **Password**: root123
- **Database File**: metabase.db.mv.db (sudah disertakan dalam repository)

**Setup Dashboard**:
```bash
# Jalankan Metabase dengan Docker
docker run -d -p 3000:3000 --name metabase metabase/metabase

# Import database
docker cp metabase.db.mv.db metabase:/metabase-data/
```

## Menjalankan Sistem Machine Learning

Sistem machine learning telah dikembangkan menggunakan **Streamlit** sebagai interface web yang user-friendly.

**Cara menjalankan locally**:
```bash
# Install dependencies
pip install streamlit pandas scikit-learn joblib numpy

# Jalankan aplikasi
streamlit run streamlit_app.py

# Akses di browser
# http://localhost:8501
```

**Cara menggunakan**:
1. Masukkan data mahasiswa (umur, nilai, status pembayaran, dll.)
2. Klik tombol "Predict Status"
3. Sistem akan memberikan:
   - Prediksi status (Dropout/Enrolled/Graduate)
   - Tingkat confidence
   - Level risiko
   - Rekomendasi tindakan

**Link Prototype**: [Student Status Predictor - Streamlit App]([https://your-app-name.streamlit.ap](https://student-status-predictor.streamlit.app/)p)

**Setup lokal**:
```bash
streamlit run streamlit_app.py
```

**Features**:
- Interface yang mudah digunakan
- Prediksi real-time
- Analisis risiko
- Rekomendasi tindakan
- Batch prediction untuk multiple students

## Conclusion

Berdasarkan analisis yang telah dilakukan, diperoleh beberapa kesimpulan penting:

### Key Findings:

1. **Model Performance**: Model Random Forest memberikan performa terbaik dengan akurasi **85.2%** dalam memprediksi status mahasiswa.

2. **Faktor Utama Dropout**:
   - **Academic Performance**: Nilai semester 1 dan 2 yang rendah
   - **Financial Issues**: Status tunggakan biaya kuliah
   - **Age Factor**: Usia masuk yang lebih tua (>22 tahun)
   - **Previous Academic Record**: Nilai kualifikasi sebelumnya yang rendah

3. **Distribusi Status**:
   - **Graduate**: 50.2% mahasiswa berhasil lulus
   - **Dropout**: 32.1% mahasiswa dropout
   - **Enrolled**: 17.7% mahasiswa masih aktif

4. **Risk Patterns**: Mahasiswa dengan kombinasi nilai rendah dan masalah finansial memiliki risiko dropout **5x lebih tinggi**.

### Impact:
- **Early Detection**: Sistem dapat mengidentifikasi 85% mahasiswa berisiko dengan confidence tinggi
- **Resource Optimization**: Memungkinkan alokasi sumber daya yang lebih efektif
- **Proactive Intervention**: Memungkinkan tindakan preventif sebelum mahasiswa dropout

### Rekomendasi Action Items

Berdasarkan hasil analisis, berikut adalah rekomendasi strategis untuk Jaya Jaya Institut:

#### 1. **Implementasi Early Warning System**
- Deploy model prediksi secara real-time dalam sistem akademik
- Setup alert otomatis untuk konselor ketika mahasiswa masuk kategori "High Risk"
- Monitor mahasiswa dengan confidence dropout > 80% secara mingguan

#### 2. **Program Dukungan Finansial**
- **Beasiswa Darurat**: Program beasiswa khusus untuk mahasiswa berisiko tinggi
- **Sistem Cicilan Fleksibel**: Opsi pembayaran yang lebih mudah untuk mengurangi tunggakan
- **Work-Study Program**: Program kerja sambil kuliah untuk mahasiswa yang membutuhkan

#### 3. **Peningkatan Dukungan Akademik**
- **Tutorial Intensif**: Program tutorial tambahan untuk mahasiswa dengan nilai semester rendah
- **Peer Mentoring**: Sistem mentoring dari senior kepada junior
- **Academic Counseling**: Konseling akademik rutin untuk mahasiswa berisiko

#### 4. **Strategi Retensi Mahasiswa**
- **Intervention Program**: Program intervensi intensif untuk mahasiswa medium-high risk
- **Student Engagement**: Kegiatan ekstrakurikuler untuk meningkatkan engagement
- **Career Guidance**: Bimbingan karir untuk motivasi jangka panjang

#### 5. **Monitoring dan Evaluasi**
- **Dashboard Real-time**: Implementasi dashboard untuk monitoring berkelanjutan
- **Regular Assessment**: Evaluasi bulanan terhadap efektivitas program
- **Data-Driven Decision**: Pengambilan keputusan berdasarkan insights dari data

#### 6. **Capacity Building**
- **Staff Training**: Pelatihan untuk konselor dalam menggunakan sistem prediksi
- **Data Literacy**: Peningkatan kemampuan analisis data untuk tim akademik
- **Technology Adoption**: Adopsi teknologi untuk otomasi proses monitoring

### Expected Outcomes:
- **Pengurangan dropout rate** hingga 15-20%
- **Peningkatan student satisfaction** melalui dukungan yang lebih personal
- **Optimasi resource allocation** dengan fokus pada mahasiswa yang membutuhkan
- **Peningkatan reputation** institusi melalui tingkat kelulusan yang lebih baik

---

**Proyek ini menunjukkan bagaimana data science dan machine learning dapat memberikan solusi praktis untuk permasalahan nyata di dunia pendidikan, membantu institusi dalam mengambil keputusan yang lebih informed dan proaktif.**
