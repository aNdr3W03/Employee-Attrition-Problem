# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Human Resource

## Business Understanding

Jaya Jaya Maju adalah salah satu perusahaan multinasional yang telah beroperasi sejak tahun 2000 dan memiliki lebih dari 1000 karyawan yang tersebar di seluruh penjuru negeri. Meskipun telah berkembang menjadi perusahaan yang berskala besar, Jaya Jaya Maju mengalami tantangan signifikan dalam pengelolaan sumber daya manusia. Salah satu dampak dari tantangan tersebut adalah tingginya attrition rate, dengan rasio karyawan yang meninggalkan perusahaan mencapai lebih dari 10%.

### Business Problem

**Masalah bisnis utama yang dihadapi oleh perusahaan Jaya Jaya Maju adalah tingginya attrition rate, yang saat ini berada di atas 10%**. Tingginya angka ini mengindikasikan bahwa jumlah karyawan yang memutuskan untuk mengundurkan diri lebih tinggi dibandingkan dengan yang tetap bertahan di perusahaan. Kondisi ini berdampak signifikan terhadap stabilitas operasional perusahaan. Alur kerja yang terganggu akibat kepergian karyawan menyebabkan pekerjaan menjadi tertunda atau dialihkan kepada karyawan lain, yang pada akhirnya meningkatkan beban kerja individu. Akumulasi beban tambahan ini dapat memperlambat produktivitas secara keseluruhan, memicu efek domino pada efisiensi perusahaan.

**Untuk mengatasi tantangan ini, penting untuk menganalisis secara kritis penyebab utama tingginya attrition rate**. Faktor-faktor yang berkontribusi dapat berasal dari aspek internal, seperti kualitas fasilitas, beban kerja, dan tingkat kepuasan karyawan, maupun dari aspek eksternal, seperti latar belakang individu karyawan. Identifikasi dan pemantauan faktor-faktor ini menjadi langkah penting dalam merancang strategi yang efektif untuk mengurangi attrition rate. Dengan pendekatan ini, perusahaan dapat memastikan keberlanjutan operasional dan meningkatkan retensi karyawan secara berkelanjutan.

### Project Scope

- Data preparation dan data cleaning awal
- Exploratory Data Analysis (EDA) untuk mencari tahu faktor penyebab attrition rate yang tinggi melalui visualisasi data
- Membuat business dashboard dari faktor penyebab tingginya attrition rate
- Mengembangkan model machine learning untuk memprediksi status karyawan berdasarkan variabel-variabel faktor penyebab attrition

### Preparation

**Data source:** [Employee data](https://github.com/dicodingacademy/dicoding_dataset/tree/main/employee 'Dicoding GitHub - Employee data')

**Setup environment:**

1. Clone this Repository
   ```bash
   git clone https://github.com/aNdr3W03/Employee-Attrition-Problem.git
   ```

2. Create Python Virtual Environment
   ```bash
   virtualenv venv
   ```

2. Activate the Environment
   ```bash
   venv\Scripts\activate
   ```

4. Install All the Requirements Inside "requirements.txt"
   ```bash
   pip install -r requirements.txt
   ```

**To run the Streamlit prediction:**
```bash
streamlit run streamlit_app.py
```

And to stop the streamlit application program by `ctrl + c`. 

## Business Dashboard

[Jaya Jaya Maju Employees Dashboard](https://public.tableau.com/views/JayaJayaMajuEmployeesDashboard/Attrition 'Tableau Public- Jaya Jaya Maju Employees Dashboard'), didesain sefektif mungkin untuk memberikan insight kepada para manajer departemen HR terkait attrition rate yang cukup tinggi hingga lebih dari 10%. Dashboard ini juga didesain dengan visualisasi dan pemilihan warna yang inklusif bagi penderita color blind.

![Jaya Jaya Maju Employees Dashboard](https://github.com/user-attachments/assets/0b6e67e1-8f07-48fd-b024-a4eb207dee55 'Jaya Jaya Maju Employees Dashboard')

Pada dashboard ini, terdapat 3 kolom, kolom pertama menunjukkan analisis dari distribusi jenis kelamin, status pernikahan, dan tingkat pendidikan yang telah ditempuh para pegawai. Pada kolom kedua menunjukkan grafik analisis attrition rate berdasarkan umur, peran pekerjaan dan keterlibatan pekerjaan. Sedangkan pada kolom ketiga terdapat jumlah attrition rate, rata-rata jumlah tahun berkerja setiap pegawai, rata-rata pendapatan pegawai, dan bubble chart jumlah attrition rate berdasarkan tingkat work life balance dan tingkat kepuasan kerja.

> Dari **pie chart jenis kelamin**, pekerja laki-laki lebih banyak, sebanyak 620 (58.60%) pekerja laki-laki dan 438 (41.40%) pekerja perempuan.
> Dari **pie chart status pernikahan**, pekerja yang telah menikah sebanyak 464 (43.86%), sebanyak 352 (33.27%) pekerja yang belum menikah, dan sebanyak 242 (22.87%) pekerja yang bercerai.
> Sedangkan dari **pie chart tingkat pendidikan**, pekerja yang belum kuliah sebanyak 131 (12.38%), pekerja yang sedang kuliah sebanyak 208 (19.66%), pekerja yang sudah sarjana sebesar 410 (38.75%), pekerja yang sudah magister sebesar 276 (26.09%), dan pekerja yang sudah doktor sebesar 33 (3.12%).

> Berdasarkan **grafik atrrition rate dengan usia**, jumlah pegawai yang melakukan attrition paling banyak terjadi pada usia 19 tahun, dan pada usia dua puluhan dan tiga puluhan. Sedangkan attrition rate paling tinggi terjadi pada usia 31 tahun. Dan pegawai pada usia lima puluhan seperti 53, 54, 57, 59 justru lebih memilih untuk bertahan di perusahaannya.
> Berdasarkan **grafik attrition rate dengan peran pekerjaan**, jumlah pegawai yang melakukan attrition paling tinggi adalah mereka yang memiliki peran Laboratory Technician, sedangkan yang paling sedikit adalah mereka yang memiliki peran Research Director. Selain itu, pegawai yang memiliki peran Sales Representative memiliki attrition rate yang cukup tinggi yaitu hampir mendekati populasi secara keseluruhan.
> Berdasarkan **grafik attrition rate dengan keterlibatan pekerjaan**, pegawai yang memiliki keterlibatan pekerjaan rendah memiliki kecenderungan melakukan attrition yang terlihat jelas mendekati populasi secara keseluruhan.

> Dari **grafik work life balance**, pekerja yang memiliki attrition rate tertinggi datang dari pekerja dengan worklife balance paling rendah, yaitu sebesar 32.14% pekerja yang memiliki worklife balance rendah, yaitu 18 pekerja yang melakukan attrition, dan 38 pekerja yang tidak melakukannya. Sedangkan pekerja yang memiliki work life balance sangat tinggi justru menduduki di peringkat ke-2 dengan attrition rate tertinggi setelah work life balance rendah, yaitu sebesar 19.47% dengan 22 pekerja yang melakukan attrition dan 91 pekerja yang tidak.
> Dari **grafik kepuasan kerja**, pekerja yang memiliki attrition rate tertinggi juga datang dari pekerja dengan tingkat kepuasan kerja yang rendah, yaitu sebesar 22.4% dengan 46 pekerja yang melakukan attrition dan 159 pekerja yang tidak melakukannya. Sedangkan pekerja dengan tingkat kepuasan kerja paling tinggi adalah pekerja dengan tingkat attrition paling rendah, yaitu 11.47%, sebesar 39 pekerja yang melakukan attrition dan 301 pekerja yang tidak.

## Conclusion

Berikut adalah beberapa poin penting yang bisa ditarik menjadi kesimpulan:
-   **Demografi Karyawan**: Mayoritas pekerja adalah laki-laki (58.6%), dengan sebagian besar telah menikah (43.86%) dan berpendidikan sarjana (38.75%).
-   **Attrition Rate Berdasarkan Usia**: Pekerja dengan usia 19-30 tahun memiliki tingkat attrition tertinggi, dengan puncak pada usia 31 tahun. Pekerja usia 50-an cenderung lebih memilih untuk bertahan.
-   **Attrition Berdasarkan Peran**: Posisi sebagai Laboratory Technician menunjukkan tingkat attrition tertinggi, sementara Research Director memiliki attrition terendah.
-   **Keterlibatan Kerja**: Karyawan dengan tingkat keterlibatan rendah memiliki kecenderungan untuk melakukan attrition lebih tinggi.
-   **Work-Life Balance**: Karyawan dengan work-life balance rendah memiliki tingkat attrition tertinggi, diikuti oleh mereka dengan work-life balance sangat tinggi.
-   **Kepuasan Kerja**: Karyawan dengan tingkat kepuasan kerja rendah menunjukkan tingkat attrition tertinggi, sementara mereka dengan kepuasan kerja tinggi memiliki tingkat attrition terendah.

### Recommended Action Items

Berikut beberapa rekomendai yang dapat dilakukan oleh perusahaan untuk mengatasi permasalahan attrition rate:
- Mengenai kerterlibatan pekerja, penting untuk menciptakan lingkungan kerja yang lebih inklusif dan berorientasi pada perkembangan karier.
- Evaluasi dan tingkatkan kebijakan kerja fleksibel, sebab pekerja dengan work-life balance rendah memiliki tingkat attrition yang relatif tinggi agar dapat menjaga keseimbangan dan dapat membantu mengurangi tingkat perputaran karyawan.
- Fokus pada peningkatan kepuasan kerja dengan memberikan lebih banyak umpan balik positif, pengakuan atas pencapaian, serta peluang untuk meningkatkan kesejahteraan karyawan.
- Mengingat attrition rate yang tinggi di usia 19-30 tahun, perusahaan bisa membuat program khusus untuk mempertahankan karyawan muda, seperti mentorship, peluang pengembangan karier cepat, atau pengakuan atas kontribusi mereka.
- Lakukan pemantauan berkelanjutan terhadap attrition rate dan lakukan deep analysis untuk mengidentifikasi trend dan faktor-faktor risiko lebih dini. Hal ini akan memungkinkan perusahaan untuk mengantisipasi potensi masalah dan mengambil tindakan preventif.
- Prediksi apakah pekerja memiliki kemungkinan untuk melakukan attrition, dapat dilakukan melalui website prediksi [berikut ini](https://employee-attrition-predict.streamlit.app 'Jaya Jaya Maju Employee Attrition Prediction').
