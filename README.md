# Python for Generative AI

Repository ini berisi kumpulan materi, contoh program, latihan, dan mini project Python yang disusun sebagai dasar untuk mempelajari **Generative AI**. Materi dimulai dari konsep dasar Python, dilanjutkan dengan struktur data, Object-Oriented Programming (OOP), pengolahan file dan API, hingga pengolahan data menggunakan NumPy dan Pandas.

## Identitas

| Keterangan    | Data                                   |
| ------------- | -------------------------------------- |
| Nama          | Hanan Hafizhah Zarkasi                 |
| NRP           | 5323600015                             |
| Program Studi | Teknologi Rekayasa Multimedia          |
| Fakultas      | Jurusan Teknologi Multimedia Kreatif   |
| Universitas   | Politeknik Elektronika Negeri Surabaya |
| Mata Kuliah   | Gen AI                                 |
| Kelas         | TRM 2023                               |

## Isi Repository

Repository ini terbagi menjadi lima modul utama:

| Modul                        | Topik yang Dipelajari                                                                                                                                           |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Module01_PythonFoundations` | Variabel dan tipe data, string, number, control flow, function, `*args` dan `**kwargs`, lambda, scope, closure, exception handling, serta latihan dasar Python. |
| `Module02_DataStructures`    | List, dictionary, set, tuple, comprehension, generator, dan latihan penggunaan struktur data.                                                                   |
| `Module03_OOP_Modules`       | Class dan object, inheritance, decorator, dataclass, module, package, serta latihan OOP seperti rate limiter, prompt template, dan retry decorator.             |
| `Module04_FileIO_APIs`       | Operasi file teks, JSON, CSV, async/await, simulasi pemanggilan LLM, environment variable, pengelolaan secret, dan latihan File I/O.                            |
| `Module05_PythonForData`     | NumPy, cosine similarity, dasar-dasar Pandas, data cleaning, `groupby`, agregasi, serta mini project pipeline evaluasi model.                                   |

Beberapa folder juga memiliki direktori `outputs` yang berisi tangkapan layar hasil eksekusi program. Data contoh dan hasil pengolahan disimpan dalam format seperti `.txt`, `.json`, dan `.csv`.

## Mini Project

Pada Modul 05 terdapat mini project **Model Evaluation Pipeline**. Program ini menyimulasikan respons beberapa model LLM, menghitung skor berdasarkan kata kunci, mengukur latency, menganalisis hasil menggunakan Pandas, lalu menyimpannya ke dalam file CSV.

## Teknologi yang Digunakan

- Python 3.10 atau lebih baru
- NumPy
- Pandas
- HTTPX
- python-dotenv

## Cara Menjalankan Project

1. Clone repository:

   ```bash
   git clone https://github.com/USERNAME/NAMA-REPOSITORY.git
   cd NAMA-REPOSITORY
   ```

2. Buat dan aktifkan virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   Linux/macOS:

   ```bash
   source .venv/bin/activate
   ```

3. Instal seluruh dependency:

   ```bash
   pip install -r requirements.txt
   ```

4. Jalankan file Python yang ingin dipelajari. Contoh:

   ```bash
   python Module01_PythonFoundations/01_variables_and_types.py
   python Module05_PythonForData/05_mini_project_eval_pipeline.py
   ```

## Struktur Folder

```text
PythonForGenAI/
├── Module01_PythonFoundations/
├── Module02_DataStructures/
├── Module03_OOP_Modules/
├── Module04_FileIO_APIs/
├── Module05_PythonForData/
├── requirements.txt
└── README.md
```

## Tujuan Pembelajaran

Setelah mempelajari isi repository ini, diharapkan pembaca dapat:

- Memahami dasar-dasar pemrograman Python.
- Menggunakan struktur data Python secara tepat.
- Menerapkan konsep OOP, module, dan package.
- Membaca, menulis, dan mengolah berbagai format file.
- Memahami dasar pemanggilan API dan proses asynchronous.
- Mengolah serta menganalisis data dengan NumPy dan Pandas.
- Memahami gambaran awal pipeline evaluasi model Generative AI.

## Catatan

Sebagian contoh pemanggilan LLM pada repository ini masih berupa simulasi sehingga dapat dijalankan secara lokal tanpa API key dan tanpa melakukan request ke layanan model eksternal.

---

Dibuat untuk memenuhi tugas Gen AI.
