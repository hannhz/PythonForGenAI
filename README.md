# Python for Generative AI

Repository ini berisi materi, contoh program, latihan, dan mini project Python untuk mempelajari **Generative AI**. Materi dibagi menjadi dua fase: dasar Python dan data pada Phase 1, lalu API LLM, prompt engineering, embedding, dan semantic search pada Phase 2.

## Identitas

| Keterangan | Data |
| --- | --- |
| Nama | Hanan Hafizhah Zarkasi |
| NRP | 5323600015 |
| Program Studi | Teknologi Rekayasa Multimedia |
| Fakultas | Jurusan Teknologi Multimedia Kreatif |
| Universitas | Politeknik Elektronika Negeri Surabaya |
| Mata Kuliah | Gen AI |
| Kelas | TRM 2023 |

## Isi Repository

| Modul | Topik yang Dipelajari |
| --- | --- |
| `Module01_PythonFoundations` | Variabel, tipe data, string, number, control flow, function, lambda, scope, closure, dan exception handling. |
| `Module02_DataStructures` | List, dictionary, set, tuple, comprehension, dan generator. |
| `Module03_OOP_Modules` | Class, inheritance, decorator, dataclass, module, dan package. |
| `Module04_FileIO_APIs` | File teks, JSON, CSV, async/await, simulasi API, environment variable, dan secret. |
| `Module05_PythonForData` | NumPy, cosine similarity, Pandas, data cleaning, agregasi, dan pipeline evaluasi model. |
| `Module06_LLM_APIs` | Anthropic/OpenAI SDK, streaming, tool calling, vision, token management, dan abstraction client. |
| `Module07_PromptEngineering` | System prompt, few-shot, chain-of-thought, structured output, template, dan prompt evaluation. |
| `Module08_Embeddings_SemanticSearch` | Embedding, cosine similarity, vector store, chunking, retrieval evaluation, metadata filter, hybrid search, dan cache. |

Checklist kebutuhan dan kekurangan bukti untuk setiap percobaan Phase 2 tersedia di [Phase 2/README.md](Phase%202/README.md).

## Teknologi

- Python 3.10 atau lebih baru
- NumPy dan Pandas
- HTTPX dan python-dotenv
- Anthropic Python SDK
- OpenAI Python SDK

## Cara Menjalankan

1. Clone repository dan masuk ke folder project.
2. Buat virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Instal dependency:

   ```powershell
   pip install -r requirements.txt
   ```

4. Jalankan percobaan yang diinginkan, misalnya:

   ```powershell
   python "Phase 1/Module01_PythonFoundations/01_variables_and_types.py"
   python "Phase 1/Module05_PythonForData/05_mini_project_eval_pipeline.py"
   python "Phase 2/Module08_Embeddings_SemanticSearch/02_cosine_similarity.py"
   ```

Percobaan online memerlukan `.env`. Salin `.env.example` pada modul terkait menjadi `.env`, lalu isi key milik sendiri. Panggilan API dapat menimbulkan biaya; percobaan yang bisa dijalankan offline ditandai pada checklist Phase 2.

## Struktur Folder

```text
PythonForGenAI/
|-- Phase 1/
|   |-- Module01_PythonFoundations/
|   |-- Module02_DataStructures/
|   |-- Module03_OOP_Modules/
|   |-- Module04_FileIO_APIs/
|   `-- Module05_PythonForData/
|-- Phase 2/
|   |-- Module06_LLM_APIs/
|   |-- Module07_PromptEngineering/
|   `-- Module08_Embeddings_SemanticSearch/
|-- requirements.txt
`-- README.md
```

## Mini Project

Module 05 memiliki mini project **Model Evaluation Pipeline** yang menyimulasikan respons beberapa model LLM, menghitung skor dan latency, menganalisis hasil dengan Pandas, lalu menyimpannya ke CSV.

## Catatan Keamanan

Sebagian contoh dapat dijalankan offline. Contoh lain melakukan request nyata ke Anthropic/OpenAI dan sengaja dilewati saat API key belum tersedia. Jangan commit file `.env` atau menaruh API key langsung di source code.

---

Dibuat untuk memenuhi tugas Gen AI.
