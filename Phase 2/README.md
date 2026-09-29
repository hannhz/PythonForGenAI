# Checklist Phase 2

Phase 2 sudah memiliki implementasi untuk Module 06-08. Instal dependency dari root repository terlebih dahulu:

```powershell
pip install -r requirements.txt
```

Untuk percobaan online, salin `.env.example` di folder modul menjadi `.env` dan isi key yang diperlukan. Panggilan API adalah panggilan nyata. Eksperimen teks di Module 06-07 otomatis memprioritaskan Groq jika `GROQ_API_KEY` tersedia, sehingga key Anthropic yang tidak memiliki kredit tidak ikut dipanggil.

Konfigurasi Groq yang direkomendasikan:

```env
GROQ_API_KEY=gsk_...
LLM_PROVIDER=groq
```

Untuk demo vision, Groq tetap dapat dipakai sebagai provider teks dan Gemini
dipilih secara terpisah:

```env
GEMINI_API_KEY=...
VISION_PROVIDER=gemini
GEMINI_MODEL=gemini-3.8-flash
```

Adapter bersama berada di `shared_provider.py` dan menggunakan SDK `openai` yang sudah ada, dengan endpoint kompatibel Groq. Gunakan `LLM_PROVIDER=anthropic` hanya jika ingin memaksa contoh asli memakai Anthropic.

## Module 06 - LLM APIs

| Percobaan | Kebutuhan | Status kode | Yang masih perlu dilengkapi untuk laporan |
| --- | --- | --- | --- |
| `01_setup.py` | Offline + `.env` untuk cek key | Terverifikasi offline | Screenshot kondisi key terdeteksi tanpa menampilkan key lengkap. |
| `02_anthropic_basic.py` | Groq atau Anthropic | Groq terverifikasi | Simpan output teks serta jumlah token. |
| `03_anthropic_system_prompt.py` | Groq atau Anthropic | Groq terverifikasi | Simpan output yang menunjukkan pengaruh system prompt. |
| `04_anthropic_multiturn_chat.py` | Groq atau Anthropic | Fallback siap; uji interaktif manual | Dokumentasikan minimal dua turn percakapan. |
| `05_anthropic_streaming.py` | Groq atau Anthropic | Groq terverifikasi | Simpan hasil akhir streaming dan total token. |
| `06_openai_basic.py` | Groq atau OpenAI | Groq terverifikasi | Simpan output serta token usage. |
| `07_openai_streaming.py` | Groq atau OpenAI | Groq terverifikasi | Dokumentasikan keluaran streaming. |
| `08_anthropic_tool_calling.py` | Groq atau Anthropic | Groq terverifikasi | Simpan bukti tool input, tool result, dan final answer. |
| `09_openai_tool_calling.py` | Groq atau OpenAI | Groq terverifikasi | Simpan bukti tool call dan final answer. |
| `10_vision.py` | `GEMINI_API_KEY` atau funded `ANTHROPIC_API_KEY` + gambar/URL | Fallback Gemini siap; menunggu key untuk uji API | Cantumkan gambar sumber dan deskripsi hasil model. |
| `11_token_management.py` | Offline; Anthropic opsional untuk token count | Offline terverifikasi; Groq mode melewati endpoint khusus Anthropic | Simpan hasil estimasi biaya dan context check. |
| `12_provider_agnostic_client.py` | Key provider yang dipilih | Backend Groq terverifikasi | Simpan hasil backend yang tersedia dan jelaskan abstraksinya. |
| `13_exercises.py` | Latihan 1-2 offline; latihan 3-4 online | Offline dan Groq terverifikasi | Simpan output semua latihan termasuk model comparison dan stream-to-file. |

## Module 07 - Prompt Engineering

| Percobaan | Kebutuhan | Status kode | Yang masih perlu dilengkapi untuk laporan |
| --- | --- | --- | --- |
| `01_system_prompts.py` | Groq atau Anthropic | Fallback Groq siap | Bandingkan hasil weak vs strong system prompt. |
| `02_few_shot_prompting.py` | Groq atau Anthropic | Fallback Groq siap | Validasi bahwa ketiga output dapat diparse sebagai JSON. |
| `03_chain_of_thought.py` | Groq atau Anthropic | Fallback Groq siap | Simpan perbandingan direct, CoT, zero-shot CoT, dan hasil format terstruktur. |
| `04_structured_output_json.py` | Groq, Anthropic, atau OpenAI | Dua pendekatan Groq terverifikasi | Tunjukkan hasil `json.loads`. |
| `05_prompt_templates.py` | Offline | Terverifikasi offline | Simpan hasil render dan bukti error saat variabel kurang. |
| `06_prompt_evaluation.py` | Groq atau Anthropic | Fallback Groq siap | Simpan pass rate serta detail PASS/FAIL. |
| `07_exercises.py` | Offline + Groq/Anthropic untuk bagian online | Offline terverifikasi; fallback Groq siap | Jalankan evaluasi 3 prompt x 5 snippet, uji save/load library, tiga tier JSON repair, dan validasi 3 input ranking. |

## Module 08 - Embeddings & Semantic Search

| Percobaan | Kebutuhan | Status kode | Yang masih perlu dilengkapi untuk laporan |
| --- | --- | --- | --- |
| `01_generating_embeddings.py` | `OPENAI_API_KEY` | Kode siap; API belum diuji | Simpan shape vector dan norm hasil embedding. |
| `02_cosine_similarity.py` | Offline | Terverifikasi offline | Simpan similarity matrix dan jelaskan diagonalnya. |
| `03_vector_store_semantic_search.py` | `OPENAI_API_KEY` | Kode siap; API belum diuji | Simpan top-3 untuk setiap query beserta score/rank. |
| `04_chunking_text.py` | Offline | Terverifikasi offline | Simpan isi, indeks karakter, dan overlap setiap chunk. |
| `05_retrieval_evaluation.py` | Metric offline; evaluasi store online | Offline terverifikasi; API belum diuji | Simpan precision@k, recall@k, dan MRR untuk mock serta real retrieval. |
| `06_metadata_filtered_search.py` | `OPENAI_API_KEY` | Kode siap; API belum diuji | Bandingkan hasil tanpa filter dan filter metadata. |
| `07_exercises.py` | Offline dengan mock; key opsional untuk embedding nyata | Terverifikasi offline | Simpan hasil duplicate detection, hybrid search, delete/update, dan bukti cache hit. |

## Kekurangan bukti percobaan

Kode sumber sudah tersedia, tetapi Phase 2 belum memiliki folder `outputs` atau screenshot/log hasil eksekusi seperti Phase 1. Jadi yang paling utama untuk dilengkapi adalah bukti run pada kolom terakhir. Output online tidak dapat dibuat tanpa API key pengguna dan tidak dijalankan otomatis agar tidak menimbulkan biaya.
