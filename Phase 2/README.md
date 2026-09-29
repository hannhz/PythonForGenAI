# Checklist Phase 2

Phase 2 sudah memiliki implementasi dan bukti eksekusi untuk Module 06-08. Instal dependency dari root repository terlebih dahulu:

```powershell
pip install -r requirements.txt
```

Untuk percobaan online, salin `Module06_LLM_APIs/.env.example` menjadi
`Module06_LLM_APIs/.env`, lalu isi key yang diperlukan. Konfigurasi tersebut
digunakan bersama oleh Module 06-08 sehingga API key tidak perlu disalin ke
setiap folder. Panggilan API adalah panggilan nyata. Eksperimen teks di Module
06-07 otomatis memprioritaskan Groq jika `GROQ_API_KEY` tersedia, sehingga key
Anthropic yang tidak memiliki kredit tidak ikut dipanggil.

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

Adapter teks bersama berada di `shared_provider.py` dan menggunakan endpoint
OpenAI-compatible milik Groq. Adapter embedding Module 08 berada di
`Module08_Embeddings_SemanticSearch/_embedding_provider.py` serta mendukung
OpenAI dan Gemini. Gunakan `LLM_PROVIDER=anthropic` hanya jika ingin memaksa
contoh asli memakai Anthropic.

## Module 06 - LLM APIs

| Percobaan | Kebutuhan | Status akhir | Bukti output |
| --- | --- | --- | --- |
| `01_setup.py` | Offline + `.env` untuk cek key | Terverifikasi | Tersedia di folder `outputs`. |
| `02_anthropic_basic.py` | Groq atau Anthropic | Groq terverifikasi | Teks dan token usage terdokumentasi. |
| `03_anthropic_system_prompt.py` | Groq atau Anthropic | Groq terverifikasi | Pengaruh system prompt terdokumentasi. |
| `04_anthropic_multiturn_chat.py` | Groq atau Anthropic | Groq terverifikasi | Percakapan multi-turn terdokumentasi. |
| `05_anthropic_streaming.py` | Groq atau Anthropic | Groq terverifikasi | Streaming dan total token terdokumentasi. |
| `06_openai_basic.py` | Groq atau OpenAI | Groq terverifikasi | Respons dan token usage terdokumentasi. |
| `07_openai_streaming.py` | Groq atau OpenAI | Groq terverifikasi | Keluaran streaming terdokumentasi. |
| `08_anthropic_tool_calling.py` | Groq atau Anthropic | Groq terverifikasi | Tool input, tool result, dan final answer terdokumentasi. |
| `09_openai_tool_calling.py` | Groq atau OpenAI | Groq terverifikasi | Tool call dan final answer terdokumentasi. |
| `10_vision.py` | Gemini atau Anthropic + gambar/URL | Gemini terverifikasi | Gambar Grace Hopper dan deskripsi model terdokumentasi. |
| `11_token_management.py` | Offline; Anthropic opsional untuk token count | Terverifikasi offline | Estimasi biaya dan context check terdokumentasi. |
| `12_provider_agnostic_client.py` | Key provider yang dipilih | Groq terverifikasi | Hasil backend provider terdokumentasi. |
| `13_exercises.py` | Latihan 1-2 offline; latihan 3-4 online | Offline dan Groq terverifikasi | Seluruh latihan terdokumentasi. |

## Module 07 - Prompt Engineering

| Percobaan | Kebutuhan | Status akhir | Bukti output |
| --- | --- | --- | --- |
| `01_system_prompts.py` | Groq atau Anthropic | Groq terverifikasi | Review dari strong system prompt terdokumentasi. |
| `02_few_shot_prompting.py` | Groq atau Anthropic | Groq terverifikasi | Ketiga output JSON berhasil diparse. |
| `03_chain_of_thought.py` | Groq atau Anthropic | Groq terverifikasi | Direct, CoT, zero-shot CoT, dan structured result terdokumentasi. |
| `04_structured_output_json.py` | Groq, Anthropic, atau OpenAI | Groq terverifikasi | Kedua pendekatan dan hasil parsing JSON terdokumentasi. |
| `05_prompt_templates.py` | Offline | Terverifikasi offline | Render dan validasi variabel terdokumentasi. |
| `06_prompt_evaluation.py` | Groq atau Anthropic | Groq terverifikasi | Pass rate dan detail PASS/FAIL terdokumentasi. |
| `07_exercises.py` | Offline + Groq/Anthropic untuk bagian online | Offline dan Groq terverifikasi | Tiga level review, prompt library, JSON repair, dan ranking terdokumentasi. |

## Module 08 - Embeddings & Semantic Search

Percobaan embedding memprioritaskan `OPENAI_API_KEY`. Jika key tersebut tidak
tersedia, Module 08 otomatis memakai `GEMINI_API_KEY` dari konfigurasi Module
06 dengan `gemini-embedding-001` dan output 1536 dimensi.

| Percobaan | Kebutuhan | Status akhir | Bukti output |
| --- | --- | --- | --- |
| `01_generating_embeddings.py` | OpenAI atau Gemini | Gemini terverifikasi | Shape `(5, 1536)` dan norm `1.0000` terdokumentasi. |
| `02_cosine_similarity.py` | Offline | Terverifikasi offline | Pairwise similarity dan diagonal matrix terdokumentasi. |
| `03_vector_store_semantic_search.py` | OpenAI atau Gemini | Gemini terverifikasi | Top-3 beserta score/rank setiap query terdokumentasi. |
| `04_chunking_text.py` | Offline | Terverifikasi offline | Isi, indeks karakter, dan overlap chunk terdokumentasi. |
| `05_retrieval_evaluation.py` | Metric offline; OpenAI/Gemini untuk evaluasi store | Gemini terverifikasi | Precision@k, recall@k, dan MRR terdokumentasi. |
| `06_metadata_filtered_search.py` | OpenAI atau Gemini | Gemini terverifikasi | Hasil tanpa filter dan filter metadata terdokumentasi. |
| `07_exercises.py` | Offline dengan mock; OpenAI/Gemini opsional | Terverifikasi offline | Duplicate detection, hybrid search, update/delete, dan cache hit terdokumentasi. |

## Status bukti percobaan

Folder `outputs` untuk Module 06, 07, dan 08 sudah berisi screenshot hasil
eksekusi. Percobaan online sudah dijalankan menggunakan Groq untuk text
generation, Gemini untuk vision, serta Gemini Embeddings untuk semantic search.
Nilai similarity dan susunan kalimat jawaban dapat sedikit berubah antar-run,
tetapi struktur output dan tujuan setiap latihan tetap sama dengan repository
acuan.
