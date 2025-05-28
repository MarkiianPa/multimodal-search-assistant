# 🔍 Multimodal Search Assistant (Text + Image)

This project implements a **multimodal Retrieval-Augmented Generation (RAG)** system combining **text and image search** using vector embeddings and large language models. It supports semantic querying, ranks results across modalities, and presents answers via an interactive web UI.

---

## 📁 Project Structure

```
├── app.py                   # Streamlit frontend
├── scrapper.py              # Scrapes articles from DeepLearning.ai
├── media_downloader.py      # Downloads and stores media locally
├── ingest_data.py           # Embeds and ingests data into Qdrant
├── evaluating.py            # Evaluation scripts (Precision@K, Recall@K)
├── LLM_search.py            # Query handling, retrieval, Gemini integration
├── requirements.txt         # Python dependencies
├── README.md                # Documentation (this file)
└── data/
    ├── the_batch_articles.csv           # Raw scraped content
    ├── articles_with_local_images.csv   # Content with local image paths
    └── media/                           # Downloaded media files
```

---

## ⚙️ Features

- 🔍 Semantic search across both **text** and **image** data.
- 🤖 Uses **CLIP** and **SentenceTransformer** for vectorization.
- 📦 Stores and retrieves vectors using **Qdrant**.
- 🧠 Leverages **Gemini 2.0 LLM** to rank and answer queries.
- 🖼️ Visual frontend built with **Streamlit**.

---

## 🚀 Quick Start

### 1. 🔧 Clone the Repository

```bash
git clone https://github.com/your-username/multimodal-search-assistant.git
cd multimodal-search-assistant
```

### 2. 🐍 Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
```

### 3. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. 🔐 Set Up Environment Variables

Create a `data.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

---


## 📊 Data Pipeline

> 🧠 **Tip:** You can **skip Step 1, Step 2, and Step 3** if you already have a snapshot of the database.
> 
> Simply run Qdrant in Docker and restore the snapshot as shown in the [Backing Up Qdrant](#-backing-up-qdrant) section.
> This will recreate the entire vector database without needing to re-scrape or reprocess data.


### Step 1: Scrape Articles

```bash
python scrapper.py
```

### Step 2: Download Media

```bash
python media_downloader.py
```

### Step 3: Ingest Data into Qdrant

Make sure Qdrant is running locally (e.g., via Docker):

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Then:

```bash
python ingest_data.py
```

---

## 🧪 Evaluation (Optional)

You can evaluate how well the system retrieves relevant content for multiple queries by running the evaluation script.

Run `evaluating.py` and pass a JSON-encoded list of queries as a command-line argument:

Example:


```bash
python evaluating.py --queries '["What is reinforcement learning?", "Recent breakthroughs in AI"]'
```

---

## 💻 Launch the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🧠 Models Used

- **Sentence Transformer**: `intfloat/e5-base`
- **Image Model**: `CLIP ViT-L-14` (OpenCLIP)
- **LLM**: Google Gemini 2.0 Flash
- **Vector DB**: [Qdrant](https://qdrant.tech/)

---

## 💾 Backing Up Qdrant

To create a snapshot (backup):

```bash
curl -X POST "http://localhost:6333/collections/articles_collection/snapshots"
```

To restore:

```bash
curl -X POST "http://localhost:6333/collections/articles_collection/snapshots/recover"      -H "Content-Type: application/json"      -d '{"location": "snapshots/articles_collection/your_snapshot_file.tar.gz"}'
```

---

## 📜 License

MIT License — see `LICENSE` file for details.

---

## 🤝 Contributing

Feel free to fork this project and contribute via pull requests.