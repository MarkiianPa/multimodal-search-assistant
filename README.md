# 🔍 Multimodal Search Assistant

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
git clone https://github.com/MarkiianPa/multimodal-search-assistant.git
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
> Simply run Qdrant in Docker and restore the snapshot.
>
> Make sure Qdrant is running locally (e.g., via Docker):
>
> ```bash
> docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
> ```
>
> Then open this link in your browser:
> [http://localhost:6333/dashboard#/collections](http://localhost:6333/dashboard#/collections)  
> Click on **"Upload snapshot"**, choose your `.snapshot` snapshot file, and restore it.
>
> This will recreate the entire vector database without needing to re-scrape or reprocess data.
>
> 📂 **Also:** Make sure you have the associated image files from the media dataset.  
> Place all required images in the `data/media/` directory of the project.  
> This is important for image rendering and evaluation in the app.


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
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Then:

```bash
python ingest_data.py
```

---

## 💻 Launch the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🧪 Evaluation (Optional)

You can evaluate how well the system retrieves relevant content for multiple queries by running the evaluation script.

Run `evaluating.py` and pass a JSON-encoded list of queries as a command-line argument:

Example:

```bash
python evaluating.py --queries "[\"What is reinforcement learning?\", \"Recent breakthroughs in AI\"]"
```
