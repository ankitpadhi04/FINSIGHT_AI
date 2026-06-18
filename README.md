# FinSight AI

Demo app: [FinSight AI](https://finsightai-iascglfwxkzknnvoz3qg2p.streamlit.app/)

A personal finance analysis web application that processes UPI payment statements and delivers AI-powered spending insights, risk assessment, and actionable savings recommendations.

Built as a final year Computer Science project using RAG, LangGraph, fine-tuned DistilBERT, and Groq LLaMA.

---

## What it does

Upload a Google Pay or PhonePe PDF statement, enter your monthly income, and the application will:

- Extract and parse all transactions from the PDF
- Categorise each transaction into Food, Travel, Shopping, Entertainment, Health, Bills, Groceries, Education, Transfer, or Investment
- Calculate your overall financial risk level — Low, Medium, or High — based on spending ratios
- Identify overspent categories with specific excess amounts
- Generate personalised AI suggestions using Retrieval-Augmented Generation
- Produce a downloadable PDF report with charts, risk assessment, and transaction details

---

## Architecture

The pipeline is orchestrated using LangGraph with five sequential agent nodes:

```
PDF Upload
    |
Parse Agent         — extracts transactions using pdfplumber
    |
Categorize Agent    — DistilBERT (fine-tuned) with Groq LLaMA fallback
    |
Risk Agent          — calculates spend ratios and flags overspent categories
    |
RAG Suggestion Agent — FAISS retrieval + Groq LLaMA generation
    |
Report Agent        — assembles Streamlit dashboard and PDF export
```

The categorizer uses a hybrid approach: a fine-tuned DistilBERT model handles known merchant names with high confidence (threshold 0.75), and falls back to Groq LLaMA for ambiguous or garbled local names — a common issue with Indian UPI PDF exports.

---

## Tech Stack

| Layer | Technology |
|---|---|
| PDF parsing | pdfplumber |
| ML classifier | DistilBERT fine-tuned on Indian UPI transactions |
| Model hosting | HuggingFace Hub |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector store | FAISS (in-memory, per session) |
| LLM | Groq — LLaMA 3.3 70B |
| Agent framework | LangGraph |
| Frontend | Streamlit |
| Charts | Plotly |
| PDF export | ReportLab |

All services used are free tier. No database or persistent storage — each session is stateless.

---

## ML Model

The transaction categorizer is a DistilBERT model fine-tuned on a custom dataset of Indian UPI payment descriptions.

- Base model: `distilbert-base-uncased`
- Categories: 10 (Food, Travel, Shopping, Entertainment, Health, Bills, Groceries, Education, Transfer, Investment)
- Training samples: ~1000 (augmented from ~200 base examples with garbling simulation)
- Final accuracy: 98.9%
- Final F1 score: 98.9%
- Hosted at: `huggingface.co/ankitpadhi04/finsight-categorizer`

The dataset was augmented to simulate the garbled text that GPay and PhonePe PDFs produce due to overlapping text layers in their PDF rendering.

---

## Project Structure

```
finsight-ai/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── packages.txt            # System dependencies for Streamlit Cloud
├── .streamlit/
│   └── config.toml
├── src/
│   ├── parser.py           # PDF parsing and transaction extraction
│   ├── categorizer.py      # Hybrid DistilBERT + Groq categorizer
│   ├── risk.py             # Risk scoring logic
│   ├── rag.py              # FAISS vector store and RAG pipeline
│   ├── agents.py           # LangGraph agent graph
│   └── report.py           # PDF report generation
├── model/
│   └── train.py            # DistilBERT fine-tuning script
├── data/
│   ├── transactions.csv        # Base labeled dataset
│   ├── expand_dataset.py       # Dataset augmentation script
│   └── transactions_augmented.csv
└── knowledge/
    └── finance_tips.txt    # RAG knowledge base
```

---

## Local Setup

**Requirements:** Python 3.10+

```bash
# Clone the repository
git clone https://github.com/your-username/finsight-ai.git
cd finsight-ai

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Add environment variables
# Create a .env file with:
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token

# Run the app
streamlit run app.py
```

---

## Environment Variables

| Variable | Description | Where to get it |
|---|---|---|
| `GROQ_API_KEY` | Groq API key for LLaMA inference | console.groq.com |
| `HF_TOKEN` | HuggingFace token for model access | huggingface.co/settings/tokens |

For local development, store these in a `.env` file.
For Streamlit Cloud deployment, add them under Advanced Settings > Secrets.

---

## Supported PDF Formats

Currently tested with:

- Google Pay monthly transaction statements
- PhonePe transaction history exports (parser in progress)

The parser handles garbled text produced by overlapping PDF text layers, which is a known issue with Indian payment app PDF exports. Transactions are extracted using regex pattern matching on the consistent (though garbled) structure rather than relying on clean text.

---

## Limitations

- Each session is stateless — uploaded data is not stored between sessions
- PDF parsing accuracy depends on the statement format; heavily customised or scanned PDFs may not parse correctly
- The DistilBERT model was trained on a synthetic augmented dataset and may misclassify uncommon local merchant names
- Groq free tier has rate limits; analysing very large statements (200+ transactions) may hit limits

---

## Training the Model

To retrain the categorizer on new data:

```bash
# Expand the dataset
python data/expand_dataset.py

# Fine-tune DistilBERT
python model/train.py

# Push to HuggingFace Hub
python model/push_to_hub.py
```

Training takes approximately 10-15 minutes on CPU or 2-3 minutes on a GPU.

---

## Acknowledgements

- [Groq](https://groq.com) for fast free LLaMA inference
- [HuggingFace](https://huggingface.co) for model hosting and transformers library
- [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- [Streamlit](https://streamlit.io) for free cloud deployment

---
