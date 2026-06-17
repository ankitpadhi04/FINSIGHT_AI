# src/rag.py

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

load_dotenv()

KNOWLEDGE_PATH = "knowledge/finance_tips.txt"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

_embedder   = None
_index      = None
_chunks     = None
_groq       = None


def load_knowledge():
    """Load and embed finance tips into FAISS index."""
    global _embedder, _index, _chunks

    if _index is not None:
        return

    print("Loading RAG knowledge base...")
    _embedder = SentenceTransformer(EMBEDDING_MODEL)

    with open(KNOWLEDGE_PATH, "r") as f:
        _chunks = [line.strip() for line in f if line.strip()]

    embeddings = _embedder.encode(_chunks, convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    _index = faiss.IndexFlatIP(embeddings.shape[1])
    _index.add(embeddings.astype(np.float32))
    print(f"RAG ready — {len(_chunks)} tips indexed.")


def retrieve(query: str, top_k: int = TOP_K) -> list:
    """Retrieve top_k most relevant tips for a query."""
    load_knowledge()
    query_vec = _embedder.encode([query], convert_to_numpy=True)
    query_vec = query_vec / np.linalg.norm(query_vec)
    _, indices = _index.search(query_vec.astype(np.float32), top_k)
    return [_chunks[i] for i in indices[0] if i < len(_chunks)]


def get_groq():
    global _groq
    if _groq is None:
        _groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq


def generate_suggestions(risk_report: dict) -> str:
    """
    Takes risk report and generates personalised AI suggestions
    using RAG retrieved tips as context.
    """
    load_knowledge()

    # Build a query from the risk report
    overspent_cats = [o["category"] for o in risk_report["overspent"]]
    query = f"{risk_report['risk_level']} risk spending. Overspent on {', '.join(overspent_cats) if overspent_cats else 'general expenses'}."

    tips = retrieve(query)
    context = "\n".join(f"- {t}" for t in tips)

    prompt = f"""You are a personal finance advisor for Indian users.
Based on the user's spending analysis and financial tips below, give specific actionable suggestions.

User's financial summary:
- Monthly income: ₹{risk_report['monthly_income']}
- Total spent: ₹{risk_report['total_spent']}
- Risk level: {risk_report['risk_level']} ({risk_report['risk_score']}% of income spent)
- Savings potential: ₹{risk_report['savings_potential']}
- Overspent categories: {', '.join([f"{o['category']} ({o['actual_pct']}% vs {o['limit_pct']}% limit)" for o in risk_report['overspent']]) if risk_report['overspent'] else 'None'}

Relevant financial tips:
{context}

Give 4 to 5 specific, actionable suggestions tailored to this user's spending pattern.
Be direct, practical, and mention specific amounts where possible.
Format as a numbered list."""

    client = get_groq()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()