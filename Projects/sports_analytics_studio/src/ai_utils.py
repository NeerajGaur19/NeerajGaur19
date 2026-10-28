from __future__ import annotations
import os
from pathlib import Path

SPORTS_KNOWLEDGE = [
    "Athlete performance analysis compares workload, speed, endurance, recovery, and context over time.",
    "Injury risk classification needs a clearly defined historical target and careful validation; it is not a medical diagnosis.",
    "Talent identification should account for age, role, maturation, competition level, opportunity, and position-specific requirements.",
    "Event analytics can include possession, shots, passes, fouls, expected goals, player minutes, and match outcomes.",
    "Anomaly detection identifies unusual observations; an anomaly is not automatically an error or a meaningful event.",
    "Missing values should be investigated before removal. Missingness can carry information, and dropping rows can introduce bias.",
    "Duplicate columns should be removed only after verifying they are genuinely redundant. Duplicate rows may represent repeated events.",
    "Avoid data leakage: do not include information that would only be known after the prediction time.",
    "Workload indicators should be interpreted in sport and athlete context; acute/chronic workload ratios alone should not be treated as definitive injury predictors.",
    "Model performance should be assessed with appropriate validation, class balance, calibration, and relevant metrics rather than accuracy alone.",
    "A talent ranking is not objective unless metrics, weights, sport, position, age and competition context are defined.",
]

def _embedding_function():
    """Use the requested Sentence Transformers model through LangChain's HuggingFaceEmbeddings."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def retrieve_sports_notes(query: str, persist_directory: str = "chroma_db") -> list[str]:
    """Retrieve relevant sports-analytics guidance from a persistent Chroma collection."""
    try:
        import chromadb
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        persist = str(Path(persist_directory))
        vectorstore = Chroma(
            collection_name="sports_analytics_notes",
            embedding_function=embeddings,
            persist_directory=persist,
        )
        # Populate the collection once. Chroma's persistent directory keeps these notes between runs.
        if vectorstore._collection.count() == 0:
            vectorstore.add_texts(
                texts=SPORTS_KNOWLEDGE,
                ids=[f"sports-note-{i}" for i in range(len(SPORTS_KNOWLEDGE))]
            )
        docs = vectorstore.similarity_search(query, k=min(4, len(SPORTS_KNOWLEDGE)))
        return [d.page_content for d in docs]
    except Exception as e:
        # Keep the rest of the application available if local embedding dependencies are missing.
        words = set(query.lower().split())
        ranked = sorted(SPORTS_KNOWLEDGE, key=lambda s: len(words & set(s.lower().split())), reverse=True)
        return ranked[:3]

def ask_gemini(question: str, dataset_summary: str = "") -> str:
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not configured. Add it to your .env file and restart Streamlit.")
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SystemMessage
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.2)
        system = (
            "You are a cautious sports data analytics assistant. Use retrieved context where relevant. Explain assumptions, "
            "distinguish correlation from causation, protect athlete privacy, and never present model outputs as medical diagnoses. "
            "If data is synthetic or incomplete, say so. Do not invent computed metrics."
        )
        prompt = f"Dataset context:\n{dataset_summary[:7000]}\n\nUser question:\n{question}"
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        return str(response.content)
    except Exception as e:
        raise RuntimeError(f"Gemini request failed: {e}") from e
