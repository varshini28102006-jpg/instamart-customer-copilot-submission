"""policy_rag.py -- Policy chunking, TF-IDF retrieval, grounded QA, and evaluation."""

import os
import glob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm


def chunk_documents(policies_dir="data/policies", chunk_size=40, overlap=10):
    """Splits policy documents into word chunks with specified overlap."""
    policy_files = sorted(glob.glob(os.path.join(policies_dir, "*.txt")))
    chunks = []

    step = chunk_size - overlap if overlap > 0 else chunk_size

    for pf in policy_files:
        fname = os.path.basename(pf)
        with open(pf, "r", encoding="utf-8") as f:
            words = f.read().split()

        start = 0
        while start < len(words):
            chunk_words = words[start : start + chunk_size]
            chunks.append({
                "source": fname,
                "text": " ".join(chunk_words),
                "start_idx": start,
                "end_idx": start + len(chunk_words)
            })
            if start + chunk_size >= len(words):
                break
            start += step

    return chunks


class PolicyRetriever:
    """TF-IDF vectorizer and cosine similarity retriever."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform([c["text"] for c in chunks])

    def retrieve(self, query, k=3):
        """Retrieves top k most relevant chunks for a given query."""
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.X)[0]
        top_indices = sims.argsort()[::-1][:k]

        results = []
        for idx in top_indices:
            results.append({
                "source": self.chunks[idx]["source"],
                "score": float(sims[idx]),
                "text": self.chunks[idx]["text"]
            })
        return results


def answer_policy_question(question, retriever, k=3):
    """Retrieves top chunks and generates a grounded response using mock_llm."""
    retrieved = retriever.retrieve(question, k=k)

    context_blocks = []
    for r in retrieved:
        context_blocks.append(f"[{r['source']}]\n{r['text']}")
    context_str = "\n".join(context_blocks)

    messages = [
        {
            "role": "system",
            "content": "POLICY ASSISTANT. Answer only from the supplied context. "
                       "If the information is not in the context, say you could not find the information."
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{context_str}\n\nQUESTION:\n{question}"
        }
    ]

    resp = mock_llm.chat(messages)
    answer_text = resp.choices[0].message.content
    sources = [r["source"] for r in retrieved]
    return answer_text, sources, retrieved


def evaluate_policy_rag(qa_path="data/raw/policy_eval.csv", overlap=10):
    """Evaluates retrieval accuracy, fact match accuracy, and unanswerable fallback."""
    chunks = chunk_documents(overlap=overlap)
    retriever = PolicyRetriever(chunks)
    qa_df = pd.read_csv(qa_path)

    results = []
    doc_matches, fact_matches, fallback_matches = 0, 0, 0
    ans_count, unans_count = 0, 0

    for _, row in qa_df.iterrows():
        q = row["question"]
        exp_doc = row["expected_doc"]
        exp_fact = row["expected_fact"]

        ans_text, sources, retrieved = answer_policy_question(q, retriever, k=3)

        if exp_doc != "NONE":
            ans_count += 1
            doc_hit = exp_doc in sources
            fact_hit = exp_fact in ans_text
            if doc_hit:
                doc_matches += 1
            if fact_hit:
                fact_matches += 1
            results.append({
                "question": q,
                "expected_doc": exp_doc,
                "expected_fact": exp_fact,
                "doc_hit": doc_hit,
                "fact_hit": fact_hit,
                "answer": ans_text
            })
        else:
            unans_count += 1
            fallback_hit = ans_text.startswith("I could not find that information")
            if fallback_hit:
                fallback_matches += 1
            results.append({
                "question": q,
                "expected_doc": "NONE",
                "expected_fact": "",
                "doc_hit": True,
                "fact_hit": fallback_hit,
                "answer": ans_text
            })

    summary = {
        "overlap": overlap,
        "total_chunks": len(chunks),
        "doc_retrieval_acc": f"{doc_matches}/{ans_count}",
        "expected_fact_acc": f"{fact_matches}/{ans_count}",
        "fallback_acc": f"{fallback_matches}/{unans_count}",
        "results": results
    }

    return summary


if __name__ == "__main__":
    print("--- 10-Word Overlap Evaluation ---")
    s10 = evaluate_policy_rag(overlap=10)
    print(f"Total chunks: {s10['total_chunks']}")
    print(f"Doc Retrieval: {s10['doc_retrieval_acc']}")
    print(f"Fact Retrieval: {s10['expected_fact_acc']}")
    print(f"Fallback Retrieval: {s10['fallback_acc']}")

    print("\n--- 0-Word Overlap Evaluation ---")
    s0 = evaluate_policy_rag(overlap=0)
    print(f"Total chunks: {s0['total_chunks']}")
    print(f"Doc Retrieval: {s0['doc_retrieval_acc']}")
    print(f"Fact Retrieval: {s0['expected_fact_acc']}")
    print(f"Fallback Retrieval: {s0['fallback_acc']}")
