"""Streamlit interface for the GenAI RAG Evaluator.

    streamlit run app.py
"""
from __future__ import annotations

import os
import time

import pandas as pd
import streamlit as st

from src.config import SEARCH_TYPES, SUPPORTED_EXTENSIONS, ConfigError, get_settings, has_api_key
from src.evaluate import METRICS
from src.generator import stream_answer
from src.history import clear_history, load_history, log_run
from src.ingest import ingest_directory, ingest_paths, save_upload
from src.pipeline import RAGResult, evaluate_result, parse_test_set, run_batch
from src.retriever import delete_source, get_vectorstore, list_sources, reset_vectorstore, retrieve

METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
}
METRIC_HELP = {
    "faithfulness": "Share of claims in the answer that are supported by the retrieved context. "
    "Low values indicate hallucination.",
    "answer_relevancy": "How directly the answer addresses the question.",
    "context_precision": "Whether the relevant chunks are ranked at the top of the retrieved context.",
}

st.set_page_config(page_title="GenAI RAG & Evaluator", page_icon="🤖", layout="wide")

try:
    settings = get_settings()
except ConfigError as exc:
    st.error(f"Configuration error: {exc}")
    st.stop()


def fmt(score) -> str:
    return "n/a" if score is None or pd.isna(score) else f"{score:.2f}"


def show_metrics(scores: dict) -> None:
    for column, metric in zip(st.columns(len(METRICS)), METRICS):
        column.metric(METRIC_LABELS[metric], fmt(scores.get(metric)), help=METRIC_HELP[metric])
    if any(fmt(scores.get(metric)) == "n/a" for metric in METRICS):
        st.caption("n/a = the metric could not be computed for this answer.")


def show_reports(reports) -> None:
    for report in reports:
        note = f" — {report.detail}" if report.detail else ""
        text = f"**{report.source}**: {report.status}, {report.chunks} chunks{note}"
        if report.status == "error":
            st.error(text)
        elif report.status in ("empty", "skipped"):
            st.warning(text)
        else:
            st.success(text)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    if not has_api_key():
        key = st.text_input(
            "OpenAI API key", type="password",
            help="Used for this session only. Set OPENAI_API_KEY in .env to make it permanent.",
        )
        if key.strip():
            os.environ["OPENAI_API_KEY"] = key.strip()
            st.rerun()
    else:
        st.success("OpenAI API key loaded")

    top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, min(settings.top_k, 10))
    search_type = st.selectbox(
        "Retrieval strategy", SEARCH_TYPES, index=SEARCH_TYPES.index(settings.search_type),
        help="similarity = closest chunks; mmr = closest chunks with more variety.",
    )
    auto_evaluate = st.checkbox("Evaluate every answer automatically", value=False)
    st.divider()
    st.caption(
        f"Chat model: `{settings.chat_model}`  \nEvaluation model: `{settings.eval_model}`  \n"
        f"Embeddings: `{settings.embedding_model}`  \n"
        f"Chunking: {settings.chunk_size} / {settings.chunk_overlap} overlap"
    )

st.title("🤖 GenAI RAG Pipeline with Automated RAGAS Evaluation")
st.write(
    "Interact with your custom knowledge base and view hallucination & relevance metrics."
)

if not has_api_key():
    st.info("Enter your OpenAI API key in the sidebar (or add it to `.env`) to get started.")
    st.stop()

try:
    vectorstore = get_vectorstore(settings)
    sources = list_sources(settings, vectorstore)
except Exception as exc:
    st.error(f"Could not open the vector database: {exc}")
    st.stop()

tab_ask, tab_docs, tab_batch, tab_history = st.tabs(
    ["💬 Ask", "📚 Documents", "🧪 Batch evaluation", "📈 History"]
)

# ---------------------------------------------------------------- ask
with tab_ask:
    if not sources:
        st.info("No documents indexed yet. Add some in the **Documents** tab.")

    with st.form("ask"):
        question = st.text_input("Ask a question about your ingested documents:")
        ground_truth = st.text_area(
            "Reference answer (optional)", height=70,
            help="If you know the correct answer, context precision is measured against it.",
        )
        submitted = st.form_submit_button("Ask", type="primary", disabled=not sources)

    if submitted and question.strip():
        answered = False
        try:
            started = time.perf_counter()
            with st.spinner("Searching your documents..."):
                chunks = retrieve(question, top_k, search_type, settings, vectorstore)
            st.subheader("💡 Generated Answer")
            answer = st.write_stream(stream_answer(question, chunks, settings))
            result = RAGResult(
                question=question.strip(),
                answer=str(answer).strip(),
                contexts=chunks,
                ground_truth=ground_truth.strip(),
                latency_s=round(time.perf_counter() - started, 2),
            )
            if auto_evaluate and chunks:
                with st.spinner("Calculating Faithfulness & Relevance..."):
                    evaluate_result(result, settings)
            log_run(settings.history_file, result.to_dict())
            st.session_state["result"] = result
            answered = True
        except Exception as exc:
            st.error(f"Something went wrong while answering: {exc}")
        if answered:
            st.rerun()
    elif submitted:
        st.warning("Please enter a question.")

    result: RAGResult | None = st.session_state.get("result")
    if result:
        left, right = st.columns([3, 2])
        with left:
            st.subheader("💡 Generated Answer")
            st.caption(f"Q: {result.question}  ·  {result.latency_s}s")
            st.write(result.answer)

            st.subheader("📚 Retrieved Context Chunks")
            if not result.contexts:
                st.write("No matching chunks were found.")
            for number, chunk in enumerate(result.contexts, start=1):
                score = f" · relevance {chunk.score:.2f}" if chunk.score is not None else ""
                with st.expander(f"[{number}] {chunk.label}{score}"):
                    st.write(chunk.content)
        with right:
            st.subheader("📊 RAGAS Evaluation Metrics")
            if result.scores:
                show_metrics(result.scores)
            elif not result.contexts:
                st.write("Nothing to evaluate — no context was retrieved.")
            elif st.button("Run Evaluation Metrics"):
                try:
                    with st.spinner("Calculating Faithfulness & Relevance..."):
                        evaluate_result(result, settings)
                    log_run(settings.history_file, result.to_dict())
                except Exception as exc:
                    result.scores = None
                    st.error(f"Evaluation failed: {exc}")
                if result.scores:
                    st.rerun()
            else:
                st.caption("Scores range from 0 (poor) to 1 (excellent).")

# ---------------------------------------------------------------- documents
with tab_docs:
    st.subheader("Add documents")
    uploads = st.file_uploader(
        "Upload PDF, TXT or Markdown files",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )
    col_upload, col_folder = st.columns(2)
    index_uploads = col_upload.button("Index uploaded files", type="primary", disabled=not uploads)
    index_folder = col_folder.button(f"Index everything in {settings.data_dir.name}/")
    if index_uploads or index_folder:
        reports = None
        try:
            with st.spinner("Chunking and embedding..."):
                if index_uploads:
                    paths = [save_upload(f.name, f.getvalue(), settings) for f in uploads]
                    reports = ingest_paths(paths, settings, vectorstore)
                else:
                    reports = ingest_directory(None, settings, vectorstore)
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")
        if reports:
            st.session_state["reports"] = reports
            st.rerun()
        elif reports is not None:
            st.warning(f"No supported files found in `{settings.data_dir}`.")
    if st.session_state.get("reports"):
        show_reports(st.session_state.pop("reports"))

    st.subheader("Indexed documents")
    if not sources:
        st.write("Nothing indexed yet.")
    else:
        st.caption(f"{len(sources)} document(s), {sum(sources.values())} chunks")
        for name, count in sources.items():
            col_name, col_count, col_delete = st.columns([6, 2, 2])
            col_name.write(f"📄 {name}")
            col_count.write(f"{count} chunks")
            if col_delete.button("Remove", key=f"delete-{name}"):
                delete_source(name, settings, vectorstore)
                st.rerun()
        with st.expander("Danger zone"):
            st.write("Removes every document from the vector database. Files in the data folder are kept.")
            if st.checkbox("I understand") and st.button("Clear vector database"):
                reset_vectorstore(settings, vectorstore)
                st.session_state.pop("result", None)
                st.rerun()

# ---------------------------------------------------------------- batch
with tab_batch:
    st.write(
        "Upload a CSV test set to benchmark the pipeline. It needs a `question` column; "
        "an optional `ground_truth` column holds reference answers."
    )
    test_file = st.file_uploader("Test set (CSV)", type=["csv"], key="testset")
    if test_file is not None:
        try:
            rows = parse_test_set(pd.read_csv(test_file))
        except Exception as exc:
            rows = []
            st.error(f"Could not read the test set: {exc}")
        if rows:
            st.caption(f"{len(rows)} question(s) loaded.")
            if st.button("Run batch evaluation", type="primary", disabled=not sources):
                bar = st.progress(0.0, text="Answering questions...")
                try:
                    table = run_batch(
                        rows, top_k, search_type, settings, vectorstore,
                        progress=lambda done, total: bar.progress(
                            done / total, text=f"Answered {done}/{total} — scoring comes next"
                        ),
                    )
                    st.session_state["batch"] = table
                except Exception as exc:
                    st.error(f"Batch evaluation failed: {exc}")
                bar.empty()
            if not sources:
                st.info("Index at least one document first.")

    batch: pd.DataFrame | None = st.session_state.get("batch")
    if batch is not None:
        st.subheader("Results")
        show_metrics({m: pd.to_numeric(batch[m], errors="coerce").mean() for m in METRICS})
        failed = int((batch["error"] != "").sum())
        if failed:
            st.warning(f"{failed} question(s) failed — see the `error` column.")
        st.dataframe(batch)
        st.download_button(
            "Download results (CSV)", batch.to_csv(index=False).encode("utf-8"),
            file_name="rag_evaluation_results.csv", mime="text/csv",
        )

# ---------------------------------------------------------------- history
with tab_history:
    history = load_history(settings.history_file)
    if history.empty:
        st.write("No questions asked yet.")
    else:
        scored = history.dropna(subset=list(METRICS), how="all")
        st.caption(f"{len(history)} run(s), {len(scored)} evaluated")
        if not scored.empty:
            st.subheader("Average scores")
            show_metrics({metric: scored[metric].mean() for metric in METRICS})
            if len(scored) > 1:
                st.subheader("Scores over time")
                st.line_chart(scored.reset_index(drop=True)[list(METRICS)])
        st.subheader("All runs")
        st.dataframe(history.iloc[::-1])
        col_download, col_clear = st.columns(2)
        col_download.download_button(
            "Download history (CSV)", history.to_csv(index=False).encode("utf-8"),
            file_name="rag_history.csv", mime="text/csv",
        )
        if col_clear.button("Clear history"):
            clear_history(settings.history_file)
            st.rerun()
