"""EvalPipeline Pro — minimalist observability dashboard (Streamlit).

Reads ONLY from the FastAPI backend (no direct DB access):
  GET /api/v1/evals/history      -> run table + score/latency trends
  GET /api/v1/evals/runs/{id}    -> per-case drill-down with judge reasoning
  POST /api/v1/evals/run         -> optional "Run evaluation now" button
  POST /api/v1/rag/query         -> live RAG playground

Run:  streamlit run dashboard.py          (API_BASE_URL env, default http://localhost:8000)
"""

import os
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")  # TODO(hosting)
METRICS = {
    "avg_faithfulness": "Faithfulness",
    "avg_relevancy": "Answer relevancy",
    "avg_contextual_precision": "Contextual precision",
}

st.set_page_config(page_title="EvalPipeline Pro", page_icon="🧪", layout="wide")


# --------------------------------------------------------------------------- API helpers
def _get(path: str, params: Optional[Dict[str, Any]] = None, timeout: float = 30.0) -> Any:
    response = httpx.get(f"{API_BASE_URL}{path}", params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _post(path: str, payload: Dict[str, Any], timeout: float = 600.0) -> Any:
    response = httpx.post(f"{API_BASE_URL}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=15, show_spinner=False)
def load_history(limit: int) -> List[Dict[str, Any]]:
    return _get("/api/v1/evals/history", params={"limit": limit})


@st.cache_data(ttl=60, show_spinner=False)
def load_run(run_id: int) -> Dict[str, Any]:
    return _get(f"/api/v1/evals/runs/{run_id}")


def health() -> Optional[Dict[str, Any]]:
    try:
        return _get("/health", timeout=5.0)
    except Exception:  # noqa: BLE001 - shown as "API down" in the sidebar
        return None


def fmt(value: Optional[float], digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🧪 EvalPipeline Pro")
    st.caption(f"API: `{API_BASE_URL}`")
    info = health()
    if info is None:
        st.error("API unreachable")
        st.stop()
    st.success(f"API v{info.get('version')} · {info.get('environment')}")
    if not info.get("llm_configured", False):
        st.warning("LLM_API_KEY not configured — evals and RAG queries will fail")  # TODO(apikey)

    history_limit = st.slider("Runs to load", 5, 200, 50, step=5)
    if st.button("🔄 Refresh"):
        load_history.clear()
        load_run.clear()
        st.rerun()

    st.divider()
    st.subheader("Run evaluation now")
    threshold = st.number_input("Quality threshold", 0.0, 1.0, 0.85, step=0.05)
    prompt_version = st.selectbox("Prompt version", ["v1", "v0-naive"])
    if st.button("▶ Run full suite", type="primary", use_container_width=True):
        with st.spinner("Evaluating all golden cases (this takes 30-120 s)…"):
            try:
                result = _post(
                    "/api/v1/evals/run",
                    {"quality_threshold": threshold, "prompt_version": prompt_version},
                )
                load_history.clear()
                load_run.clear()
                st.success(f"Run #{result['run_id']}: **{result['overall_status']}**")
            except httpx.HTTPStatusError as exc:
                st.error(f"{exc.response.status_code}: {exc.response.text[:400]}")
            except httpx.HTTPError as exc:
                st.error(f"request failed: {exc}")

# --------------------------------------------------------------------------- data
try:
    history = load_history(history_limit)
except httpx.HTTPError as exc:
    st.error(f"Could not load history: {exc}")
    st.stop()

tab_overview, tab_runs, tab_playground = st.tabs(["📈 Overview", "🔍 Run details", "💬 RAG playground"])

# =========================================================================== overview
with tab_overview:
    if not history:
        st.info("No evaluation runs yet. Use **Run full suite** in the sidebar or call `POST /api/v1/evals/run`.")
    else:
        df = pd.DataFrame(history)
        df["created_at"] = pd.to_datetime(df["created_at"])
        df = df.sort_values("created_at")
        latest = df.iloc[-1]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Latest status", latest["overall_status"], f"run #{latest['run_id']}")
        c2.metric("Faithfulness", fmt(latest["avg_faithfulness"]))
        c3.metric("Answer relevancy", fmt(latest["avg_relevancy"]))
        c4.metric("Contextual precision", fmt(latest["avg_contextual_precision"]))
        c5.metric("Avg latency", f"{fmt(latest['avg_latency_ms'], 0)} ms")

        pass_rate = (df["overall_status"] == "PASS").mean() * 100
        st.caption(
            f"{len(df)} runs loaded · pass rate {pass_rate:.0f}% · "
            f"latest commit `{latest['commit_hash']}` · prompt `{latest['prompt_version']}`"
        )

        col_scores, col_latency = st.columns([3, 2])
        with col_scores:
            st.subheader("Quality scores per run")
            long = df.melt(
                id_vars=["run_id", "created_at", "commit_hash", "prompt_version", "overall_status"],
                value_vars=list(METRICS),
                var_name="metric",
                value_name="score",
            ).dropna(subset=["score"])
            long["metric"] = long["metric"].map(METRICS)
            fig = px.line(
                long, x="created_at", y="score", color="metric", markers=True,
                hover_data=["run_id", "commit_hash", "prompt_version", "overall_status"],
            )
            fig.add_hline(
                y=float(latest["quality_threshold"]), line_dash="dash", line_color="red",
                annotation_text=f"threshold {latest['quality_threshold']:.2f}",
            )
            fig.update_yaxes(range=[0, 1.02], title="score")
            fig.update_xaxes(title="")
            fig.update_layout(legend_title="", margin=dict(t=10, b=10), height=380)
            st.plotly_chart(fig, use_container_width=True)

        with col_latency:
            st.subheader("Latency trend")
            fig_l = go.Figure()
            fig_l.add_trace(go.Bar(
                x=df["created_at"], y=df["avg_latency_ms"],
                marker_color=["#2ecc71" if s == "PASS" else "#e74c3c" for s in df["overall_status"]],
                hovertext=[f"run #{r} · {c}" for r, c in zip(df["run_id"], df["commit_hash"])],
            ))
            fig_l.update_yaxes(title="avg end-to-end latency (ms)")
            fig_l.update_layout(margin=dict(t=10, b=10), height=380)
            st.plotly_chart(fig_l, use_container_width=True)
            st.caption("green = PASS, red = FAIL")

        st.subheader("Run history")
        table = df.sort_values("created_at", ascending=False)[[
            "run_id", "created_at", "overall_status", "commit_hash", "prompt_version",
            "avg_faithfulness", "avg_relevancy", "avg_contextual_precision", "avg_latency_ms",
            "passed_cases", "failed_cases", "errored_cases", "duration_seconds",
        ]]
        st.dataframe(
            table.style.map(
                lambda v: "color: #2ecc71; font-weight: bold" if v == "PASS"
                else ("color: #e74c3c; font-weight: bold" if v == "FAIL" else ""),
                subset=["overall_status"],
            ),
            use_container_width=True, hide_index=True,
        )

# =========================================================================== run details
with tab_runs:
    if not history:
        st.info("Nothing to inspect yet.")
    else:
        options = {
            f"#{r['run_id']} · {r['overall_status']} · {r['commit_hash']} · {r['prompt_version']} · "
            f"{pd.to_datetime(r['created_at']).strftime('%Y-%m-%d %H:%M')}": r["run_id"]
            for r in history
        }
        chosen = st.selectbox("Run", list(options))
        run = load_run(options[chosen])

        status_color = "green" if run["overall_status"] == "PASS" else "red"
        st.markdown(
            f"### Run #{run['run_id']} — :{status_color}[{run['overall_status']}]  "
            f"<small>generator `{run['generator_model']}` · judge `{run['judge_model']}` · "
            f"embeddings `{run['embedding_model']}` · threshold {run['quality_threshold']:.2f} · "
            f"{run['duration_seconds']:.1f}s</small>",
            unsafe_allow_html=True,
        )

        cases = run.get("cases", [])
        cdf = pd.DataFrame([{
            "case": c["case_key"], "category": c["category"], "passed": c["passed"],
            "faithfulness": c["faithfulness"], "relevancy": c["relevancy"],
            "contextual_precision": c["contextual_precision"], "latency_ms": c["latency_ms"],
            "error": c["error"],
        } for c in cases])

        only_failed = st.toggle("Show failed / errored cases only", value=True)
        view = cdf[~cdf["passed"]] if only_failed else cdf
        st.dataframe(view, use_container_width=True, hide_index=True)

        if not cdf.empty:
            st.subheader("Per-case scores")
            cl = cdf.melt(id_vars=["case", "category"],
                          value_vars=["faithfulness", "relevancy", "contextual_precision"],
                          var_name="metric", value_name="score").dropna()
            fig_c = px.bar(cl, x="case", y="score", color="metric", barmode="group",
                           hover_data=["category"])
            fig_c.add_hline(y=run["quality_threshold"], line_dash="dash", line_color="red")
            fig_c.update_yaxes(range=[0, 1.02])
            fig_c.update_layout(legend_title="", margin=dict(t=10, b=10), height=340)
            st.plotly_chart(fig_c, use_container_width=True)

        st.subheader("Judge reasoning")
        for c in (c for c in cases if (not c["passed"]) or not only_failed):
            icon = "✅" if c["passed"] else "❌"
            with st.expander(f"{icon} {c['case_key']} · {c['category']} · F={fmt(c['faithfulness'])} "
                             f"R={fmt(c['relevancy'])} CP={fmt(c['contextual_precision'])}"):
                st.markdown(f"**Query**  \n{c['query']}")
                left, right = st.columns(2)
                left.markdown("**Expected answer**")
                left.info(c["expected_answer"])
                right.markdown("**Actual answer**")
                (right.warning if not c["passed"] else right.success)(c["actual_answer"] or "_(no output)_")
                if c["error"]:
                    st.error(f"Error: {c['error']}")
                for name, m in (c.get("metrics") or {}).items():
                    if not m.get("applicable", True):
                        st.caption(f"**{name}** — not applicable: {m.get('reasoning')}")
                        continue
                    verdict = "✅" if m.get("passed") else "❌"
                    st.markdown(f"**{name}** {verdict} score={fmt(m.get('score'))}  \n{m.get('reasoning', '')}")
                    details = m.get("details") or {}
                    bad_claims = details.get("unsupported_claims")
                    if bad_claims:
                        st.markdown("Unsupported claims:\n" + "\n".join(f"- {x}" for x in bad_claims))
                    if details.get("relevance_by_rank") is not None:
                        st.caption(f"relevance by rank: {details['relevance_by_rank']}")
                with st.popover("Retrieved chunks"):
                    for ch in c.get("retrieved_chunks", []):
                        st.markdown(f"**#{ch['rank']}** job_id={ch['job_id']} · sim={ch['similarity']:.3f}")
                        st.code(ch["content"], language=None)

# =========================================================================== playground
with tab_playground:
    st.subheader("Query the RAG service")
    query = st.text_area("Candidate query", "Senior Python backend roles in Bengaluru with FastAPI", height=80)
    top_k = st.slider("top_k", 1, 10, 5)
    if st.button("Ask", type="primary"):
        with st.spinner("Retrieving + generating…"):
            try:
                res = _post("/api/v1/rag/query", {"query": query, "top_k": top_k}, timeout=120.0)
            except httpx.HTTPStatusError as exc:
                st.error(f"{exc.response.status_code}: {exc.response.text[:500]}")
                res = None
            except httpx.HTTPError as exc:
                st.error(f"request failed: {exc}")
                res = None
        if res:
            a = res["answer"]
            st.markdown(f"**Answer** (confidence: {a['confidence']}, answerable: {a['is_answerable']})")
            st.write(a["answer"])
            for rec in a["recommendations"]:
                st.markdown(f"- **{rec['title']}** at **{rec['company']}** (job_id={rec['job_id']}) — "
                            f"{rec['match_reason']}")
            if a["caveats"]:
                st.warning("Caveats: " + " ".join(a["caveats"]))
            st.caption(
                f"retrieval {res['retrieval_latency_ms']:.0f} ms · generation {res['generation_latency_ms']:.0f} ms · "
                f"total {res['latency_ms']:.0f} ms · model `{res['generator_model']}` · prompt `{res['prompt_version']}`"
            )
            with st.expander("Retrieved chunks"):
                for ch in res["retrieved_chunks"]:
                    st.markdown(f"**#{ch['rank']}** {ch['title']} @ {ch['company']} · sim={ch['similarity']:.3f}")
                    st.code(ch["content"], language=None)
