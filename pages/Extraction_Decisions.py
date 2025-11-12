import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Extraction Decisions & Trade-offs", layout="centered")

st.title("Extraction Decisions & Trade-offs")

flowchart_path = Path("data/images/flowchart.png")
if flowchart_path.exists():
    st.image(str(flowchart_path), caption="Pipeline flowchart", use_container_width=True)
else:
    st.warning(f"Flowchart not found at {flowchart_path}. Place flowchart.png at this path to display it.")

st.markdown("""
This panel explains the rationale behind the extraction pipeline and highlights trade-offs so reviewers can focus on *engineering judgement* rather than implementation details.
""")

st.header("Design goals")
st.markdown("""
- Accurate extraction of invoice-level fields and line items.
- Minimal and targeted use of expensive models (LLMs / vision-LLMs).
- Clear fallbacks and auditable outputs for manual review.
- Scalable choices that reduce unnecessary compute.
""")

st.header("Key decisions and why")

st.subheader("1) Split PDF vs Image processing")
st.markdown("""
- PDFs often contain a native text layer; extracting that is far cheaper and more reliable than OCR.
- Running OCR on text-backed PDFs wastes tokens and time.
- Some PDFs are scanned images; therefore the pipeline checks text length and falls back to rendering+OCR only when needed.
- Benefit: preserves accuracy while cutting compute and cost.
""")

st.subheader("2) OCR: EasyOCR (lightweight)")
st.markdown("""
- EasyOCR provides a good trade-off between speed and quality for initial extraction.
- The LLM correction layer is designed to handle minor OCR noise, so maximal OCR precision is unnecessary at the first pass.
- This saves GPU/token cost and simplifies local runs.
""")

st.subheader("3) Regex-first normalization")
st.markdown("""
- Deterministic regex rules extract obvious values (dates, amounts, invoice IDs) quickly and cheaply.
- Regex also provides signal strength for confidence scoring (e.g., exact currency matches increase confidence).
- Regex is brittle on layout changes, so it is used as a first-pass filter, not the final arbiter.
""")

st.subheader("4) LLM correction (text) with structured JSON + confidence")
st.markdown("""
- A text-LLM receives OCR + regex hints and returns a strict JSON with per-field confidence.
- Use cases: semantic normalization (vendor name cleaning), resolving ambiguous dates, and fixing OCR errors in numbers.
- The LLM's confidence score gates whether to accept results or escalate to a heavier vision-LLM.
""")

st.subheader("5) Vision-LLM fallback")
st.markdown("""
- Triggered only when the text-LLM reports low confidence or when line-item table parsing fails.
- Vision-LLMs understand layout and tables, reducing missed or misaligned line items.
- Trade-off: higher compute and engineering complexity; justified for difficult or high-value invoices.
""")

st.header("Confidence scoring and gating")
st.markdown("""
- Confidence combines: regex match quality, LLM-reported confidence, and structural heuristics (presence of currency symbols, numeric parsing success).
- Thresholds:
  - Above `MIN_CONFIDENCE`: auto-accept.
  - Below `MIN_CONFIDENCE`: re-run vision-LLM.
- This minimizes unnecessary vision-LLM calls while keeping the error-rate low.
""")

st.header("Bonus: hybrid ensemble and evaluation")
st.markdown("""
- Hybrid ensemble: regex → text-LLM → vision-LLM fallback. Use ensembles only when they materially improve accuracy.
- Field-level evaluation: compute precision, recall, F1 on a labeled subset and expose per-field F1 to reveal weaknesses (vendor, dates, unit prices are typical failure points).
- Confidence scores enable easy batching of manual review and model retraining data collection.
""")

