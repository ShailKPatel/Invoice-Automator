import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Flowchart", layout="centered", page_icon="↕")
st.title("Data extraction flowchart")

html = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <style>
    body { 
      font-family: Inter, Arial, Helvetica, sans-serif; 
      background:#f6f8fb; 
      padding:18px 18px 4px 18px; /* reduced bottom padding */
      margin:0;
    }
    .canvas { 
      background: white; 
      border-radius:10px; 
      box-shadow: 0 6px 18px rgba(20,30,60,0.08); 
      padding:12px 12px 4px 12px; /* reduced bottom padding */
      margin-bottom:0;
    }
    svg { 
      width:100%; 
      height:82vh; /* slightly shorter for less vertical scroll */
      overflow: visible; 
      display:block;
    }
    .process { fill:#ffffff; stroke:#2b4158; stroke-width:2; rx:8; }
    .data { fill:#fff9e6; stroke:#b07b00; stroke-width:2; }
    .decision { fill:#fff; stroke:#b03030; stroke-width:2; }
    .parallelogram { fill:#e8f6ff; stroke:#0b5fa5; stroke-width:2; }
    text { font-size:13px; fill:#0b1630; }
    .label { font-size:12px; fill:#16324a; font-weight:600; }
    .small { font-size:11px; fill:#2b4158; }
    .arrowtext { font-size:13px; fill:#2b4158; font-weight:500; }
    .marker { fill:#2b4158; }
    .note { font-size:12px; fill:#4a5a6a; }
    .footer { margin-top:6px; color:#4a5a6a; font-size:13px; }
  </style>
</head>
<body>
  <div class="canvas">
    <svg viewBox="0 0 1000 950" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Data extraction flowchart">
      <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="10" refY="5" orient="auto" markerUnits="strokeWidth">
          <path d="M0 0 L10 5 L0 10 z" class="marker" />
        </marker>
      </defs>

      <!-- Flowchart content unchanged -->
      <g>
        <polygon class="parallelogram" points="200,80 780,80 720,140 140,140" />
        <text x="470" y="112" text-anchor="middle" font-size="16" font-weight="700">Data input: PDFs and Images</text>
      </g>

      <g>
        <rect class="process" x="90" y="220" width="260" height="70" rx="8" />
        <text x="220" y="255" text-anchor="middle" class="label">PDF Text Extraction</text>
        <text x="220" y="275" text-anchor="middle" class="small">(native PDF parsing)</text>
      </g>

      <g>
        <rect class="process" x="650" y="220" width="260" height="70" rx="8" />
        <text x="780" y="255" text-anchor="middle" class="label">Images</text>
        <text x="780" y="275" text-anchor="middle" class="small">EasyOCR</text>
      </g>

      <path d="M240 140 L160 220" stroke="#2b4158" stroke-width="2.2" fill="none" marker-end="url(#arrow)"/>
      <text x="205" y="210" text-anchor="middle" class="arrowtext">PDF</text>

      <path d="M700 140 L780 220" stroke="#2b4158" stroke-width="2.2" fill="none" marker-end="url(#arrow)"/>
      <text x="730" y="210" text-anchor="middle" class="arrowtext">Image</text>

      <path d="M350 255 C470 260 600 260 650 255" stroke="#b03030" stroke-width="2" fill="none" marker-end="url(#arrow)"/>
      <text x="500" y="240" text-anchor="middle" class="arrowtext">no data found</text>

      <g>
        <rect class="process" x="320" y="360" width="360" height="80" rx="8" />
        <text x="500" y="395" text-anchor="middle" class="label">Store file path and extracted text</text>
        <text x="500" y="415" text-anchor="middle" class="small">(save original file path plus extracted text)</text>
      </g>

      <path d="M220 290 L220 390 L320 390" stroke="#2b4158" stroke-width="2.2" fill="none" marker-end="url(#arrow)"/>
      <text x="270" y="330" text-anchor="middle" class="arrowtext">extracted text</text>

      <path d="M760 290 L760 390 L680 390" stroke="#2b4158" stroke-width="2.2" fill="none" marker-end="url(#arrow)"/>
      <text x="670" y="330" text-anchor="middle" class="arrowtext">extracted text (EasyOCR)</text>

      <path d="M500 440 L500 520" stroke="#2b4158" stroke-width="2.2" fill="none" marker-end="url(#arrow)"/>
      <g>
        <rect class="process" x="370" y="520" width="260" height="64" rx="8" />
        <text x="500" y="553" text-anchor="middle" class="label">Regex correction</text>
        <text x="500" y="571" text-anchor="middle" class="small">(normalize, extract fields)</text>
      </g>

      <path d="M500 584 L500 660" stroke="#2b4158" stroke-width="2.2" fill="none" marker-end="url(#arrow)"/>
      <g>
        <rect class="process" x="340" y="660" width="320" height="70" rx="8" />
        <text x="500" y="695" text-anchor="middle" class="label">LLM correction (text only)</text>
        <text x="500" y="713" text-anchor="middle" class="small">(Mistral 7B)</text>
      </g>

      <path d="M500 730 L500 830" stroke="#2b4158" stroke-width="2.2" fill="none" marker-end="url(#arrow)"/>
      <g>
        <rect class="process" x="360" y="830" width="280" height="70" rx="8" />
        <text x="500" y="865" text-anchor="middle" class="label">Store CSV</text>
        <text x="500" y="883" text-anchor="middle" class="small">(final structured output)</text>
      </g>

      <g transform="translate(770,654)"> 
        <polygon class="decision" points="90,0 170,46 90,92 10,46" /> 
        <text x="90" y="48" text-anchor="middle" class="label">Low confidence?</text>
      </g>

      <path d="M660 700 L780 700" stroke="#b07b00" stroke-width="2.2" fill="none" marker-end="url(#arrow)"/>
      <text x="730" y="680" text-anchor="middle" class="arrowtext">low confidence score</text>

      <g>
        <rect class="process" x="750" y="830" width="220" height="72" rx="8" />
        <text x="860" y="860" text-anchor="middle" class="label">LLM correction (vision only)</text>
        <text x="860" y="880" text-anchor="middle" class="small">(use multimodal model to re-extract)</text>
      </g>

      <path d="M860 745 L860 830" stroke="#2b4158" stroke-width="2.2" fill="none" marker-end="url(#arrow)"/>
      <text x="865" y="775" class="arrowtext">if fails / needs vision</text>

      <path d="M750 870 640 870" stroke="#2b4158" stroke-width="2.2" fill="none" marker-end="url(#arrow)"/>
      <text x="695" y="855" text-anchor="middle" class="arrowtext">recovered fields</text>
    </svg>
  </div>
</body>
</html>
"""

components.html(html, height=750, scrolling=False)
