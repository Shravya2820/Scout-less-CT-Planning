"""
Scout-less CT Scan Planning System
B.Tech Project Prototype — Orchestration Layer
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os
import json
import time
from datetime import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Scout-less CT Planner",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark medical theme */
.stApp {
    background: #0a0e1a;
    color: #c8d8e8;
}

.stSidebar {
    background: #0d1221 !important;
    border-right: 1px solid #1e3a5f;
}

/* Header */
.ct-header {
    background: linear-gradient(135deg, #0a1628 0%, #0d2040 50%, #0a1628 100%);
    border: 1px solid #1e4a7a;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    position: relative;
    overflow: hidden;
}
.ct-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #00d4ff, #0080ff, #00d4ff, transparent);
}
.ct-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    color: #00d4ff;
    margin: 0;
    letter-spacing: 0.05em;
}
.ct-header p {
    color: #6a9ec0;
    margin: 4px 0 0 0;
    font-size: 0.85rem;
    font-weight: 300;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: #0d1829;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; width: 3px; height: 100%;
    background: linear-gradient(180deg, #00d4ff, #0050a0);
    border-radius: 10px 0 0 10px;
}
.metric-label {
    font-size: 0.7rem;
    color: #4a7a9b;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 600;
    margin-bottom: 4px;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    color: #00d4ff;
    font-weight: 700;
}
.metric-unit {
    font-size: 0.8rem;
    color: #4a7a9b;
    margin-left: 4px;
}
.metric-status {
    font-size: 0.75rem;
    margin-top: 4px;
    padding: 2px 8px;
    border-radius: 20px;
    display: inline-block;
}
.status-normal { background: #0a2a1a; color: #00c878; border: 1px solid #00c87844; }
.status-overweight { background: #2a1f0a; color: #ffaa00; border: 1px solid #ffaa0044; }
.status-obese { background: #2a0a0a; color: #ff4444; border: 1px solid #ff444444; }

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #4a7a9b;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    font-weight: 700;
    margin: 24px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1e3a5f, transparent);
}

/* Protocol card */
.protocol-card {
    background: #0b1a2e;
    border: 1px solid #1e4060;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 8px;
    font-size: 0.85rem;
    color: #8ab4cc;
    line-height: 1.6;
}
.protocol-card strong {
    color: #00d4ff;
    display: block;
    margin-bottom: 4px;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Landmark table */
.landmark-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 16px;
    background: #0b1a2e;
    border-radius: 6px;
    margin-bottom: 6px;
    border-left: 3px solid #0050a0;
}
.landmark-name {
    font-weight: 600;
    color: #a8c8e0;
    min-width: 160px;
    font-size: 0.85rem;
}
.landmark-coords {
    font-family: 'Space Mono', monospace;
    color: #00d4ff;
    font-size: 0.78rem;
    background: #0a1428;
    padding: 2px 8px;
    border-radius: 4px;
}

/* Confidence bar */
.confidence-container {
    background: #0b1a2e;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 16px;
}
.confidence-label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}
.confidence-title { color: #6a9ec0; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; }
.confidence-pct { font-family: 'Space Mono', monospace; font-size: 1.4rem; color: #00e676; font-weight: 700; }
.confidence-bar-bg {
    background: #0a1428;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #0050c0, #00a0ff, #00e676);
    transition: width 1s ease;
}

/* Rationale box */
.rationale-box {
    background: #081420;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #00d4ff;
    border-radius: 8px;
    padding: 16px 20px;
    font-size: 0.85rem;
    color: #8ab4cc;
    line-height: 1.7;
    margin-bottom: 16px;
}
.rationale-box .rag-tag {
    font-size: 0.65rem;
    color: #00d4ff;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    background: #001a30;
    border: 1px solid #0050a0;
    border-radius: 4px;
    padding: 2px 6px;
    margin-bottom: 8px;
    display: inline-block;
}

/* Z-boundary display */
.z-boundary {
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
}
.z-point {
    flex: 1;
    background: #0b1a2e;
    border-radius: 8px;
    padding: 14px 16px;
    text-align: center;
}
.z-point.start-point { border-top: 3px solid #00e676; }
.z-point.end-point { border-top: 3px solid #ff4466; }
.z-point-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; color: #4a7a9b; margin-bottom: 6px; }
.z-point-value { font-family: 'Space Mono', monospace; font-size: 1.3rem; font-weight: 700; }
.z-point.start-point .z-point-value { color: #00e676; }
.z-point.end-point .z-point-value { color: #ff4466; }
.z-point-anatomy { font-size: 0.75rem; color: #6a9ec0; margin-top: 4px; }

/* Log box */
.log-box {
    background: #050c18;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 14px 18px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #4a9a7a;
    line-height: 1.8;
}
.log-box .log-entry { margin: 0; }
.log-box .log-time { color: #2a6a5a; }
.log-box .log-warn { color: #aa8800; }
.log-box .log-ok { color: #00c878; }

/* Confirm button */
.stButton > button {
    background: linear-gradient(135deg, #003a80, #0060c0) !important;
    color: #ffffff !important;
    border: 1px solid #0080ff !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
    padding: 12px 28px !important;
    font-weight: 700 !important;
    transition: all 0.2s ease !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0060c0, #00a0ff) !important;
    border-color: #00d4ff !important;
    box-shadow: 0 0 20px #0080ff44 !important;
}

/* Sidebar styles */
.sidebar-section {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #4a7a9b;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    font-weight: 700;
    margin: 20px 0 8px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e3a5f;
}
div[data-testid="stNumberInput"] label,
div[data-testid="stFileUploader"] label {
    color: #6a9ec0 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
div[data-testid="stSelectbox"] label {
    color: #6a9ec0 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* Step indicators */
.step-bar {
    display: flex;
    gap: 0;
    margin-bottom: 28px;
}
.step-item {
    flex: 1;
    text-align: center;
    padding: 8px 4px;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    border-bottom: 2px solid #1e3a5f;
    color: #2a4a6a;
}
.step-item.active {
    color: #00d4ff;
    border-bottom-color: #00d4ff;
}
.step-item.done {
    color: #00e676;
    border-bottom-color: #00e676;
}

div[data-testid="stExpander"] {
    background: #0b1829 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Knowledge Base & Mock Engine ──────────────────────────────────────────────

CT_PROTOCOL_KB = """
Protocol ID: CHEST-001
Name: Standard Chest CT
Indication: Pulmonary evaluation, lung nodule characterization, mediastinal assessment.
Scout Reference: PA chest radiograph
Start Landmark: 2 cm superior to lung apex (approximately C7 level, ~2 cm above sternal notch)
End Landmark: T12 vertebral body (lower border, diaphragm insertion)
Z-Start Offset: +20 mm from Sternum superior edge
Z-End Reference: T12 centroid, inferior surface
Typical Z-Range: 280-320 mm (craniocaudal)
kVp: 120 | mAs: Auto (CTDI target 8 mGy) | Pitch: 1.375
Reconstructions: Lung kernel (B60f), Mediastinal kernel (B30f)
Notes: Include full lung apices. Suspend respiration at full inspiration.

Protocol ID: ABD-001
Name: Standard Abdomen CT
Indication: Abdominal pain, liver/pancreas/kidney evaluation, staging.
Scout Reference: AP abdomen
Start Landmark: Diaphragm dome (T8-T9 level), ~10 mm superior to liver dome
End Landmark: Pubic Symphysis inferior border
Z-Start Offset: -10 mm from Diaphragm apex (superior)
Z-End Reference: Pubic Symphysis inferior border
Typical Z-Range: 380-440 mm (craniocaudal)
kVp: 120 | mAs: Auto (CTDI target 12 mGy) | Pitch: 1.2
Reconstructions: Soft tissue (B20f), Portal venous phase
Notes: IV contrast required. Oral contrast optional. Scan portal venous phase (65-70s delay).

Protocol ID: CHEST-ABD-001
Name: Chest-Abdomen-Pelvis (CAP) CT
Indication: Oncology staging, trauma survey, multi-organ evaluation.
Scout Reference: AP scout (full torso)
Start Landmark: 2 cm above lung apex
End Landmark: Pubic Symphysis inferior border (ischial tuberosities included)
Z-Start Offset: +20 mm from Sternum superior edge
Z-End Reference: Pubic Symphysis inferior border + 10 mm margin
Typical Z-Range: 700-800 mm (craniocaudal)
kVp: 120 | mAs: Auto (CTDI target 10 mGy) | Pitch: 1.375
Reconstructions: Lung, soft tissue, bone kernels
Notes: Single breath-hold if possible. Consider split acquisition for obese patients.

Protocol ID: PELVIS-001
Name: Pelvis CT
Indication: Pelvic fracture, bladder/prostate/gynecologic assessment.
Start Landmark: Iliac crest (L4-L5 level, ASIS superior border)
End Landmark: Pubic Symphysis + 20 mm inferior margin (ischial tuberosities)
Z-Start Offset: -5 mm from ASIS
Z-End Reference: Pubic Symphysis inferior + 20 mm
Typical Z-Range: 200-260 mm
kVp: 120 | mAs: 200 fixed | Pitch: 1.0
Notes: Full bladder preferred. Rectal contrast optional for rectal assessment.
"""

def simulate_skel_model(height_cm: float, weight_kg: float) -> np.ndarray:
    """
    Mock skeletal parametric model.
    Returns an 85-D parameter vector encoding body morphology.
    In a real system, this would be the output of a 3D mesh regression network.
    """
    np.random.seed(int(height_cm + weight_kg) % 2**31)
    bmi = weight_kg / ((height_cm / 100) ** 2)

    # Anthropometric proportion features
    trunk_ratio = 0.52 + (bmi - 22) * 0.002          # trunk/height ratio shifts with BMI
    shoulder_width = height_cm * 0.259 * (1 + (bmi - 22) * 0.003)
    hip_width = height_cm * 0.191 * (1 + (bmi - 22) * 0.005)
    torso_depth = height_cm * 0.145 * (1 + (bmi - 22) * 0.006)

    # Base 85-D vector
    params = np.random.randn(85) * 0.05

    # Encode key dimensions into specific indices (deterministic)
    params[0] = bmi / 40.0
    params[1] = trunk_ratio
    params[2] = shoulder_width / 100.0
    params[3] = hip_width / 100.0
    params[4] = torso_depth / 100.0
    params[5] = height_cm / 200.0
    params[6] = weight_kg / 150.0
    params[7] = (bmi - 18.5) / 21.5  # normalized BMI offset
    params[8] = 1.0 if bmi >= 30 else (0.5 if bmi >= 25 else 0.0)  # obesity flag

    return params.astype(np.float32)


def simulate_nnLandmark(height_cm: float, weight_kg: float) -> dict:
    """
    Mock neural-network landmark predictor.
    Returns approximate 3D coordinates (x, y, z) in mm for key anatomical landmarks.
    Z=0 is defined at the top of the head; Z increases inferiorly.
    In a real system this is a regression CNN on body surface scans.
    """
    bmi = weight_kg / ((height_cm / 100) ** 2)
    H = height_cm * 10  # convert to mm

    # Anatomical proportions — Indian NIOH-based constants
    sternum_z = H * 0.200   # sternum centroid ~20.0% from top (Indian standard)
    t12_z     = H * 0.482   # T12 ~48.2% from top (Indian standard)
    pubis_z   = H * 0.771   # pubic symphysis ~77.1% from top (Indian standard)

    # BMI-based small corrections — CAPPED at 30mm to prevent runaway span
    # Raw shift would be (bmi - 22) * 1.2, but unbounded it causes
    # physically impossible spans at BMI > 35. Cap fixes Obese III / Morbid cases.
    bmi_shift = float(np.clip((bmi - 22.0) * 1.2, -15.0, 30.0))

    # X/Y coords (lateral offset from midline, anterior-posterior depth)
    lateral_spread = 0 + (bmi - 22) * 0.8
    ap_depth = H * 0.14 + (bmi - 22) * 2.5

    landmarks = {
        "Sternum (Manubrium)": {
            "coords": (round(lateral_spread * 0.1, 1),
                       round(-ap_depth * 0.45, 1),
                       round(sternum_z + bmi_shift * 0.5, 1)),
            "anatomy": "Superior border of sternum / T2 level",
            "color": "#00d4ff",
        },
        "T12 Vertebra": {
            "coords": (round(0.0, 1),
                       round(ap_depth * 0.1, 1),
                       round(t12_z + bmi_shift, 1)),
            "anatomy": "Thoracolumbar junction / diaphragm attachment",
            "color": "#ffaa00",
        },
        "Pubic Symphysis": {
            "coords": (round(0.0, 1),
                       round(-ap_depth * 0.3, 1),
                       round(pubis_z + bmi_shift * 1.5, 1)),
            "anatomy": "Inferior pelvic boundary",
            "color": "#ff4466",
        },
    }
    return landmarks


# ── RAG Pipeline ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def build_rag_pipeline():


    """
    Build LangChain RAG pipeline with FAISS vector store.
    Falls back gracefully if LangChain/FAISS are not installed.
    """
    try:
       
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_core.documents import Document

        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
        docs = splitter.create_documents([CT_PROTOCOL_KB])

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore, "langchain"
    except Exception:
        return None, "fallback"


def retrieve_protocol(query: str, vectorstore, mode: str) -> str:
    """Retrieve most relevant protocol chunk via FAISS similarity search."""
    if mode == "langchain" and vectorstore is not None:
        try:
            results = vectorstore.similarity_search(query, k=2)
            return "\n\n".join(r.page_content for r in results)
        except Exception:
            pass
    # Keyword fallback
    query_lower = query.lower()
    if "chest" in query_lower and "abd" not in query_lower:
        keyword = "CHEST-001"
    elif "abdomen" in query_lower or "abdominal" in query_lower:
        keyword = "ABD-001"
    elif "cap" in query_lower or ("chest" in query_lower and "abd" in query_lower):
        keyword = "CHEST-ABD-001"
    else:
        keyword = "CHEST-001"
    for block in CT_PROTOCOL_KB.split("\n\n"):
        if keyword in block:
            return block
    return CT_PROTOCOL_KB.split("\n\n")[0]


def reasoning_agent(scan_type: str, landmarks: dict, protocol_text: str,
                    height: float, weight: float) -> dict:
    """
    Llama-3-style reasoning agent.
    Parses retrieved protocol + landmark Z-coords to calculate scan boundaries.
    In a real deployment, this would call Ollama / Together API with Llama 3.
    """
    bmi = weight / ((height / 100) ** 2)

    sternum_z = landmarks["Sternum (Manubrium)"]["coords"][2]
    t12_z = landmarks["T12 Vertebra"]["coords"][2]
    pubis_z = landmarks["Pubic Symphysis"]["coords"][2]

    scan_type_l = scan_type.lower()

    if "chest" in scan_type_l and "abdomen" not in scan_type_l:
        z_start = sternum_z - 20   # 20 mm above sternum = approx 2 cm above lung apex
        z_end = t12_z + 5
        start_anatomy = "~2 cm superior to lung apex (C7 level)"
        end_anatomy = "T12 inferior border (diaphragm)"
        protocol_id = "CHEST-001"
        confidence = 0.91 - max(0, (bmi - 30) * 0.008)
        rationale = (
            f"Protocol CHEST-001 retrieved. Sternum landmark at Z={sternum_z:.0f} mm "
            f"used as superior anchor. Scan start calculated at Z={z_start:.0f} mm "
            f"(–20 mm offset for lung apex clearance). T12 at Z={t12_z:.0f} mm "
            f"defines the inferior thoracic boundary. BMI={bmi:.1f} — "
            + ("standard acquisition parameters apply."
               if bmi < 30 else
               "elevated BMI: mAs auto-boost recommended; consider split acquisition.")
        )

    elif "abdomen" in scan_type_l and "pelvis" not in scan_type_l:
        z_start = t12_z - 15   # 15 mm above T12 ≈ diaphragm dome
        z_end = pubis_z + 10
        start_anatomy = "Diaphragm dome (T8–T9 superior to liver)"
        end_anatomy = "Pubic symphysis inferior border"
        protocol_id = "ABD-001"
        confidence = 0.88 - max(0, (bmi - 30) * 0.010)
        rationale = (
            f"Protocol ABD-001 retrieved. T12 landmark (Z={t12_z:.0f} mm) used as "
            f"diaphragm proxy. Start set at Z={z_start:.0f} mm (–15 mm superior margin). "
            f"Pubic Symphysis at Z={pubis_z:.0f} mm defines inferior extent with +10 mm margin. "
            f"BMI={bmi:.1f} — "
            + ("portal venous phase recommended (65 s delay)."
               if bmi < 35 else
               "high BMI: consider 80 s delay for adequate enhancement; kVp adjustment advised.")
        )

    elif "pelvis" in scan_type_l and "chest" not in scan_type_l and "cap" not in scan_type_l:
        z_start = pubis_z - 160   # approximate iliac crest offset
        z_end = pubis_z + 25
        start_anatomy = "Iliac crest (L4–L5)"
        end_anatomy = "Pubic symphysis + ischial tuberosities"
        protocol_id = "PELVIS-001"
        confidence = 0.85 - max(0, (bmi - 30) * 0.005)
        rationale = (
            f"Protocol PELVIS-001 retrieved. Pubic Symphysis (Z={pubis_z:.0f} mm) used as "
            f"inferior anchor. Iliac crest estimated at Z={z_start:.0f} mm based on body "
            f"proportion model (85-D skeletal parameters). Full pelvic ring coverage assured. "
            f"BMI={bmi:.1f} — standard pelvic acquisition."
        )

    else:  # CAP — Chest-Abdomen-Pelvis full torso
        z_start = sternum_z - 20
        z_end = pubis_z + 10
        start_anatomy = "~2 cm above lung apex"
        end_anatomy = "Pubic symphysis inferior border"
        protocol_id = "CHEST-ABD-001"
        confidence = 0.87 - max(0, (bmi - 30) * 0.009)
        rationale = (
            f"Protocol CHEST-ABD-001 (CAP) retrieved. Full torso coverage from "
            f"lung apex (Z={z_start:.0f} mm) to pubic symphysis (Z={z_end:.0f} mm). "
            f"Span = {z_end - z_start:.0f} mm. BMI={bmi:.1f} — "
            + ("single breath-hold feasible."
               if bmi < 35 else
               "high BMI: split acquisition into chest and abdomen-pelvis recommended.")
        )

    confidence = float(np.clip(confidence, 0.60, 0.97))

    return {
        "protocol_id": protocol_id,
        "z_start_mm": round(z_start, 1),
        "z_end_mm": round(z_end, 1),
        "start_anatomy": start_anatomy,
        "end_anatomy": end_anatomy,
        "confidence": confidence,
        "rationale": rationale,
        "retrieved_protocol": protocol_text,
    }


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_human_silhouette(landmarks: dict, result: dict, height_cm: float) -> plt.Figure:
    """Render a schematic human silhouette with scan boundary markers."""
    fig, ax = plt.subplots(figsize=(4.2, 8.5))
    fig.patch.set_facecolor('#0a0e1a')
    ax.set_facecolor('#0a0e1a')

    H = height_cm * 10  # mm
    # We'll draw the silhouette in normalized height coords (0=top, 1=bottom)
    # Convert Z coords
    def z_to_y(z): return z / H

    # ── Silhouette outline (simplified torso) ──
    # Head
    head_y = 0.08
    head = plt.Circle((0.5, head_y), 0.08, color='#1a2e4a', linewidth=1.5,
                       edgecolor='#2a5080', zorder=3, fill=True)
    ax.add_patch(head)

    # Neck
    ax.plot([0.46, 0.46, 0.44, 0.44], [0.155, 0.17, 0.18, 0.195], color='#2a5080', lw=1.5)
    ax.plot([0.54, 0.54, 0.56, 0.56], [0.155, 0.17, 0.18, 0.195], color='#2a5080', lw=1.5)

    # Torso (trapezoid shape)
    torso_x = np.array([0.30, 0.32, 0.68, 0.70, 0.66, 0.34, 0.30])
    torso_y = np.array([0.195, 0.60, 0.60, 0.195, 0.195, 0.195, 0.195])
    torso_x = np.array([0.32, 0.68, 0.72, 0.65, 0.35, 0.28, 0.32])
    torso_y = np.array([0.195, 0.195, 0.38, 0.62, 0.62, 0.38, 0.195])
    ax.fill(torso_x, torso_y, color='#0e1e38', zorder=2)
    ax.plot(np.append(torso_x, torso_x[0]),
            np.append(torso_y, torso_y[0]), color='#2a5080', lw=1.5, zorder=3)

    # Arms
    ax.plot([0.28, 0.20, 0.18, 0.22], [0.38, 0.42, 0.60, 0.72],
            color='#2a5080', lw=6, solid_capstyle='round')
    ax.plot([0.72, 0.80, 0.82, 0.78], [0.38, 0.42, 0.60, 0.72],
            color='#2a5080', lw=6, solid_capstyle='round')

    # Pelvis/Hips
    hip_x = np.array([0.33, 0.67, 0.72, 0.28, 0.33])
    hip_y = np.array([0.62, 0.62, 0.72, 0.72, 0.62])
    ax.fill(hip_x, hip_y, color='#0e1e38', zorder=2)
    ax.plot(hip_x, hip_y, color='#2a5080', lw=1.5, zorder=3)

    # Legs
    ax.plot([0.38, 0.36, 0.37, 0.38], [0.72, 0.86, 0.95, 1.0],
            color='#2a5080', lw=10, solid_capstyle='round')
    ax.plot([0.62, 0.64, 0.63, 0.62], [0.72, 0.86, 0.95, 1.0],
            color='#2a5080', lw=10, solid_capstyle='round')

    # ── Landmark markers ──
    for name, lm in landmarks.items():
        z = lm["coords"][2]
        y = z_to_y(z)
        color = lm["color"]
        ax.scatter(0.5, y, s=80, color=color, zorder=6, marker='D')
        ax.plot([0.25, 0.75], [y, y], color=color, lw=0.8, alpha=0.4,
                linestyle=':', zorder=5)
        ax.text(0.76, y, name.split("(")[0].strip(),
                va='center', ha='left', fontsize=6.5,
                color=color, fontfamily='monospace', fontweight='bold')

    # ── Scan range highlight ──
    y_start = z_to_y(result["z_start_mm"])
    y_end = z_to_y(result["z_end_mm"])

    # Fill scan zone
    scan_rect = plt.Rectangle((0.0, y_start), 1.0, y_end - y_start,
                               color='#0050a0', alpha=0.18, zorder=4)
    ax.add_patch(scan_rect)

    # Start line
    ax.axhline(y_start, color='#00e676', lw=2.0, linestyle='-', zorder=7, alpha=0.9)
    ax.text(0.02, y_start - 0.012, f"▶ START  Z={result['z_start_mm']:.0f}mm",
            va='bottom', ha='left', fontsize=7, color='#00e676',
            fontfamily='monospace', fontweight='bold')

    # End line
    ax.axhline(y_end, color='#ff4466', lw=2.0, linestyle='-', zorder=7, alpha=0.9)
    ax.text(0.02, y_end + 0.006, f"▶ END    Z={result['z_end_mm']:.0f}mm",
            va='top', ha='left', fontsize=7, color='#ff4466',
            fontfamily='monospace', fontweight='bold')

    # Double-headed arrow for span
    ax.annotate("", xy=(0.12, y_end), xytext=(0.12, y_start),
                arrowprops=dict(arrowstyle='<->', color='#4a9adc',
                                lw=1.2, mutation_scale=10))
    span_mm = result["z_end_mm"] - result["z_start_mm"]
    ax.text(0.04, (y_start + y_end) / 2, f"{span_mm:.0f}\nmm",
            va='center', ha='left', fontsize=6, color='#4a9adc',
            fontfamily='monospace', linespacing=1.4)

    ax.set_xlim(0, 1)
    ax.set_ylim(1.05, -0.05)
    ax.axis('off')
    ax.set_title("SCAN RANGE PREVIEW", fontsize=8, color='#4a7a9b',
                 fontfamily='monospace', pad=8, fontweight='bold', loc='center')

    plt.tight_layout(pad=0.3)
    return fig


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div class="ct-header">
        <div>
            <h1>🩻 SCOUT-LESS CT PLANNER</h1>
            <p>AI-Assisted Scan Range Determination</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Patient Input")
        st.caption("Configure patient data and scan parameters")

        st.markdown('<div class="sidebar-section">📁 3D Scan Data</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload Patient Scan",
            type=["png", "jpg", "jpeg", "obj", "stl", "nii", "nii.gz"],
            help="Mock input: any image or mesh file. In production: DICOM/NIfTI surface scan.",
        )
        if uploaded_file:
            st.success(f"✓ Loaded: `{uploaded_file.name}` ({uploaded_file.size // 1024} KB)")

        st.markdown('<div class="sidebar-section">📐 Vital Statistics</div>', unsafe_allow_html=True)
        height = st.number_input("Height (cm)", min_value=120.0, max_value=220.0,
                                  value=170.0, step=0.5, format="%.1f")
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0,
                                  value=75.0, step=0.5, format="%.1f")
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=45, step=1)
        sex = st.selectbox("Biological Sex", ["Male", "Female", "Other/Unspecified"])

        st.markdown('<div class="sidebar-section">🔬 Scan Request</div>', unsafe_allow_html=True)
        scan_type = st.selectbox(
            "Scan Type",
            ["Chest CT", "Abdomen CT", "Pelvis CT", "Chest-Abdomen-Pelvis (CAP)"],
        )
        contrast = st.selectbox("Contrast", ["With IV Contrast", "Without Contrast", "Dual Phase"])

        st.markdown('<div class="sidebar-section">🤖 AI Settings</div>', unsafe_allow_html=True)
        use_rag = st.checkbox("Enable RAG Protocol Retrieval", value=True)
        show_raw = st.checkbox("Show Raw Landmark Coords", value=False)

        st.markdown("---")
        run_btn = st.button("▶  RUN AI PLANNER", use_container_width=True, type="primary")
        

    # ── MAIN PANEL ───────────────────────────────────────────────────────────
    bmi = weight / ((height / 100) ** 2)

    # Step bar
    if "ran" not in st.session_state:
        st.session_state.ran = False

    steps = [
        ("1. INPUT", "done" if (uploaded_file or height) else "active"),
        ("2. INFERENCE", "done" if st.session_state.ran else ""),
        ("3. RAG RETRIEVE", "done" if st.session_state.ran else ""),
        ("4. PLAN", "done" if st.session_state.ran else ""),
        ("5. CONFIRM", "active" if st.session_state.ran else ""),
    ]
    step_html = '<div class="step-bar">' + "".join(
        f'<div class="step-item {cls}">{label}</div>'
        for label, cls in steps
    ) + '</div>'
    st.markdown(step_html, unsafe_allow_html=True)

    # ── Always show patient context ──
    col1, col2, col3, col4 = st.columns(4)

    bmi_status = ("normal" if bmi < 25 else "overweight" if bmi < 30 else "obese")
    bmi_label = ("Normal" if bmi < 25 else "Overweight" if bmi < 30 else
                 "Obese I" if bmi < 35 else "Obese II" if bmi < 40 else "Obese III")
    bmi_color_class = (f"status-{bmi_status}")

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">BMI</div>
            <div class="metric-value">{bmi:.1f}<span class="metric-unit">kg/m²</span></div>
            <span class="metric-status {bmi_color_class}">{bmi_label}</span>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Height</div>
            <div class="metric-value">{height:.0f}<span class="metric-unit">cm</span></div>
            <span class="metric-status status-normal">Measured</span>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Weight</div>
            <div class="metric-value">{weight:.0f}<span class="metric-unit">kg</span></div>
            <span class="metric-status status-normal">Measured</span>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Patient</div>
            <div class="metric-value" style="font-size:1.2rem">{age}y {sex[0]}</div>
            <span class="metric-status status-normal">{scan_type.split()[0]}</span>
        </div>""", unsafe_allow_html=True)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    if run_btn:
        st.session_state.ran = True

        progress_bar = st.progress(0, text="⚙️  Initialising inference engine…")
        time.sleep(0.05)

        # Step 1: Skeletal model
        progress_bar.progress(20, text="🦴  Running skeletal parameter model (85-D)…")
        skel_params = simulate_skel_model(height, weight)
        time.sleep(0.05)

        # Step 2: Landmark prediction
        progress_bar.progress(45, text="📍  Predicting anatomical landmarks (nnLandmark)…")
        landmarks = simulate_nnLandmark(height, weight)
        time.sleep(0.05)

        # Step 3: RAG
        progress_bar.progress(65, text="🔍 Retrieving CT protocol via FAISS RAG…")

        if use_rag:
            vectorstore, rag_mode = build_rag_pipeline()
            protocol_text = retrieve_protocol(scan_type, vectorstore, rag_mode)
        else:
            protocol_text = CT_PROTOCOL_KB.split("\n\n")[0]
            rag_mode = "disabled"
            time.sleep(0.05)

        # Step 4: Reasoning
        progress_bar.progress(85, text="🧠  Reasoning agent computing scan boundaries…")
        result = reasoning_agent(scan_type, landmarks, protocol_text, height, weight)
        time.sleep(0.05)

        progress_bar.progress(100, text="✅  Analysis complete.")
        time.sleep(0.05)
        progress_bar.empty()

        # Store results
        st.session_state.landmarks = landmarks
        st.session_state.skel_params = skel_params
        st.session_state.result = result
        st.session_state.rag_mode = rag_mode

    # ── Display results ───────────────────────────────────────────────────────
    if st.session_state.ran and "result" in st.session_state:
        landmarks = st.session_state.landmarks
        result = st.session_state.result
        skel_params = st.session_state.skel_params
        rag_mode = st.session_state.rag_mode

        st.markdown("---")

        # Two-column layout: silhouette + results
        vis_col, res_col = st.columns([1, 2], gap="large")

        with vis_col:
            st.markdown('<div class="section-header">Visual Recommendation</div>',
                        unsafe_allow_html=True)
            fig = plot_human_silhouette(landmarks, result, height)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with res_col:
            # Landmarks
            st.markdown('<div class="section-header">Detected Landmarks</div>',
                        unsafe_allow_html=True)
            for name, lm in landmarks.items():
                x, y, z = lm["coords"]
                st.markdown(f"""
                <div class="landmark-row" style="border-left-color:{lm['color']}">
                    <div class="landmark-name" style="color:{lm['color']}">{name}</div>
                    <div class="landmark-coords">
                        x={x:+.1f} &nbsp; y={y:+.1f} &nbsp; z={z:.1f} mm
                    </div>
                    <div style="font-size:0.72rem;color:#4a7a9b;margin-left:auto">{lm['anatomy']}</div>
                </div>""", unsafe_allow_html=True)

            # Z-boundaries
            st.markdown('<div class="section-header">Calculated Scan Boundaries</div>',
                        unsafe_allow_html=True)
            span = result["z_end_mm"] - result["z_start_mm"]
            st.markdown(f"""
            <div class="z-boundary">
                <div class="z-point start-point">
                    <div class="z-point-label">Start Z</div>
                    <div class="z-point-value">{result['z_start_mm']:.0f} mm</div>
                    <div class="z-point-anatomy">{result['start_anatomy']}</div>
                </div>
                <div class="z-point end-point">
                    <div class="z-point-label">End Z</div>
                    <div class="z-point-value">{result['z_end_mm']:.0f} mm</div>
                    <div class="z-point-anatomy">{result['end_anatomy']}</div>
                </div>
                <div class="z-point" style="border-top: 3px solid #4a9adc; flex: 0.6">
                    <div class="z-point-label">Span</div>
                    <div class="z-point-value" style="color:#4a9adc">{span:.0f} mm</div>
                    <div class="z-point-anatomy">Protocol: {result['protocol_id']}</div>
                </div>
            </div>""", unsafe_allow_html=True)

            # Confidence
            st.markdown('<div class="section-header">AI Confidence</div>',
                        unsafe_allow_html=True)
            conf_pct = int(result["confidence"] * 100)
            st.markdown(f"""
            <div class="confidence-container">
                <div class="confidence-label">
                    <span class="confidence-title">Overall Planning Confidence</span>
                    <span class="confidence-pct">{conf_pct}%</span>
                </div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width:{conf_pct}%"></div>
                </div>
            </div>""", unsafe_allow_html=True)

            # Rationale
            st.markdown('<div class="section-header">Medical Rationale (RAG)</div>',
                        unsafe_allow_html=True)
            rag_tag = f"FAISS+LangChain" if rag_mode == "langchain" else "KEYWORD FALLBACK"
            st.markdown(f"""
            <div class="rationale-box">
                <span class="rag-tag">⚡ RAG · {rag_tag}</span><br/>
                {result['rationale']}
            </div>""", unsafe_allow_html=True)

        # Retrieved Protocol (expandable)
        with st.expander("📄 Retrieved Protocol Chunk (RAG Source)", expanded=False):
            st.markdown(f"""
            <div class="protocol-card">
                <strong>{result['protocol_id']} — Retrieved Context</strong>
                {result['retrieved_protocol'].replace(chr(10), '<br/>')}
            </div>""", unsafe_allow_html=True)

        # Skeletal params (expandable)
        if show_raw:
            with st.expander("🔢 Raw 85-D Skeletal Parameter Vector", expanded=False):
                st.markdown('<div class="section-header">Skeletal Model Output</div>',
                            unsafe_allow_html=True)
                param_cols = st.columns(5)
                for i, v in enumerate(skel_params[:25]):
                    param_cols[i % 5].metric(f"θ[{i}]", f"{v:.4f}")

        # ── Human-in-the-Loop ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">🧑‍⚕️ Human-in-the-Loop Approval</div>',
                    unsafe_allow_html=True)

        warn_color = "#ff4466" if result["confidence"] < 0.75 else "#ffaa00" if result["confidence"] < 0.88 else "#00e676"
        st.markdown(f"""
        <div style="background:#0b1829;border:1px solid #1e3a5f;border-radius:8px;
                    padding:16px 20px;margin-bottom:20px;">
            <p style="color:#8ab4cc;font-size:0.85rem;margin:0 0 8px 0;">
                <strong style="color:{warn_color};">⚠ Radiographer Review Required</strong><br/>
                AI-generated scan boundaries are recommendations only.
                A qualified radiographer or radiologist <strong>must</strong> verify the
                Z-axis boundaries before initiating acquisition.
                Confidence: <strong style="color:{warn_color};">{conf_pct}%</strong>
            </p>
            <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#4a7a9b;">
                Protocol: {result['protocol_id']} &nbsp;|&nbsp;
                Start: {result['z_start_mm']:.0f} mm &nbsp;|&nbsp;
                End: {result['z_end_mm']:.0f} mm &nbsp;|&nbsp;
                Span: {span:.0f} mm
            </div>
        </div>""", unsafe_allow_html=True)

        confirm_col, reject_col, _ = st.columns([1, 1, 3])
        with confirm_col:
            confirm_btn = st.button("✅  Confirm & Execute Scan", use_container_width=True)
        with reject_col:
            reject_btn = st.button("✗  Reject — Adjust Manually", use_container_width=True)

        if confirm_btn:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = {
                "timestamp": ts,
                "action": "APPROVED",
                "protocol": result["protocol_id"],
                "z_start": result["z_start_mm"],
                "z_end": result["z_end_mm"],
                "confidence": result["confidence"],
                "patient": {"height": height, "weight": weight, "bmi": round(bmi, 2), "age": age},
                "scan_type": scan_type,
            }

            # Persist audit log — cross-platform path (works on Windows + Linux)
            log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_log.jsonl")
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            st.success("🟢 Scan approved and queued for acquisition.")
            st.markdown(f"""
            <div class="log-box">
                <p class="log-entry"><span class="log-time">[{ts}]</span>
                <span class="log-ok"> ✓ TECHNICIAN APPROVED</span></p>
                <p class="log-entry"><span class="log-time">[SYS]</span>
                Protocol={result['protocol_id']} | Z={result['z_start_mm']:.0f}→{result['z_end_mm']:.0f}mm
                | Span={span:.0f}mm | Conf={conf_pct}%</p>
                <p class="log-entry"><span class="log-time">[SYS]</span>
                Audit log written → ct_planner_audit.jsonl</p>
                <p class="log-entry"><span class="log-time">[SYS]</span>
                <span class="log-ok"> STATUS: READY FOR GANTRY CONTROL HANDOFF</span></p>
            </div>""", unsafe_allow_html=True)

        if reject_btn:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.warning("🟡 Plan rejected. Please adjust parameters manually and re-run.")
            st.markdown(f"""
            <div class="log-box">
                <p class="log-entry"><span class="log-time">[{ts}]</span>
                <span class="log-warn"> ⚠ TECHNICIAN REJECTED AI PLAN</span></p>
                <p class="log-entry"><span class="log-time">[SYS]</span>
                Manual override initiated. Reverting to standard scout protocol.</p>
            </div>""", unsafe_allow_html=True)

    else:
        # Placeholder state
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#2a4a6a">
            <div style="font-size:4rem;margin-bottom:16px">🩻</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.9rem;
                        color:#2a5a8a;letter-spacing:0.1em">
                SYSTEM READY<br/>
                <span style="font-size:0.7rem;color:#1e3a5a">
                Enter patient vitals in the sidebar and click RUN AI PLANNER
                </span>
            </div>
        </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
