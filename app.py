"""
Scout-less CT Scan Planning System
Orchestration Layer (Upgraded Architecture)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, Ellipse, FancyBboxPatch
from matplotlib.path import Path
import matplotlib.patches as mpatches
import os, json, time, re
from datetime import datetime
from scipy.ndimage import distance_transform_edt, gaussian_filter
from dotenv import load_dotenv
load_dotenv()

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

/* LLM badge */
.llm-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: linear-gradient(135deg, #0a1a3a, #001830);
    border: 1px solid #0070d0; border-radius: 20px;
    padding: 3px 12px; font-size: 0.68rem; color: #40b0ff;
    font-family: 'Space Mono', monospace; letter-spacing: 0.08em;
    margin-bottom: 10px; font-weight: 700;
}
.llm-badge .dot { width: 6px; height: 6px; border-radius: 50%;
    background: #00d4ff; animation: blink 1.4s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }
.llm-source-badge {
    font-size: 0.62rem; padding: 2px 8px; border-radius: 4px;
    font-family: 'Space Mono', monospace; display: inline-block; margin-left: 6px;
}
.llm-ollama  { background:#0a2010; color:#00c878; border:1px solid #00c87840; }
.llm-groq    { background:#1a1000; color:#ffaa00; border:1px solid #ffaa0040; }
.llm-fallback{ background:#0a0a20; color:#6a9ec0; border:1px solid #2a4a7040; }
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
    np.random.seed(int(height_cm + weight_kg) % 2**31)
    bmi = weight_kg / ((height_cm / 100) ** 2)
    trunk_ratio = 0.52 + (bmi - 22) * 0.002
    shoulder_width = height_cm * 0.259 * (1 + (bmi - 22) * 0.003)
    hip_width = height_cm * 0.191 * (1 + (bmi - 22) * 0.005)
    torso_depth = height_cm * 0.145 * (1 + (bmi - 22) * 0.006)

    params = np.random.randn(85) * 0.05
    params[0] = bmi / 40.0
    params[1] = trunk_ratio
    params[2] = shoulder_width / 100.0
    params[3] = hip_width / 100.0
    params[4] = torso_depth / 100.0
    params[5] = height_cm / 200.0
    params[6] = weight_kg / 150.0
    params[7] = (bmi - 18.5) / 21.5
    params[8] = 1.0 if bmi >= 30 else (0.5 if bmi >= 25 else 0.0)
    return params.astype(np.float32)


# ── 1. MULTI-MODAL INPUT SIMULATION & LATE FUSION ─────────────────────────────

def simulate_multimodal_fusion(rgb_status: str, depth_status: str, ir_status: str, bmi: float) -> dict:
    """
    Applies Late Fusion strategy to process inputs from three distinct sensors.
    Individual camera tracking weightings adapt dynamically based on patient morphology factors.
    """
    # RGB baseline tracking confidence drops under shadows/gowns
    rgb_conf = 0.94 if rgb_status == "Optimal" else 0.45
    # Standard depth systems struggle with complex clothing layers
    depth_conf = 0.92 if depth_status == "Optimal" else 0.50
    # Thermal/Infrared identifies true body anatomy boundaries regardless of fabric
    ir_conf = 0.96 if ir_status == "Optimal" else 0.60
    
    # Extreme physical morphology degrades optical accuracy
    if bmi > 38.0:
        rgb_conf *= 0.85
        depth_conf *= 0.75  # Significant attenuation on standard spatial boundaries
        ir_conf *= 0.95     # IR remains highly resilient to clothing structures
        
    # Late Fusion: Compute weighted ensemble confidence coefficient
    fused_confidence = (rgb_conf * 0.25) + (depth_conf * 0.35) + (ir_conf * 0.40)
    
    # Construct a descriptive, unified feature representation array
    fused_vector = np.array([rgb_conf, depth_conf, ir_conf, fused_confidence], dtype=np.float32)
    
    return {
        "fused_vector": fused_vector,
        "combined_confidence": float(fused_confidence),
        "status_summary": f"RGB={rgb_status} | Depth={depth_status} | IR={ir_status}"
    }


# ── 2. 3D HEATMAP REGRESSION & SOFT-ARGMAX ────────────────────────────────────

def generate_edt_heatmaps(height_cm: float, weight_kg: float, noise_scale: float = 2.5) -> dict:
    """
    Simulates volumetric probability distributions across a 3D structural voxel lattice grid.
    Uses Euclidean Distance Transform (EDT) mappings centered on anatomical targets.
    Grid shape: (Z=128, Y=32, X=32) representing the bounding space.
    """
    bmi = weight_kg / ((height_cm / 100) ** 2)
    H = height_cm * 10  # mm
    
    # Expected centroid positions
    sternum_z_target = H * 0.200
    t12_z_target     = H * 0.482
    pubis_z_target   = H * 0.771
    
    # Scale adjustments
    bmi_shift = float(np.clip((bmi - 22.0) * 1.2, -15.0, 32.0))
    
    # Ground-truth coordinate mappings translated onto the 128-slice Z grid
    z_map_sternum = (sternum_z_target + bmi_shift * 0.5) / H * 127
    z_map_t12     = (t12_z_target + bmi_shift) / H * 127
    z_map_pubis   = (pubis_z_target + bmi_shift * 1.5) / H * 127
    
    # Increase uncertainty based on structural noise
    peak_width_base = 3.5 + (max(0.0, bmi - 25.0) * 0.3) + (noise_scale * 0.5)
    
    landmarks_meta = {
        "Carina (Sternum Anchor)": {"z_vox": z_map_sternum, "pw": peak_width_base, "color": "#00d4ff", "anatomy": "Superior border of sternum / T2 level"},
        "T12 Vertebra": {"z_vox": z_map_t12, "pw": peak_width_base * 1.2, "color": "#ffaa00", "anatomy": "Thoracolumbar junction / diaphragm attachment"},
        "Pubic Symphysis": {"z_vox": z_map_pubis, "pw": peak_width_base * 1.5, "color": "#ff4466", "anatomy": "Inferior pelvic boundary"}
    }
    
    heatmaps = {}
    for name, data in landmarks_meta.items():
        # Instantiate a clean 3D lattice field space
        grid = np.zeros((128, 32, 32), dtype=np.float32)
        
        # Center target coordinate indices
        cz, cy, cx = int(np.clip(data["z_vox"], 0, 127)), 16, 16
        grid[cz, cy, cx] = 1.0
        
        # Compute distance transforms relative to the source point
        edt = distance_transform_edt(1.0 - grid)
        
        # Formulate Gaussian probabilistic fields based on distance weights
        sigma = data["pw"]
        heatmap_3d = np.exp(-(edt ** 2) / (2.0 * (sigma ** 2)))
        # Normalize probability map distribution
        heatmap_3d /= np.sum(heatmap_3d)
        
        heatmaps[name] = {
            "volume": heatmap_3d,
            "peak_width": float(sigma),
            "color": data["color"],
            "anatomy": data["anatomy"]
        }
        
    return heatmaps


def extract_coords_from_heatmap(heatmap_3d: np.ndarray, height_cm: float) -> tuple:
    """
    Applies a Soft-Argmax mathematical transformation layer across the 3D probability matrix.
    Achieves sub-voxel precise tracking localization via soft density center-of-mass evaluation.
    """
    Z, Y, X = heatmap_3d.shape
    
    # Generate continuous index coordinate coordinate tensors
    z_indices = np.arange(Z, dtype=np.float32)
    y_indices = np.arange(Y, dtype=np.float32)
    x_indices = np.arange(X, dtype=np.float32)
    
    # Collapse spatial dimensions to evaluate center of mass projections
    prob_z = np.sum(heatmap_3d, axis=(1, 2))
    prob_y = np.sum(heatmap_3d, axis=(0, 2))
    prob_x = np.sum(heatmap_3d, axis=(0, 1))
    
    # Execute Soft-Argmax expectation formulas: E[v] = SUM(v * P(v))
    soft_z = float(np.sum(z_indices * prob_z))
    soft_y = float(np.sum(y_indices * prob_y))
    soft_x = float(np.sum(x_indices * prob_x))
    
    # Translate structural grid index coordinates into physical scanning metrics (mm)
    H_mm = height_cm * 10
    physical_z = (soft_z / (Z - 1)) * H_mm
    
    # Center translation offsets for physical coordinate visualization mapping
    physical_x = (soft_x - (X / 2)) * 8.0
    physical_y = (soft_y - (Y / 2)) * 8.0
    
    return physical_x, physical_y, physical_z


# ── RAG Pipeline ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def build_rag_pipeline():
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
    if mode == "langchain" and vectorstore is not None:
        try:
            results = vectorstore.similarity_search(query, k=2)
            return "\n\n".join(r.page_content for r in results)
        except Exception:
            pass
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


# ── 3. LLAMA 3 REASONING CHAIN ────────────────────────────────────────────────

# Structured JSON schema Llama must return
_LLAMA_SYSTEM = """You are a Senior Clinical Radiologist and CT Protocol Specialist.
Your task: given patient biometrics, detected anatomical landmarks, and a retrieved
CT protocol, compute exact Z-axis scan boundaries and provide a 2-sentence clinical
justification.

RULES:
- Always cite the Protocol ID in your rationale.
- If BMI is outside 18–35, OR any landmark peak_width > 7.5 mm, set low_confidence=true
  and recommend "Manual Scout Over-ride".
- Return ONLY valid JSON — no markdown, no preamble, no extra keys.

OUTPUT SCHEMA (strict):
{
  "suggested_start_z": <float mm>,
  "suggested_end_z":   <float mm>,
  "protocol_id":       <string>,
  "medical_rationale": <2-sentence string citing protocol and patient BMI>,
  "low_confidence":    <true|false>,
  "override_message":  <string or "">
}"""

_LLAMA_HUMAN = """PATIENT CONTEXT:
- Height: {height_cm} cm | Weight: {weight_kg} kg | BMI: {bmi:.1f}
- Age group: {age_note}
- Scan requested: {scan_type}

DETECTED LANDMARKS (Soft-Argmax, sub-voxel mm):
- Carina (T4-T5):        Z = {carina_z:.1f} mm  | Peak Width (uncertainty): {carina_pw:.2f} mm
- T12 Vertebra:          Z = {t12_z:.1f} mm    | Peak Width (uncertainty): {t12_pw:.2f} mm
- Pubic Symphysis:       Z = {pubis_z:.1f} mm  | Peak Width (uncertainty): {pubis_pw:.2f} mm

MULTI-MODAL FUSION CONFIDENCE: {fusion_conf:.1%}
(RGB={rgb_c:.2f} | Depth={dep_c:.2f} | IR={ir_c:.2f})

RETRIEVED PROTOCOL (FAISS RAG):
{protocol_text}

Using ONLY the landmark Z coordinates and the protocol above, compute the scan boundaries.
Apply the stated Z-Start and Z-End offsets from the protocol. Return JSON only."""


def _rule_based_fallback(scan_type: str, landmarks: dict, protocol_text: str,
                          height: float, weight: float,
                          combined_sensor_conf: float) -> dict:
    """Original rule-based engine — used when no LLM is available."""
    bmi = weight / ((height / 100) ** 2)
    carina_z = landmarks["Carina (Sternum Anchor)"]["coords"][2]
    t12_z    = landmarks["T12 Vertebra"]["coords"][2]
    pubis_z  = landmarks["Pubic Symphysis"]["coords"][2]
    max_pw   = max(lm["peak_width"] for lm in landmarks.values())

    safety_trigger = max_pw > 7.5 or bmi > 45.0 or combined_sensor_conf < 0.70
    safety_message = ("High Uncertainty Detected: Reverting to Ultra-Low Dose Scout Validation."
                      if safety_trigger else "")

    s = scan_type.lower()
    if "chest" in s and "abdomen" not in s and "cap" not in s:
        z_start, z_end = carina_z - 20, t12_z + 5
        start_a, end_a = "~2 cm superior to lung apex (C7)", "T12 inferior border"
        pid, base_c = "CHEST-001", 0.93
    elif "abdomen" in s and "pelvis" not in s and "cap" not in s:
        z_start, z_end = t12_z - 15, pubis_z + 10
        start_a, end_a = "Diaphragm dome (T8–T9)", "Pubic symphysis inferior"
        pid, base_c = "ABD-001", 0.90
    elif "pelvis" in s and "chest" not in s and "cap" not in s:
        z_start, z_end = pubis_z - 160, pubis_z + 25
        start_a, end_a = "Iliac crest (L4–L5)", "Pubic symphysis + ischial tub."
        pid, base_c = "PELVIS-001", 0.88
    else:
        z_start, z_end = carina_z - 20, pubis_z + 10
        start_a, end_a = "~2 cm above lung apex", "Pubic symphysis inferior"
        pid, base_c = "CHEST-ABD-001", 0.89

    conf = float(np.clip(base_c * combined_sensor_conf - max_pw * 0.01, 0.30, 0.98))
    if safety_trigger:
        conf = float(np.clip(conf * 0.5, 0.30, 0.55))
        rationale = (f"⚠ SAFETY GATE ACTIVE — Peak Width {max_pw:.2f} mm / BMI {bmi:.1f}. "
                     f"{safety_message} Manual scout mandatory.")
    else:
        rationale = (f"Protocol {pid} verified via RAG. Carina at Z={carina_z:.0f} mm, "
                     f"T12 at Z={t12_z:.0f} mm. Fusion confidence {combined_sensor_conf:.1%}. "
                     f"BMI={bmi:.1f} — aleatoric variance within thresholds (max PW={max_pw:.2f} mm).")

    return {
        "protocol_id":    pid,
        "z_start_mm":     round(z_start, 1),
        "z_end_mm":       round(z_end, 1),
        "start_anatomy":  start_a,
        "end_anatomy":    end_a,
        "confidence":     conf,
        "rationale":      rationale,
        "retrieved_protocol": protocol_text,
        "safety_trigger": safety_trigger,
        "safety_message": safety_message,
        "llm_backend":    "rule_fallback",
    }


def reasoning_agent(scan_type: str, landmarks: dict, protocol_text: str,
                    height: float, weight: float,
                    combined_sensor_conf: float,
                    llm_backend: str = "auto",
                    groq_api_key: str = "") -> dict:
    """
    LlamaReasoningChain — Clinical Decision Support System.

    Tries backends in this order (based on llm_backend setting):
      1. Ollama  (local Llama 3 — zero cost, best for demo)
      2. Groq    (cloud Llama 3 — needs API key, fastest)
      3. Rule-based fallback (always available)

    The LLM receives: patient biometrics, Soft-Argmax landmark Z coordinates,
    peak-width aleatoric uncertainty, fusion confidence, and the RAG-retrieved
    protocol.  It returns structured JSON with z_start, z_end, rationale,
    low_confidence flag, and override message.
    """
    bmi = weight / ((height / 100) ** 2)
    carina_z  = landmarks["Carina (Sternum Anchor)"]["coords"][2]
    t12_z     = landmarks["T12 Vertebra"]["coords"][2]
    pubis_z   = landmarks["Pubic Symphysis"]["coords"][2]
    carina_pw = landmarks["Carina (Sternum Anchor)"]["peak_width"]
    t12_pw    = landmarks["T12 Vertebra"]["peak_width"]
    pubis_pw  = landmarks["Pubic Symphysis"]["peak_width"]

    fv = landmarks.get("_fusion_vector", [0.9, 0.9, 0.9])
    age_note = "Paediatric (<18)" if height < 145 else "Adult (18–60)" if height < 185 else "Tall adult (>185 cm)"

    human_msg = _LLAMA_HUMAN.format(
        height_cm=height, weight_kg=weight, bmi=bmi,
        age_note=age_note, scan_type=scan_type,
        carina_z=carina_z, carina_pw=carina_pw,
        t12_z=t12_z, t12_pw=t12_pw,
        pubis_z=pubis_z, pubis_pw=pubis_pw,
        fusion_conf=combined_sensor_conf,
        rgb_c=fv[0], dep_c=fv[1], ir_c=fv[2],
        protocol_text=protocol_text[:800],
    )

    llm_response = None
    used_backend = "rule_fallback"

    # ── Try Ollama (local Llama 3) ────────────────────────────────────────────
    if llm_backend in ("auto", "ollama"):
        try:
            from langchain_ollama import ChatOllama
            from langchain_core.messages import SystemMessage, HumanMessage
            llm = ChatOllama(model="llama3", temperature=0.1,
                             timeout=25, base_url="http://localhost:11434")
            resp = llm.invoke([SystemMessage(content=_LLAMA_SYSTEM),
                               HumanMessage(content=human_msg)])
            llm_response = resp.content
            used_backend = "ollama"
        except Exception:
            pass

    # ── Try Groq (cloud Llama 3) ──────────────────────────────────────────────
    if llm_response is None and llm_backend in ("auto", "groq") and groq_api_key:
        try:
            from langchain_groq import ChatGroq
            from langchain_core.messages import SystemMessage, HumanMessage
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1,
                           groq_api_key=groq_api_key)
            resp = llm.invoke([SystemMessage(content=_LLAMA_SYSTEM),
                               HumanMessage(content=human_msg)])
            llm_response = resp.content
            used_backend = "groq"
            st.session_state["_llm_raw"] = llm_response
        except Exception:
            pass

    # ── Parse JSON from LLM response ─────────────────────────────────────────
    if llm_response is not None:
        try:
            # Strip any markdown fences the model may have added
            clean = re.sub(r"```(?:json)?|```", "", llm_response).strip()
            parsed = json.loads(clean)

            z_start  = float(parsed["suggested_start_z"])
            z_end    = float(parsed["suggested_end_z"])
            pid      = str(parsed.get("protocol_id", "CHEST-001"))
            rationale= str(parsed.get("medical_rationale", ""))
            low_conf = bool(parsed.get("low_confidence", False))
            override = str(parsed.get("override_message", ""))

            # Sanity-check the LLM output
            if z_end <= z_start or z_start < 0 or z_end > height * 12:
                raise ValueError("LLM returned physically invalid Z values")

            safety_trigger = low_conf
            safety_message = override if override else (
                "High Uncertainty Detected: Reverting to Ultra-Low Dose Scout Validation."
                if low_conf else "")

            # Derive anatomy labels from protocol_id
            _anatomy = {
                "CHEST-001":     ("~2 cm superior to lung apex", "T12 inferior border"),
                "ABD-001":       ("Diaphragm dome (T8–T9)",      "Pubic symphysis inferior"),
                "PELVIS-001":    ("Iliac crest (L4–L5)",         "Pubic symphysis + ischial tub."),
                "CHEST-ABD-001": ("~2 cm above lung apex",       "Pubic symphysis inferior"),
            }
            start_a, end_a = _anatomy.get(pid, ("Computed by LLM", "Computed by LLM"))

            max_pw = max(lm["peak_width"] for lm in landmarks.values())
            conf_raw = 0.93 * combined_sensor_conf - max_pw * 0.01
            if safety_trigger:
                conf_raw *= 0.55
            confidence = float(np.clip(conf_raw, 0.30, 0.98))

            return {
                "protocol_id":       pid,
                "z_start_mm":        round(z_start, 1),
                "z_end_mm":          round(z_end, 1),
                "start_anatomy":     start_a,
                "end_anatomy":       end_a,
                "confidence":        confidence,
                "rationale":         rationale,
                "retrieved_protocol":protocol_text,
                "safety_trigger":    safety_trigger,
                "safety_message":    safety_message,
                "llm_backend":       used_backend,
            }
        except Exception:
            pass   # fall through to rule-based

    # ── Rule-based fallback ───────────────────────────────────────────────────
    result = _rule_based_fallback(scan_type, landmarks, protocol_text,
                                   height, weight, combined_sensor_conf)
    result["llm_backend"] = "rule_fallback"
    return result




# ── 4. ANATOMICAL SILHOUETTE WITH ORGANS ──────────────────────────────────────

def plot_human_silhouette(landmarks: dict, result: dict, height_cm: float,
                           sex: str = "Male", bmi: float = 22.0) -> plt.Figure:
    """
    Renders a detailed anatomical human silhouette that adapts to:
      - Sex  : Male vs Female body shape (shoulder/hip ratio, breast outline)
      - BMI  : silhouette width scales with adiposity
      - Scan : organ highlights match the selected scan type
    Overlays glowing EDT heatmap blobs, Soft-Argmax landmark markers,
    schematic internal organs, and clear START / END scan boundary lines.
    """
    fig, ax = plt.subplots(figsize=(5.0, 10.0))
    fig.patch.set_facecolor('#080d18')
    ax.set_facecolor('#080d18')

    H = height_cm * 10
    def z_to_y(z): return float(z) / H

    is_female = "female" in sex.lower()
    bmi_scale = float(np.clip((bmi - 18.5) / 25.0, 0.0, 1.0))  # 0=lean, 1=obese

    # ── Body width scalars ────────────────────────────────────────────────────
    shoulder_w = (0.195 if is_female else 0.230) + bmi_scale * 0.040
    hip_w      = (0.225 if is_female else 0.185) + bmi_scale * 0.055
    waist_w    = (0.150 if is_female else 0.160) + bmi_scale * 0.060
    torso_depth_factor = 1.0 + bmi_scale * 0.35   # used for organ sizing hints

    cx = 0.50   # centre x

    # ─────────────────────────────────────────────────────────────────────────
    # BODY OUTLINE (filled polygon, spine-region darker)
    # Points go: left shoulder → left waist → left hip → crotch → right ...
    # ─────────────────────────────────────────────────────────────────────────
    sh_y, hip_y, crotch_y = 0.20, 0.62, 0.72
    waist_y = (sh_y + hip_y) / 2 + 0.02

    left_x  = [cx - shoulder_w, cx - waist_w,  cx - hip_w,   cx - hip_w*0.6]
    left_y  = [sh_y,            waist_y,        hip_y,        crotch_y]
    right_x = [cx + shoulder_w, cx + waist_w,  cx + hip_w,   cx + hip_w*0.6]
    right_y = left_y

    # Combine into closed polygon
    body_xs = left_x  + [cx] + list(reversed(right_x)) + [cx + shoulder_w]
    body_ys = left_y  + [crotch_y] + list(reversed(right_y)) + [sh_y]
    body_fill = '#0d1f38'
    body_edge = '#2a5a8a'

    ax.fill(body_xs, body_ys, color=body_fill, zorder=2, alpha=0.95)
    ax.plot(body_xs + [body_xs[0]], body_ys + [body_ys[0]],
            color=body_edge, lw=1.8, zorder=3)

    # ── Head ─────────────────────────────────────────────────────────────────
    head_r  = 0.072 + bmi_scale * 0.010
    head_cy = 0.095
    head = Ellipse((cx, head_cy), head_r * 2, head_r * 2.1,
                   color='#0e1e38', zorder=3)
    ax.add_patch(head)
    # head outline
    theta = np.linspace(0, 2*np.pi, 80)
    ax.plot(cx + head_r * np.cos(theta),
            head_cy + head_r * 1.05 * np.sin(theta),
            color=body_edge, lw=1.5, zorder=4)

    # Neck
    neck_w = 0.030 + bmi_scale * 0.012
    ax.fill([cx-neck_w, cx-neck_w, cx+neck_w, cx+neck_w],
            [head_cy+head_r*0.7, sh_y, sh_y, head_cy+head_r*0.7],
            color='#0d1f38', zorder=3)
    ax.plot([cx-neck_w, cx-neck_w], [head_cy+head_r*0.7, sh_y],
            color=body_edge, lw=1.5, zorder=4)
    ax.plot([cx+neck_w, cx+neck_w], [head_cy+head_r*0.7, sh_y],
            color=body_edge, lw=1.5, zorder=4)

    # ── Arms ─────────────────────────────────────────────────────────────────
    arm_w   = 5 + bmi_scale * 3
    elbow_y = (sh_y + hip_y) / 2
    hand_y  = hip_y + 0.06
    for sign in (-1, +1):
        xs = [cx + sign*(shoulder_w - 0.01),
              cx + sign*(shoulder_w + 0.07),
              cx + sign*(shoulder_w + 0.085),
              cx + sign*(shoulder_w + 0.05)]
        ys = [sh_y, elbow_y, (elbow_y+hand_y)/2, hand_y]
        ax.plot(xs, ys, color=body_edge, lw=arm_w,
                solid_capstyle='round', zorder=2)

    # ── Female breast outline ─────────────────────────────────────────────────
    if is_female:
        breast_y = sh_y + 0.10
        breast_r_x = 0.055 + bmi_scale * 0.025
        breast_r_y = 0.040 + bmi_scale * 0.020
        for sign in (-1, +1):
            b = Ellipse((cx + sign * 0.055, breast_y),
                        breast_r_x * 2, breast_r_y * 2,
                        color='#122a48', zorder=4, alpha=0.7)
            ax.add_patch(b)
            ax.plot(cx + sign*0.055 + breast_r_x*np.cos(theta),
                    breast_y + breast_r_y*np.sin(theta),
                    color='#3a6a9a', lw=0.8, alpha=0.6, zorder=5)

    # ── Legs ─────────────────────────────────────────────────────────────────
    leg_w   = 10 + bmi_scale * 6
    knee_y  = 0.855
    foot_y  = 1.00
    for sign, leg_x in ((-1, cx - hip_w*0.42), (+1, cx + hip_w*0.42)):
        ax.plot([leg_x, leg_x + sign*0.015, leg_x + sign*0.010, leg_x],
                [crotch_y, knee_y, (knee_y+foot_y)/2, foot_y],
                color=body_edge, lw=leg_w, solid_capstyle='round', zorder=2)

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL ORGANS (schematic, dim, inside the torso)
    # ─────────────────────────────────────────────────────────────────────────
    org_alpha = 0.28

    # Lungs
    lung_top_y    = z_to_y(H * 0.22)
    lung_bottom_y = z_to_y(H * 0.45)
    lung_h        = lung_bottom_y - lung_top_y
    lung_cx_off   = 0.068
    for sign in (-1, +1):
        lung = Ellipse((cx + sign * lung_cx_off,
                        (lung_top_y + lung_bottom_y) / 2),
                       0.090, lung_h,
                       color='#1a4a6a', alpha=org_alpha, zorder=3)
        ax.add_patch(lung)
        ax.plot(cx + sign*lung_cx_off + 0.045*np.cos(theta),
                (lung_top_y+lung_bottom_y)/2 + lung_h/2*np.sin(theta),
                color='#2a7aaa', lw=0.6, alpha=org_alpha*1.5, zorder=3)

    # Heart (left-centred)
    heart_cy = z_to_y(H * 0.28)
    heart = Ellipse((cx - 0.028, heart_cy), 0.055, 0.060,
                    color='#3a1a2a', alpha=org_alpha * 1.4, zorder=4)
    ax.add_patch(heart)
    ax.plot(cx - 0.028 + 0.028*np.cos(theta),
            heart_cy + 0.030*np.sin(theta),
            color='#8a2a4a', lw=0.7, alpha=org_alpha*2, zorder=4)

    # Liver (right upper abdomen)
    liver_cy = z_to_y(H * 0.505)
    liver = Ellipse((cx + 0.050, liver_cy), 0.110, 0.060,
                    color='#2a1a0a', alpha=org_alpha * 1.6, zorder=3)
    ax.add_patch(liver)
    ax.plot(cx + 0.050 + 0.055*np.cos(theta),
            liver_cy + 0.030*np.sin(theta),
            color='#7a3a1a', lw=0.6, alpha=org_alpha*2, zorder=3)

    # Stomach (left mid-abdomen)
    stom_cy = z_to_y(H * 0.515)
    stomach = Ellipse((cx - 0.055, stom_cy), 0.070, 0.050,
                      color='#1a2a1a', alpha=org_alpha, zorder=3)
    ax.add_patch(stomach)

    # Kidneys
    kid_cy  = z_to_y(H * 0.545)
    kid_h   = 0.055 * (1 + bmi_scale*0.1)
    for sign in (-1, +1):
        kid = Ellipse((cx + sign * 0.080, kid_cy),
                      0.030, kid_h,
                      color='#1a2a0a', alpha=org_alpha*1.5, zorder=3)
        ax.add_patch(kid)
        ax.plot(cx + sign*0.080 + 0.015*np.cos(theta),
                kid_cy + kid_h/2*np.sin(theta),
                color='#3a6a2a', lw=0.6, alpha=org_alpha*2, zorder=3)

    # Intestines / bowel (mid-lower abdomen)
    bowel_cy = z_to_y(H * 0.610)
    bowel = Ellipse((cx, bowel_cy), 0.165 + bmi_scale*0.03, 0.080,
                    color='#1a1a2a', alpha=org_alpha, zorder=3)
    ax.add_patch(bowel)

    # Bladder (lower pelvis)
    blad_cy = z_to_y(H * 0.72)
    bladder = Ellipse((cx, blad_cy), 0.055, 0.040,
                      color='#0a1a2a', alpha=org_alpha*1.2, zorder=3)
    ax.add_patch(bladder)

    # Spine (midline dotted line)
    spine_top  = z_to_y(H * 0.16)
    spine_bot  = z_to_y(H * 0.73)
    ax.plot([cx, cx], [spine_top, spine_bot],
            color='#2a4a6a', lw=1.0, linestyle=':', alpha=0.5, zorder=4)

    # ─────────────────────────────────────────────────────────────────────────
    # HEATMAP PROBABILITY BLOBS
    # ─────────────────────────────────────────────────────────────────────────
    for name, lm in landmarks.items():
        z_c   = lm["coords"][2]
        y_c   = z_to_y(z_c)
        color = lm["color"]
        pw    = lm.get("peak_width", 6.0)

        # Glowing Gaussian blob
        sigma_y = pw / H * 1.8
        sigma_x = 0.020 + bmi_scale * 0.005
        y_span  = np.linspace(y_c - sigma_y*4, y_c + sigma_y*4, 60)
        x_span  = np.linspace(0.28, 0.72, 30)
        Xg, Yg  = np.meshgrid(x_span, y_span)
        blob    = np.exp(-((Xg - cx)**2 / (2*sigma_x**2) +
                           (Yg - y_c)**2  / (2*sigma_y**2)))

        cname = {"#00d4ff": "cool", "#ffaa00": "Wistia",
                 "#ff4466": "gist_heat"}.get(color, "cool")
        ax.contourf(Xg, Yg, blob, levels=10,
                    cmap=plt.get_cmap(cname), alpha=0.40, zorder=5)

        # Soft-Argmax centre dot
        ax.scatter(cx, y_c, s=55, color=color,
                   edgecolors='#ffffff', linewidths=0.7, zorder=7, marker='o',
                   path_effects=[pe.withStroke(linewidth=3, foreground='#00000066')])
        ax.plot([0.24, 0.76], [y_c, y_c], color=color,
                lw=0.7, alpha=0.35, linestyle='--', zorder=6)
        ax.text(0.77, y_c, name.split("(")[0].strip(),
                va='center', ha='left', fontsize=6.5,
                color=color, fontfamily='monospace', fontweight='bold')

    # ─────────────────────────────────────────────────────────────────────────
    # SCAN RANGE — prominent START / END with arrows & shading
    # ─────────────────────────────────────────────────────────────────────────
    y_start = z_to_y(result["z_start_mm"])
    y_end   = z_to_y(result["z_end_mm"])
    safety  = result.get("safety_trigger", False)

    scan_color_fill = '#ff3322' if safety else '#0055cc'
    lc_start = '#ffaa00' if safety else '#00ff88'
    lc_end   = '#ff4444' if safety else '#ff4466'
    lstyle   = (0, (4, 3)) if safety else '-'

    # Shaded scan zone
    rect = plt.Rectangle((0.22, y_start), 0.56, y_end - y_start,
                          color=scan_color_fill, alpha=0.12, zorder=6)
    ax.add_patch(rect)
    # Side tick marks to make boundaries unmistakable
    for side_x in (0.22, 0.78):
        ax.plot([side_x - 0.015, side_x + 0.015], [y_start, y_start],
                color=lc_start, lw=2.0, zorder=9)
        ax.plot([side_x - 0.015, side_x + 0.015], [y_end, y_end],
                color=lc_end, lw=2.0, zorder=9)

    # Horizontal boundary lines
    ax.axhline(y_start, xmin=0.0, xmax=1.0,
               color=lc_start, lw=2.2, linestyle=lstyle, zorder=8, alpha=0.95)
    ax.axhline(y_end,   xmin=0.0, xmax=1.0,
               color=lc_end,   lw=2.2, linestyle=lstyle, zorder=8, alpha=0.95)

    # Labels
    ax.text(0.02, y_start - 0.013,
            f"▶ START  Z={result['z_start_mm']:.0f} mm",
            va='bottom', ha='left', fontsize=7, color=lc_start,
            fontfamily='monospace', fontweight='bold')
    ax.text(0.02, y_end + 0.007,
            f"▶ END    Z={result['z_end_mm']:.0f} mm",
            va='top', ha='left', fontsize=7, color=lc_end,
            fontfamily='monospace', fontweight='bold')

    # Double-headed span arrow
    ax.annotate("", xy=(0.11, y_end), xytext=(0.11, y_start),
                arrowprops=dict(arrowstyle='<->', color='#5aaae0', lw=1.2,
                                mutation_scale=10))
    span_mm = result["z_end_mm"] - result["z_start_mm"]
    ax.text(0.025, (y_start + y_end) / 2,
            f"{span_mm:.0f}\nmm", va='center', ha='left', fontsize=6.5,
            color='#5aaae0', fontfamily='monospace', linespacing=1.4)

    # ── Title & axes ─────────────────────────────────────────────────────────
    sex_tag = "♀ FEMALE" if is_female else "♂ MALE"
    bmi_tag = ("LEAN" if bmi < 22 else "NORMAL" if bmi < 25 else
               "OVERWEIGHT" if bmi < 30 else f"OBESE BMI{bmi:.0f}")
    ax.set_title(f"ANATOMICAL SCAN PLANNER  ·  {sex_tag}  ·  {bmi_tag}",
                 fontsize=7.5, color='#4a7a9b',
                 fontfamily='monospace', pad=8, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(1.04, -0.04)
    ax.axis('off')
    plt.tight_layout(pad=0.2)
    return fig


    """
    Renders an anatomical patient silhouette outline graph, plotting an advanced 
    2D projection mapping of volumetric probability heatmaps as smooth glowing gradients.
    """
    fig, ax = plt.subplots(figsize=(4.5, 9.0))
    fig.patch.set_facecolor('#0a0e1a')
    ax.set_facecolor('#0a0e1a')

    H = height_cm * 10  # mm
    def z_to_y(z): return z / H

    # ── Silhouette outline construction ──
    head = plt.Circle((0.5, 0.08), 0.08, color='#1a2e4a', linewidth=1.5, edgecolor='#2a5080', zorder=2)
    ax.add_patch(head)
    ax.plot([0.46, 0.46, 0.44, 0.44], [0.155, 0.17, 0.18, 0.195], color='#2a5080', lw=1.5)
    ax.plot([0.54, 0.54, 0.56, 0.56], [0.155, 0.17, 0.18, 0.195], color='#2a5080', lw=1.5)

    torso_x = np.array([0.32, 0.68, 0.72, 0.65, 0.35, 0.28, 0.32])
    torso_y = np.array([0.195, 0.195, 0.38, 0.62, 0.62, 0.38, 0.195])
    ax.fill(torso_x, torso_y, color='#0e1e38', zorder=1)
    ax.plot(np.append(torso_x, torso_x[0]), np.append(torso_y, torso_y[0]), color='#2a5080', lw=1.5, zorder=2)

    ax.plot([0.28, 0.20, 0.18, 0.22], [0.38, 0.42, 0.60, 0.72], color='#2a5080', lw=6, solid_capstyle='round')
    ax.plot([0.72, 0.80, 0.82, 0.78], [0.38, 0.42, 0.60, 0.72], color='#2a5080', lw=6, solid_capstyle='round')

    hip_x = np.array([0.33, 0.67, 0.72, 0.28, 0.33])
    hip_y = np.array([0.62, 0.62, 0.72, 0.72, 0.62])
    ax.fill(hip_x, hip_y, color='#0e1e38', zorder=1)
    ax.plot(hip_x, hip_y, color='#2a5080', lw=1.5, zorder=2)

    ax.plot([0.38, 0.36, 0.37, 0.38], [0.72, 0.86, 0.95, 1.0], color='#2a5080', lw=10, solid_capstyle='round')
    ax.plot([0.62, 0.64, 0.63, 0.62], [0.72, 0.86, 0.95, 1.0], color='#2a5080', lw=10, solid_capstyle='round')

    # ── 2D GLOWING HEATMAP PROJECTION OVERLAYS ──
    # Projects the volumetric tensor arrays along the coronal view axis
    for name, lm in landmarks.items():
        z_center = lm["coords"][2]
        y_center_norm = z_to_y(z_center)
        
        # Pull 3D volume, squashing depth axes to construct relative probability projections
        volume_3d = lm["volume"]
        coronal_projection = np.sum(volume_3d, axis=2) # Shape: (128, 32)
        
        # Map localized subgrid spans centered on the visual coordinate node
        z_mesh = np.linspace(y_center_norm - 0.08, y_center_norm + 0.08, 64)
        x_mesh = np.linspace(0.32, 0.68, 32)
        X_m, Z_m = np.meshgrid(x_mesh, z_mesh)
        
        # Calculate localized 2D Gaussian scaling matrices to simulate glowing fields
        dist_sq = ((X_m - 0.5)**2 / 0.04) + ((Z_m - y_center_norm)**2 / 0.0035)
        sigma_glow = lm["peak_width"] / 12.0
        glow_intensity = np.exp(-dist_sq / (2.0 * sigma_glow**2))
        
        # Superimpose custom colormap gradients representing anatomical confidence zones
        if name.startswith("Carina"):
            cmap_glow = plt.cm.get_cmap("cool")
        elif name.startswith("T12"):
            cmap_glow = plt.cm.get_cmap("Wistia")
        else:
            cmap_glow = plt.cm.get_cmap("gist_heat")
            
        ax.contourf(X_m, Z_m, glow_intensity, levels=14, cmap=cmap_glow, alpha=0.45, zorder=3)
        
        # Exact Soft-Argmax Center Coordinate Indicator Point
        ax.scatter(0.5, y_center_norm, s=45, color=lm["color"], edgecolors='#ffffff', linewidths=0.5, zorder=5, marker='o')
        ax.plot([0.22, 0.78], [y_center_norm, y_center_norm], color=lm["color"], lw=0.7, alpha=0.35, linestyle='--', zorder=4)
        
        ax.text(0.79, y_center_norm, name.split("(")[0].strip(), va='center', ha='left',
                fontsize=7, color=lm["color"], fontfamily='monospace', fontweight='bold')

    # ── Target Volume Scan Box Highlights ──
    y_start = z_to_y(result["z_start_mm"])
    y_end = z_to_y(result["z_end_mm"])

    if result.get("safety_trigger", False):
        # High uncertainty visual boundary styling adjustments
        scan_rect = plt.Rectangle((0.0, y_start), 1.0, y_end - y_start, color='#ff4444', alpha=0.08, zorder=2)
        line_style = (0, (3, 3)) # Dashed boundary tracking line style
        line_color_start, line_color_end = '#ffaa00', '#ff4444'
    else:
        scan_rect = plt.Rectangle((0.0, y_start), 1.0, y_end - y_start, color='#0050a0', alpha=0.16, zorder=2)
        line_style = '-'
        line_color_start, line_color_end = '#00e676', '#ff4466'
        
    ax.add_patch(scan_rect)

    # Upper start marker lines
    ax.axhline(y_start, color=line_color_start, lw=1.8, linestyle=line_style, zorder=6)
    ax.text(0.02, y_start - 0.012, f"▶ START  Z={result['z_start_mm']:.0f}mm",
            va='bottom', ha='left', fontsize=7, color=line_color_start, fontfamily='monospace', fontweight='bold')

    # Lower terminate boundary markers
    ax.axhline(y_end, color=line_color_end, lw=1.8, linestyle=line_style, zorder=6)
    ax.text(0.02, y_end + 0.006, f"▶ END    Z={result['z_end_mm']:.0f}mm",
            va='top', ha='left', fontsize=7, color=line_color_end, fontfamily='monospace', fontweight='bold')

    # Double-headed bounding tracking vectors
    ax.annotate("", xy=(0.10, y_end), xytext=(0.10, y_start), arrowprops=dict(arrowstyle='<->', color='#4a9adc', lw=1.1))
    span_mm = result["z_end_mm"] - result["z_start_mm"]
    ax.text(0.03, (y_start + y_end) / 2, f"{span_mm:.0f}\nmm", va='center', ha='left', fontsize=6.5, color='#4a9adc', fontfamily='monospace')

    ax.set_xlim(0, 1)
    ax.set_ylim(1.05, -0.05)
    ax.axis('off')
    ax.set_title("3D HEATMAP REGRESSION FIELD", fontsize=8, color='#4a7a9b', fontfamily='monospace', pad=8, fontweight='bold')

    plt.tight_layout(pad=0.2)
    return fig


# ── Main App Execution ────────────────────────────────────────────────────────

def main():
    st.markdown("""
    <div class="ct-header">
        <div>
            <h1>🩻 SCOUT-LESS CT PLANNER</h1>
            <p>Multi-Modal Fusion · Heatmap Regression · Llama 3 Clinical Reasoning</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SIDEBAR CONFIGURATIONS ──
    with st.sidebar:
        st.markdown("### ⚙️ Patient Input Setup")
        st.caption("Configure multi-sensor fusion inputs & metrics")

        st.markdown('<div class="sidebar-section">📡 Multi-Modal Surface Sensors</div>', unsafe_allow_html=True)
        rgb_cam = st.selectbox("RGB Camera (Texture Tracking)", ["Optimal", "Degraded (Shadows/Obstructions)"])
        depth_cam = st.selectbox("Depth Camera (3D Spatial)", ["Optimal", "Degraded (Thick Gowns/Layers)"])
        ir_cam = st.selectbox("Infrared Sensor (Thermal Contour)", ["Optimal", "Degraded (Extreme Thermal Shielding)"])

        st.markdown('<div class="sidebar-section">📐 Body Morphometry</div>', unsafe_allow_html=True)
        height = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=170.0, step=0.5, format="%.1f")
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=75.0, step=0.5, format="%.1f")
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=45, step=1)
        sex = st.selectbox("Biological Sex", ["Male", "Female", "Other/Unspecified"])

        st.markdown('<div class="sidebar-section">🔬 Acquisition Target</div>', unsafe_allow_html=True)
        scan_type = st.selectbox("Scan Type", ["Chest CT", "Abdomen CT", "Pelvis CT", "Chest-Abdomen-Pelvis (CAP)"])
        contrast = st.selectbox("Contrast Profile", ["With IV Contrast", "Without Contrast", "Dual Phase"])

        st.markdown('<div class="sidebar-section">🤖 Advanced Engine Controls</div>', unsafe_allow_html=True)
        noise_level = st.slider("Heatmap Uncertainty (Noise Scale)", 0.5, 10.0, 2.5, 0.5)
        use_rag = st.checkbox("Enable RAG Protocol Retrieval", value=True)
        show_raw = st.checkbox("Show Raw Multi-Modal Metrics", value=False)

        st.markdown('<div class="sidebar-section">🧠 LLM Reasoning Backend</div>', unsafe_allow_html=True)
        llm_backend = st.selectbox(
            "Reasoning Engine",
            ["auto", "ollama (local Llama 3)", "groq (cloud Llama 3)", "rule-based only"],
            help="auto = try Ollama → Groq → rule-based"
        )
        llm_backend_key = llm_backend.split()[0]   # extract 'auto','ollama','groq','rule-based'
        groq_api_key = ""
        if "groq" in llm_backend:
            groq_api_key = st.text_input("Groq API Key", type="password",
                                          placeholder="gsk_...",
                                          help="Get free key at console.groq.com")
        if llm_backend_key == "ollama":
            st.caption("ℹ️ Requires Ollama running: `ollama run llama3`")
        elif llm_backend_key == "auto":
            st.caption("ℹ️ Tries Ollama → Groq → rule-based automatically")

        st.markdown("---")
        run_btn = st.button("▶  RUN MULTI-MODAL PLANNER", use_container_width=True, type="primary")

    bmi = weight / ((height / 100) ** 2)

    if "ran" not in st.session_state:
        st.session_state.ran = False

    steps = [
        ("1. MULTI-SENSOR", "done" if height else "active"),
        ("2. FUSION MATRIX", "done" if st.session_state.ran else ""),
        ("3. REGRESSION FIELD", "done" if st.session_state.ran else ""),
        ("4. RISK EVALUATION", "done" if st.session_state.ran else ""),
        ("5. CONTROL CONTROL", "active" if st.session_state.ran else ""),
    ]
    step_html = '<div class="step-bar">' + "".join(f'<div class="step-item {cls}">{label}</div>' for label, cls in steps) + '</div>'
    st.markdown(step_html, unsafe_allow_html=True)

    # ── Live Vitals Grid ──
    col1, col2, col3, col4 = st.columns(4)
    bmi_status = "normal" if bmi < 25 else "overweight" if bmi < 30 else "obese"
    bmi_label = "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese I" if bmi < 35 else "Obese II" if bmi < 40 else "Obese III"
    bmi_color_class = f"status-{bmi_status}"

    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">BMI Matrix</div><div class="metric-value">{bmi:.1f}<span class="metric-unit">kg/m²</span></div><span class="metric-status {bmi_color_class}">{bmi_label}</span></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Spatial Height</div><div class="metric-value">{height:.0f}<span class="metric-unit">cm</span></div><span class="metric-status status-normal">Calculated</span></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Patient Mass</div><div class="metric-value">{weight:.0f}<span class="metric-unit">kg</span></div><span class="metric-status status-normal">Calculated</span></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Target Profile</div><div class="metric-value" style="font-size:1.1rem">{age}y {sex[0]}</div><span class="metric-status status-normal">{scan_type.split()[0]}</span></div>', unsafe_allow_html=True)

    # ── Pipeline Pipeline Execution ──
    if run_btn:
        st.session_state.ran = True
        progress_bar = st.progress(0, text="Initializing multi-modal inference pipeline...")
        time.sleep(0.04)
        
        # Step 1: Execute Sensor Fusion Tracking
        progress_bar.progress(20, text="📡 Syncing sensors & processing Late-Fusion confidence matrices...")
        fusion_result = simulate_multimodal_fusion(rgb_cam, depth_cam, ir_cam, bmi)
        time.sleep(0.04)
        
        # Step 2: Compute Volumetric Probability Maps
        progress_bar.progress(45, text="📍 Regressing 3D volumetric probability fields (EDT Matrices)...")
        raw_heatmaps = generate_edt_heatmaps(height, weight, noise_level)
        time.sleep(0.04)
        
        # Step 3: Compute Sub-voxel Precise Points via Soft-Argmax
        progress_bar.progress(65, text="🔢 Executing Soft-Argmax spatial calculations across tensors...")
        processed_landmarks = {}
        for name, data in raw_heatmaps.items():
            fx, fy, fz = extract_coords_from_heatmap(data["volume"], height)
            processed_landmarks[name] = {
                "coords": (fx, fy, fz),
                "peak_width": data["peak_width"],
                "color": data["color"],
                "anatomy": data["anatomy"],
                "volume": data["volume"]
            }
            
        # Step 4: Retrieve Context Profiles via RAG
        progress_bar.progress(80, text="🔍 Extracting tracking standard guidelines via FAISS vector layers...")
        if use_rag:
            vectorstore, rag_mode = build_rag_pipeline()
            protocol_text = retrieve_protocol(scan_type, vectorstore, rag_mode)
        else:
            protocol_text = CT_PROTOCOL_KB.split("\n\n")[0]
            rag_mode = "disabled"
            
        # Step 5: Advanced Planning and Reasoning Matrix
        progress_bar.progress(95, text="🧠 Llama 3 reasoning agent computing scan boundaries...")
        agent_plan = reasoning_agent(
            scan_type, processed_landmarks, protocol_text,
            height, weight, fusion_result["combined_confidence"],
            llm_backend=llm_backend_key,
            groq_api_key=groq_api_key,
        )
        progress_bar.progress(100, text="✅ Pipeline inference matrix loaded successfully.")
        time.sleep(0.04)
        progress_bar.empty()
        
        st.session_state.landmarks = processed_landmarks
        st.session_state.result = agent_plan
        st.session_state.rag_mode = rag_mode
        st.session_state.fusion_data = fusion_result

    # ── Display Visual Results Matrices ──
    if st.session_state.ran and "result" in st.session_state:
        landmarks = st.session_state.landmarks
        result = st.session_state.result
        rag_mode = st.session_state.rag_mode
        fusion_data = st.session_state.fusion_data

        st.markdown("---")
        vis_col, res_col = st.columns([1, 2], gap="large")

        with vis_col:
            st.markdown('<div class="section-header">Anatomical Scan Planner</div>', unsafe_allow_html=True)
            fig = plot_human_silhouette(landmarks, result, height, sex=sex, bmi=bmi)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("""
            <div class="hm-legend">
                <span>Low prob</span>
                <div class="hm-swatch"></div>
                <span>High prob</span>
                &nbsp;·&nbsp; Blobs = EDT heatmap confidence
            </div>""", unsafe_allow_html=True)

        with res_col:
            # Safety Alert System Overrides
            if result.get("safety_trigger", False):
                st.markdown(f"""
                <div style="background:#2a0a0a; border:2px solid #ff4444; border-radius:8px; padding:16px; margin-bottom:15px;">
                    <h4 style="color:#ff4444; margin:0 0 8px 0; font-family:'Space Mono',monospace;">⚠️ AUTOMATED SAFETY GATE TRIGGERED</h4>
                    <p style="color:#ffcccc; font-size:0.88rem; margin:0;"><strong>{result['safety_message']}</strong></p>
                </div>""", unsafe_allow_html=True)

            # Volumetric Tracker Extractions
            st.markdown('<div class="section-header">Soft-Argmax Precise Coordinate Extractions</div>', unsafe_allow_html=True)
            for name, lm in landmarks.items():
                x, y, z = lm["coords"]
                st.markdown(f"""
                <div class="landmark-row" style="border-left-color:{lm['color']}">
                    <div class="landmark-name" style="color:{lm['color']}">{name}</div>
                    <div class="landmark-coords">X={x:+.2f} &nbsp; Y={y:+.2f} &nbsp; Z={z:.1f} mm</div>
                    <div style="font-size:0.72rem; color:#6a9ec0; margin-left:auto;">Peak Width spatial uncertainty: {lm['peak_width']:.2f}mm</div>
                </div>""", unsafe_allow_html=True)

            # Plan Boundaries
            st.markdown('<div class="section-header">Calculated Spatial Target Boundaries</div>', unsafe_allow_html=True)
            span = result["z_end_mm"] - result["z_start_mm"]
            st.markdown(f"""
            <div class="z-boundary">
                <div class="z-point start-point"><div class="z-point-label">Planned Start Z</div><div class="z-point-value">{result['z_start_mm']:.0f} mm</div><div class="z-point-anatomy">{result['start_anatomy']}</div></div>
                <div class="z-point end-point"><div class="z-point-label">Planned End Z</div><div class="z-point-value">{result['z_end_mm']:.0f} mm</div><div class="z-point-anatomy">{result['end_anatomy']}</div></div>
                <div class="z-point" style="border-top:3px solid #4a9adc; flex:0.6"><div class="z-point-label">Acquisition Span</div><div class="z-point-value" style="color:#4a9adc">{span:.0f} mm</div><div class="z-point-anatomy">ID: {result['protocol_id']}</div></div>
            </div>""", unsafe_allow_html=True)

            # Confidence Metric Displays
            st.markdown('<div class="section-header">System Integrated Confidence Coefficient</div>', unsafe_allow_html=True)
            conf_pct = int(result["confidence"] * 100)
            bar_color = "linear-gradient(90deg, #ff4444, #ffaa00)" if result["safety_trigger"] else "linear-gradient(90deg, #0050c0, #00a0ff, #00e676)"
            st.markdown(f"""
            <div class="confidence-container">
                <div class="confidence-label"><span class="confidence-title">Combined Prediction Interval Matrix</span><span class="confidence-pct" style="color:{'#ff4444' if result['safety_trigger'] else '#00e676'}">{conf_pct}%</span></div>
                <div class="confidence-bar-bg"><div class="confidence-bar-fill" style="width:{conf_pct}%; background:{bar_color};"></div></div>
            </div>""", unsafe_allow_html=True)

            # ── LLM Debug expander ──
            with st.expander("🔍 LLM Debug — Raw Response", expanded=False):
                st.write("**Backend used:**", result.get("llm_backend", "unknown"))
                st.write("**Raw LLM text:**", st.session_state.get("_llm_raw", "not captured"))

            # Context Explanations
            st.markdown('<div class="section-header">Medical Rationale & Safety Insights</div>', unsafe_allow_html=True)
            rag_tag = "FAISS+LangChain Vector Pipeline" if rag_mode == "langchain" else "KEYWORD MATRIX FALLBACK"
            backend = result.get("llm_backend", "rule_fallback")
            llm_cls  = {"ollama": "llm-ollama", "groq": "llm-groq"}.get(backend, "llm-fallback")
            llm_lbl  = {"ollama": "⚡ Llama 3 · Ollama Local",
                        "groq":   "⚡ Llama 3 · Groq Cloud",
                        "rule_fallback": "📐 Rule-Based Engine"}.get(backend, "📐 Rule-Based Engine")
            st.markdown(f"""
            <div class="rationale-box">
                <div style="margin-bottom:8px">
                    <span class="rag-tag">⚡ RAG · {rag_tag}</span>
                    <span class="llm-source-badge {llm_cls}">{llm_lbl}</span>
                </div>
                {result['rationale']}
            </div>""", unsafe_allow_html=True)

        # Multi-sensor Fusion Data Breakdown Matrix (Expandable)
        if show_raw:
            with st.expander("🔢 Raw Late-Fusion Multi-Modal Sensor Diagnostics", expanded=False):
                st.markdown('<div class="section-header">Late Fusion Reliability Matrix Matrix</div>', unsafe_allow_html=True)
                f_cols = st.columns(4)
                f_cols[0].metric("RGB Stream Weight", f"{fusion_data['fused_vector'][0]:.2f}")
                f_cols[1].metric("Depth Matrix Weight", f"{fusion_data['fused_vector'][1]:.2f}")
                f_cols[2].metric("Infrared Stream Weight", f"{fusion_data['fused_vector'][2]:.2f}")
                f_cols[3].metric("Fused Ensemble Index Coefficient", f"{fusion_data['combined_confidence']:.4f}")

        # Clinician Handoff Operations
        st.markdown("---")
        st.markdown('<div class="section-header">🧑‍⚕️ Clinician Handoff & Protocol Verification</div>', unsafe_allow_html=True)
        warn_color = "#ff4444" if result["safety_trigger"] else "#ffaa00" if result["confidence"] < 0.85 else "#00e676"
        
        st.markdown(f"""
        <div style="background:#0b1829; border:1px solid #1e3a5f; border-radius:8px; padding:16px 20px; margin-bottom:20px;">
            <p style="color:#8ab4cc; font-size:0.85rem; margin:0 0 8px 0;">
                <strong style="color:{warn_color};">⚠️ System Action Clearance Profile Required</strong><br/>
                {"MANUAL SCAN VALIDATION REQUIRED: Automated confidence thresholds breached. Proceed with ultra-low dose tracking scout." if result['safety_trigger'] else "Scan boundaries are verified to sub-voxel thresholds. Confirm execution plan to transition gantry coordinates."}
            </p>
        </div>""", unsafe_allow_html=True)

        confirm_col, reject_col, _ = st.columns([1, 1, 3])
        with confirm_col:
            confirm_btn = st.button("✅  Confirm & Transmit Plan", use_container_width=True)
        with reject_col:
            reject_btn = st.button("❌  Override — Manual Entry", use_container_width=True)

        if confirm_btn:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_log.jsonl")
            with open(log_path, "a") as f:
                f.write(json.dumps({"timestamp": ts, "action": "TRANSMITTED", "safety_override": result["safety_trigger"], "z_start": result["z_start_mm"], "z_end": result["z_end_mm"]}) + "\n")
            
            st.success("🟢 Scan configurations routed directly to execution pipelines.")
            st.markdown(f"""
            <div class="log-box">
                <p class="log-entry"><span class="log-time">[{ts}]</span> <span class="log-ok">✓ HARDWARE CONFIGS LOCKED</span></p>
                <p class="log-entry"><span class="log-time">[SYS]</span> Boundary parameters transmitted: Z={result['z_start_mm']} to {result['z_end_mm']} | Safety Override Status: {result['safety_trigger']}</p>
                <p class="log-entry"><span class="log-time">[SYS]</span> <span class="log-ok">STATUS: READY FOR MACHINE HANDOFF</span></p>
            </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()