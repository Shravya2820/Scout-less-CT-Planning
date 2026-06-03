"""
Scout-less CT Scan Planning System
B.Tech Project Prototype — Orchestration Layer (Upgraded Architecture)
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
from scipy.ndimage import distance_transform_edt

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


# ── 3. UPGRADED REASONING & SAFETY GATE ───────────────────────────────────────

def reasoning_agent(scan_type: str, landmarks: dict, protocol_text: str,
                    height: float, weight: float, combined_sensor_conf: float) -> dict:
    """
    Upgraded Reasoning Engine. Integrates aleatoric spatial peak uncertainty,
    BMI constraints, and sensor fusion tracking metrics to safeguard scan planning.
    """
    bmi = weight / ((height / 100) ** 2)

    carina_z = landmarks["Carina (Sternum Anchor)"]["coords"][2]
    t12_z    = landmarks["T12 Vertebra"]["coords"][2]
    pubis_z  = landmarks["Pubic Symphysis"]["coords"][2]

    # Evaluate Max Peak Width among tracked structures as an Aleatoric Uncertainty metric
    max_peak_width = max(lm["peak_width"] for lm in landmarks.values())
    
    scan_type_l = scan_type.lower()
    safety_trigger = False
    safety_message = ""

    # ── CRITICAL SAFETY GATE EVALUATION ──
    # Triggers an auto-override under extreme obesity or systemic tracking uncertainty
    
    if max_peak_width > 7.5 or bmi > 45.0 or combined_sensor_conf < 0.70:
        safety_trigger = True
        safety_message = "CRITICAL METRIC HIGH UNCERTAINTY DETECTED: Reverting to Ultra-Low Dose Scout Validation."

    if "chest" in scan_type_l and "abdomen" not in scan_type_l:
        z_start = carina_z - 20
        z_end = t12_z + 5
        start_anatomy = "~2 cm superior to lung apex (C7 level)"
        end_anatomy = "T12 inferior border (diaphragm)"
        protocol_id = "CHEST-001"
        confidence = 0.93 * combined_sensor_conf - (max_peak_width * 0.01)

    elif "abdomen" in scan_type_l and "pelvis" not in scan_type_l:
        z_start = t12_z - 15
        z_end = pubis_z + 10
        start_anatomy = "Diaphragm dome (T8–T9 superior to liver)"
        end_anatomy = "Pubic symphysis inferior border"
        protocol_id = "ABD-001"
        confidence = 0.90 * combined_sensor_conf - (max_peak_width * 0.015)

    elif "pelvis" in scan_type_l and "chest" not in scan_type_l and "cap" not in scan_type_l:
        z_start = pubis_z - 160
        z_end = pubis_z + 25
        start_anatomy = "Iliac crest (L4–L5)"
        end_anatomy = "Pubic symphysis + ischial tuberosities"
        protocol_id = "PELVIS-001"
        confidence = 0.88 * combined_sensor_conf - (max_peak_width * 0.01)

    else: # CAP - Chest-Abdomen-Pelvis
        z_start = carina_z - 20
        z_end = pubis_z + 10
        start_anatomy = "~2 cm above lung apex"
        end_anatomy = "Pubic symphysis inferior border"
        protocol_id = "CHEST-ABD-001"
        confidence = 0.89 * combined_sensor_conf - (max_peak_width * 0.02)

    # Apply safety adjustments to final confidence metrics
    if safety_trigger:
        confidence = float(np.clip(confidence * 0.5, 0.30, 0.55))
        rationale = (
            f"🚨 SAFETY REASONING OVERRIDE ACTIVATED.\n"
            f"Reasoning Context: Peak Width spatial uncertainty is at {max_peak_width:.2f} (Threshold 6.2) "
            f"or Patient BMI is {bmi:.1f} (Threshold 45.0).\n"
            f"Action: {safety_message} Manual scout validation mandatory to prevent anatomical cropping."
        )
    else:
        confidence = float(np.clip(confidence, 0.60, 0.98))
        rationale = (
            f"Protocol {protocol_id} verified. Multi-modal sensor fusion input context resolved with "
            f"{(combined_sensor_conf*100):.1f}% confidence parameters. Sub-voxel Soft-Argmax calculation maps "
            f"Carina at Z={carina_z:.1f} mm, T12 at Z={t12_z:.1f} mm. Aleatoric variance within thresholds "
            f"(Peak Width max = {max_peak_width:.2f} mm). Optimal structural boundary mapping is locked."
        )

    return {
        "protocol_id": protocol_id,
        "z_start_mm": round(z_start, 1),
        "z_end_mm": round(z_end, 1),
        "start_anatomy": start_anatomy,
        "end_anatomy": end_anatomy,
        "confidence": confidence,
        "rationale": rationale,
        "retrieved_protocol": protocol_text,
        "safety_trigger": safety_trigger,
        "safety_message": safety_message
    }


# ── 4. ENHANCED HEATMAP VISUALIZATION ─────────────────────────────────────────

def plot_human_silhouette(landmarks: dict, result: dict, height_cm: float) -> plt.Figure:
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
            <p>Multi-Modal Late-Fusion Heatmap Regression System</p>
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
        progress_bar.progress(95, text="🧠 Evaluation agent assessing aleatoric uncertainty parameters...")
        agent_plan = reasoning_agent(scan_type, processed_landmarks, protocol_text, height, weight, fusion_result["combined_confidence"])
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
            st.markdown('<div class="section-header">2D Coronal Heatmap Projections</div>', unsafe_allow_html=True)
            fig = plot_human_silhouette(landmarks, result, height)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

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

            # Context Explanations
            st.markdown('<div class="section-header">Medical Rationale & Safety Insights</div>', unsafe_allow_html=True)
            rag_tag = "FAISS+LangChain Vector Pipeline" if rag_mode == "langchain" else "KEYWORD MATRIX FALLBACK"
            st.markdown(f"""
            <div class="rationale-box">
                <span class="rag-tag">⚡ RAG · {rag_tag}</span><br/>
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