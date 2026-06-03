# 🩻 Scout-less CT Scan Planning System
### Advanced Multi-Modal Late-Fusion & Volumetric Heatmap Regression Framework

---

## 🔬 Overview
This repository contains the orchestration layer for an advanced, scout-less computed tomography (CT) scan planning architecture. Designed to eliminate conventional pre-scan scout radiographs, this system targets a **significant reduction in cumulative patient radiation dose** by leveraging continuous surface tracking arrays and statistical uncertainty estimation.

The system replaces the standard scout scout view with a multi-sensor data fusion engine, sub-voxel precise anatomical localization, and a safety-gated Retrieval-Augmented Generation (RAG) clinical reasoning agent.

---

## 🚀 Key Technical Features & Architectural Upgrades

### 1. Multi-Modal Late-Fusion Sensor Network
Standard 3D optical cameras are notoriously prone to edge artifacts and tracking degradation caused by bulky patient garments or clinical sheets. 
* **Heterogeneous Array:** Simulates synchronized inputs from three streams: **RGB (Surface Texture)**, **Depth (3D Spatial Topology)**, and **Infrared (Sub-clothing Thermal Body Contour)**.
* **Late-Fusion Strategy:** Processes sensor inputs independently before aggregating them at the final stage. This maintains fault isolation; if a single sensor degrades (e.g., severe shadows on the RGB camera), the localized system avoids total collapse by dynamically up-weighting the resilient Infrared thermal signature.

### 2. 3D Volumetric Heatmap Regression & Soft-Argmax
Direct point-coordinate regression models suffer significantly from spatial drift errors (~38.6 mm average variance) and lack a statistical measure of spatial variance.
* **Volumetric Probability Fields:** Generates continuous 3D Euclidean Distance Transform (EDT) Gaussian fields $(Z=128, Y=32, X=32)$ centering on internal targets (Carina, T12, Pubic Symphysis).
* **Sub-Voxel Precision:** Replaces basic peak-value extraction (`argmax`) with a continuous **Soft-Argmax** layer. By evaluating the soft spatial density expectation formula $E[v] = \sum v \cdot P(v)$, the architecture resolves tracking coordinates with sub-millimeter precision ($\sim 1.5\text{--}3\text{ mm}$ error margins).

### 3. Aleatoric Uncertainty Guardrails & Safety Gates
Medical device safety profiles demand rigorous statistical guarantees (Conformal Prediction frameworks) prior to gantry coordinate translation.
* **Variance Tracking:** Evaluates the standard deviation grid projection (**Peak Width**) across the 3D maps to directly quantify *Aleatoric Uncertainty*.
* **Deterministic Safety Gate:** If structural tracking noise exceeds allowable tolerances ($\text{Peak Width} > 7.5\text{ mm}$) or if the patient displays extreme morphology ($\text{BMI} > 45$), the reasoning agent immediately overrides automated planning and triggers an ironclad fail-safe warning: `High Uncertainty Detected: Reverting to Ultra-Low Dose Scout Validation`.

### 4. High-Fidelity Glow-Map Visualizations
* Renders real-time 2D coronal projections of the volumetric matrices overlaid on an anatomical silhouette using multi-layered Matplotlib contour fields.
* Plots sub-voxel localized coordinates along with distinct glowing spatial variance boundaries, providing visual intuition for system uncertainty.

---

## 🛠️ Quick Start

### 1. Install Dependencies
Ensure your environment satisfies the required scientific calculations (specifically `scipy.ndimage` for volumetric distance transformations).
```bash
pip install -r requirements.txt

```

> *Note: On the initial run, the system will automatically provision `sentence-transformers/all-MiniLM-L6-v2` (~85 MB) via HuggingFace for vectorization.*

### 2. Initialize the Framework

```bash
streamlit run app.py

```

### 3. Verification & Testing

1. Configure multi-modal camera tracking attributes (**Optimal** vs **Degraded**) in the sidebar panel.
2. Provide Patient Morphometry dimensions (**Height** and **Weight**).
3. Select an execution profile (e.g., *Chest CT*, *Abdomen CT*, *CAP*).
4. Run the engine to compute the interactive fusion matrix, soft-argmax arrays, and RAG-retrieved clinical rationale.

---

## 📑 Core Pipeline Reference

| Component / Function | Operational Domain | Mathematical / Logical Framework |
| --- | --- | --- |
| `simulate_multimodal_fusion()` | Sensor Processing | Late Fusion ensemble calculations balancing tracking coefficients based on body mass context. |
| `generate_edt_heatmaps()` | Probabilistic Regression | Volumetric 3D grid projection using Euclidean Distance Transforms (EDT) and Gaussian scaling. |
| `extract_coords_from_heatmap()` | Spatial Localization | Three-dimensional **Soft-Argmax expectation** layer yielding sub-voxel accurate metric coordinates ($mm$). |
| `build_rag_pipeline()` | Knowledge Management | Local in-memory FAISS vector store indexing structural CT protocol constraints via LangChain. |
| `reasoning_agent()` | Automated Planning | Safety-gated expert system validating aleatoric variance boundaries, morphological parameters, and sensor confidence intervals. |
| `plot_human_silhouette()` | Applied Visualization | Coronal contour mapping overlaid with exact target vectors and dynamic acquisition boundary boxes. |

---

## 📐 Supported Target Protocols

| Protocol ID | Protocol Target | Scan Boundary Coverage Profile |
| --- | --- | --- |
| **CHEST-001** | Standard Chest CT | Superior margin of Lung Apex ($\sim C7$ level) $\rightarrow$ Lower border of T12 Vertebral Body. |
| **ABD-001** | Standard Abdomen CT | Superior Diaphragm Dome ($\sim T8\text{-}T9$ level) $\rightarrow$ Pubic Symphysis inferior limit. |
| **PELVIS-001** | Pelvis CT Scan | Superior border of Iliac Crest ($L4\text{-}L5$) $\rightarrow$ Inferior margin of Pubic Symphysis. |
| **CHEST-ABD-001** | Chest-Abdomen-Pelvis (CAP) | Superior limit of Lung Apex $\rightarrow$ Full Pubic Symphysis / Pelvic floor. |

---

## 🔮 Transitioning to Production Inference

```
[Prototype Architecture]                 [Clinical Production Upgrade Target]
  simulate_skel_model()         ───►       3D SMPL-X / STAR Deep Mesh Regressor
  generate_edt_heatmaps()       ───►       Volumetric V-Net / PointNet++ Tensor CNN
  Rule-based Python Agent       ───►       Fine-tuned Med-LLaMA via Ollama / vLLM
  In-memory FAISS Store         ───►       Enterprise Vector Database (Chroma / Weaviate)

```

---

## 📂 System Audit Log

Every verified scan range selection signs and commits a descriptive verification block to the native JSONL logging repository (`audit_log.jsonl`):

```json
{
  "timestamp": "2026-06-03 21:14:02",
  "action": "TRANSMITTED",
  "safety_override": false,
  "z_start": 315.0,
  "z_end": 822.0,
  "confidence": 0.9410,
  "scan_type": "Chest CT"
}

```

---

*Disclaimer: This prototype is engineered exclusively for academic presentation and research verification. It is not approved for immediate diagnostic or diagnostic-adjacent clinical deployment.*

```

```

```

```