"""# 🩻 Scout-less CT Scan Planning System
### Advanced Multi-Modal Late-Fusion & Volumetric Heatmap Regression Framework

---

## 🔬 Overview & Clinical Significance

In traditional Computed Tomography (CT), planning an acquisition range requires taking a preliminary, low-dose projection radiograph called a **Scout View** (also known as a surview, topogram, or scanogram). While the scout view itself features lower radiation than a full volumetric scan, it exposes patients to unnecessary ionizing radiation before diagnostic imaging even starts. Furthermore, misinterpretation or manual alignment of the scout range can lead to significant **spatial drift errors**, leading to anatomy clipping or unnecessary over-scanning.

The **Scout-less CT Scan Planning System** replaces the physical pre-scan scout radiograph with an intelligent multi-modal tracking and automated expert reasoning network. By continuously analyzing the patient's outer topography using clinical optical arrays, the system performs sub-voxel internal landmark localization and suggests highly precise scan boundaries. 

### 🎯 Strategic Clinical Engineering Targets
* 📉 **Radiation Dose Minimization:** Eliminates 100% of the scout-view radiation burden, which is highly significant for pediatric patients, oncology monitoring, and longitudinal screening protocols.
* 🛡️ **Anatomical Edge Protection:** Utilizes conformal probability distributions to ensure zero clipping of vital marginal tissues (e.g., preserving full lung apices in chest acquisitions).
* ⚙️ **Automated Gantry Alignment:** Translates sub-voxel physical surface parameters directly into hardware coordinates ($Z$-axis boundaries in millimeters).

---

## 🏛️ System Architecture Diagram


             +-------------------------------------------------------------+
             |               MULTI-MODAL SURFACE SENSORS                   |
             |  [RGB Camera]          [Depth Camera]          [IR Camera]  |
             +-------+----------------------+----------------------+-------+
                     |                      |                      |
                     v                      v                      v
             +-------------------------------------------------------------+
             |            LATE-FUSION CONFIDENCE MATRIX                    |
             |      Processes independent streams & dynamically balances    |
             |          sensor weightings based on patient dress.          |
             +------------------------------+------------------------------+
                                            |
                                            v
             +-------------------------------------------------------------+
             |          3D VOLUMETRIC HEATMAP REGRESSION ENGINE            |
             |     Generates continuous 3D Euclidean Distance Transform     |
             |         (EDT) probability fields centered on targets.       |
             +------------------------------+------------------------------+
                                            |
                                            v
             +-------------------------------------------------------------+
             |                  SOFT-ARGMAX SPATIAL LAYER                  |
             |       Resolves continuous centers-of-mass to bypass         |
             |            discrete sub-voxel coordinate drift.             |
             +------------------------------+------------------------------+
                                            |
                                            v
             +-------------------------------------------------------------+
             |              KNOWLEDGE ACQUISITION LAYER (RAG)              |
             |   Retrieves anatomical scanning guidelines & specific offsets |
             |             from the clinical KB using FAISS embeddings.     |
             +------------------------------+------------------------------+
                                            |
                                            v
             +-------------------------------------------------------------+
             |          SAFETY-GATED EXPERT REASONING LLM                  |
             |     Evaluates aleatoric tracking variance & morphology.     |
             |      Triggers automatic manual override if threshold fails. |
             +------------------------------+------------------------------+
                                            |
                                            v
             +-------------------------------------------------------------+
             |                  CLINICAL VISUAL INTERFACE                  |
             |    Dynamic anatomical silhouette adjusts to Sex/BMI. Maps   |
             |     glowing probability blobs & outputs locked gantry data. |
             +-------------------------------------------------------------+

---

## 🚀 Key Technical Features Deep-Dive

### 1. Multi-Modal Late-Fusion Sensor Network
Optical tracking systems in medical theater environments suffer significant accuracy degradation when patients wear thick clinical gowns or blankets. 
* **The Heterogeneous Array:** The architecture tracks three concurrent dimensions: **RGB Surface Texture**, **Depth Spatial Topology**, and **Infrared Sub-clothing Thermal Contours**.
* **Late-Fusion Topology:** Rather than projecting early pixel-level mixtures, which propagates noise, features are processed in isolated pathways. If the RGB or Depth trackers degrade due to severe ambient shadow or complex cloth draping, the framework isolates the failure, down-weights the optical pipelines, and dynamically up-weights the resilient Infrared contour signature.

### 2. 3D Volumetric Heatmap Regression via Soft-Argmax
Directly regressing absolute point-coordinates via typical convolutional backbones leads to sharp spatial drift vulnerabilities near target margins and lacks a statistical confidence measure.
* **Volumetric Probability Fields:** The engine transforms absolute prediction maps into dense continuous 3D Euclidean Distance Transform (EDT) Gaussian fields $(Z=128, Y=32, X=32)$ surrounding inner anatomical markers (Carina anchor, T12 junction, Pubic Symphysis).
* **Continuous Sub-Voxel Localization:** Standard implementations extract points using a discrete `argmax`, limiting spatial accuracy to voxel grid steps. This framework introduces a continuous **Soft-Argmax** expectation formula layer:
  $$\mathbb{E}[v] = \sum_{v} v \cdot P(v)$$
  This math layer calculates the true center of mass of the probability envelope, ensuring highly precise structural readings with sub-millimeter tracking variance.

### 3. Aleatoric Uncertainty Guardrails & Safety Gates
Clinical execution engines require strict safety criteria before translating coordinates to physical equipment.
* **Variance Projections:** By continuously monitoring the statistical standard deviation (**Peak Width**) of the volumetric 3D maps, the planning engine captures real-time *Aleatoric Uncertainty* caused by patient motion or occlusion.
* **Deterministic Fail-Safe:** If tracking variance breaches absolute tolerances ($\text{Peak Width} > 7.5\text{ mm}$) or if the patient displays extreme morphology ($\text{BMI} > 45$), the expert system bypasses automated planning, triggers a protective alert, and initiates a manual override loop: `High Uncertainty Detected: Reverting to Ultra-Low Dose Scout Validation`.

### 4. Patient Adaptive Glow-Map Visualizations
The visual interface provides real-time verification mapping:
* **Morphological Scaling:** The embedded rendering matrix dynamically modifies body thickness, shoulder-to-hip ratios, and anatomical structures based on selected biological sex parameters and target body-mass distributions.
* **Volumetric Coronal Overlay:** Projects complex internal multidimensional tensors as smooth 2D Matplotlib glowing confidence contours, matching internal target fields.

---

## 📦 Repository Structure


```

├── app.py               # Main multi-modal orchestration layer & Streamlit UI
├── test_cases.py        # Automated validation suite (Regression & Monotonicity tests)
├── README.md            # Comprehensive architecture guide & implementation manual
└── requirements.txt     # Scientific calculations & orchestration dependencies

```

---

## 🛠️ Step-by-Step Implementation Guide

### 📋 Prerequisites & Local Setup

#### Step 1: Clone the Project Repository
```bash
git clone <your-repository-url>
cd scoutless-ct-planner

```

#### Step 2: Establish an Isolated Virtual Environment

It is highly recommended to isolate the package dependencies using a native Python virtual environment.

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate

```

#### Step 3: Install Required Dependencies

The engine relies heavily on scientific calculation tools like `scipy.ndimage` for distance fields and LangChain text-embedding infrastructure.

```bash
pip install -r requirements.txt

```

> *Technical Note: On the initial application launch, the HuggingFace engine will automatically provision `sentence-transformers/all-MiniLM-L6-v2` (~85 MB) locally on your system CPU to handle semantic vector lookups.*

---

### 🧠 Configuring the Clinical Reasoning Engine

The system supports a multi-tier LLM execution architecture via LangChain providers. You can choose from three execution backends within the user interface:

#### Option A: Zero-Cost Local Inference via Ollama (Recommended for Demos)

To run the advanced Llama 3 medical reasoning chain completely locally without external network requests:

1. Download and install **Ollama** from [ollama.com](https://ollama.com).
2. Download and activate the Llama 3 model weights inside your terminal:
```bash
ollama run llama3

```


3. Ensure Ollama is running in the background at `http://localhost:11434`.

#### Option B: Ultra-Fast Cloud Inference via Groq

To deploy high-throughput planning validation using cloud hosted infrastructure:

1. Obtain a free API token from the [Groq Console](https://console.groq.com/).
2. Provide the key string directly into the **Groq API Key** secure input field within the application sidebar.

#### Option C: Deterministic Rule-Based Fallback Engine

If no Large Language Model infrastructure is active, the engine seamlessly reverts to its native Python expert rule-set without crashing.

---

### 🎮 Running the Streamlit Interface

Launch the orchestration console by executing:

```bash
streamlit run app.py

```

Once deployed, the terminal will output a local network address (typically `http://localhost:8501`). Open this link in any modern web browser.

---

## 🧪 Automated Testing & Verification Suite

The repository features a highly detailed, programmatic validation script (`test_cases.py`) that executes boundary, crash, and mathematical integrity checks across multiple patient categories (normal, extreme heights, high BMI scales, Indian population morphometry baselines).

Execute the test pipeline directly inside your active environment terminal:

```bash
# Windows
python test_cases.py

# macOS / Linux
python3 test_cases.py

```

### 🔬 Validations Performed by the Test Suite

1. **Physical Span Verifications:** Validates that calculated $Z$-axis bounds fall within strict diagnostic limitations across all scan variants (e.g., *Pelvis CT* remains within 150–250 mm; *CAP* remains within 550–1350 mm).
2. **$Z$-Ordering Checks:** Evaluates that the planned Start coordinate never overlaps or crosses past the Planned End coordinate ($\text{End } Z > \text{Start } Z$).
3. **Linearity Scaling Test:** Verifies that landmark locations adjust linearly relative to changes in patient height configurations.
4. **Uncertainty Monotonicity Check:** Ensures system confidence scores follow a steady monotonic decline as input body mass and noise dimensions expand.

---

## 📑 Programmatic Pipeline Reference

### Core Architecture Callouts

| Target Function Name | Operation Domain | Logic Framework & Scientific Calculation |
| --- | --- | --- |
| `simulate_multimodal_fusion()` | Sensor Processing | Applies late-fusion equations to independently evaluate RGB, Depth, and Infrared matrices based on physical clothes damping factors. |
| `generate_edt_heatmaps()` | Probabilistic Regression | Maps discrete landmark coordinate nodes into full continuous 3D matrices using 3D Euclidean Distance Transforms (EDT) and Gaussian filters. |
| `extract_coords_from_heatmap()` | Spatial Localization | Runs an advanced **Soft-Argmax center-of-mass** equation down the probability maps to achieve sub-voxel precise positioning values ($mm$). |
| `build_rag_pipeline()` | Knowledge Management | Deploys an in-memory FAISS document vector database using LangChain, embedding clinical scanning guidelines for real-time reference retrieval. |
| `reasoning_agent()` | Automated Planning | Safety-gated decision matrix validating tracking uncertainty, patient parameters, and structural compliance filters. |
| `plot_human_silhouette()` | Applied Visualization | Real-time Matplotlib coronal mapping script displaying custom body outlines, organic placements, and glowing confidence ranges. |

### Supported Clinical Protocol Catalog

| Protocol ID | Protocol Definition | Core Scan Range Boundary Profile |
| --- | --- | --- |
| **CHEST-001** | Standard Chest CT | Targets pulmonary characterization. From 2 cm superior to lung apex ($\sim C7$ level) down to the lower border of the T12 Vertebral Body. |
| **ABD-001** | Standard Abdomen CT | Targets hepatic/renal evaluation. From the superior margin of the Diaphragm Dome ($\sim T8\text{-}T9$) down to the Pubic Symphysis inferior border. |
| **PELVIS-001** | Pelvis CT Scan | Targets pelvic fractures/trauma. From the superior boundary of the Iliac Crest ($L4\text{-}L5$) down to the inferior margin of the Pubic Symphysis. |
| **CHEST-ABD-001** | Chest-Abdomen-Pelvis (CAP) | Full oncology staging. Combines upper apex bounds down past the absolute pelvic floor with conservative safety offsets. |

---

## 📋 System Audit Logs & Medical Handoff

Every plan confirmed within the console signs, formats, and appends a structured verification block directly into a local transaction tracking repository (`audit_log.jsonl`) for validation tracking:

```json
{
  "timestamp": "2026-06-04 11:32:15", 
  "action": "TRANSMITTED", 
  "safety_override": false, 
  "z_start": 342.5, 
  "z_end": 820.0
}

```

---

## 🔮 Production Scalability Blueprint

To scale this educational planning prototype to production-grade physical scanner networks, replace the simulation logic with clinical-grade hardware links:

```
+------------------------------------+       +------------------------------------+
|       PROTOTYPE COMPONENTS         |       |    PRODUCTION INFRASTRUCTURE       |
+------------------------------------+       +------------------------------------+
| • simulate_skel_model()            | ───►  | • 3D SMPL-X Deep Mesh Surface Model|
| • generate_edt_heatmaps()          | ───►  | • PointNet++ V-Net Tensor CNN      |
| • Rule-Based / Local LLM Agent     | ───►  | • Fine-Tuned Med-LLaMA via vLLM    |
| • In-Memory FAISS Vector Store     | ───►  | • Dedicated Enterprise Database     |
+------------------------------------+       +------------------------------------+

```

---

*Disclaimer: This prototype is engineered exclusively for academic presentation, development showcase, and research verification. It is not approved for immediate real-world diagnostic deployment or hardware-level clinical implementation.*
"""

