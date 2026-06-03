# 🩻 Scout-less CT Scan Planning System
**AI-Assisted Scan Range Determination**

---

## Overview
This Streamlit application is the **orchestration layer** of a scout-less CT scan
planning system. It replaces the conventional scout radiograph with:
1. A parametric **3D skeletal model** (mocked as an 85-D parameter vector).
2. A **neural-network landmark predictor** (mock regression model).
3. A **LangChain RAG pipeline** with FAISS vector store for CT protocol retrieval.
4. A **reasoning agent** that computes exact Z-axis scan boundaries.
5. A **human-in-the-loop approval** interface for the radiographer.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

> First run downloads `sentence-transformers/all-MiniLM-L6-v2` (~85 MB) from HuggingFace.

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Usage
1. Upload a "patient scan" (any image / mesh file — mocked in prototype).
2. Enter **Height** and **Weight** in the sidebar.
3. Select **Scan Type** (Chest, Abdomen, Pelvis, or CAP).
4. Click **▶ RUN AI PLANNER**.
5. Review AI output: landmarks, Z-boundaries, silhouette, confidence, rationale.
6. Click **Confirm & Execute Scan** to log technician approval.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI                      │
│   Sidebar (Input)          Main Panel (Output)      │
└──────────┬──────────────────────────┬──────────────┘
           │                          │
   ┌───────▼────────┐        ┌────────▼───────────┐
   │ simulate_skel_ │        │  Decision UI       │
   │ model()        │        │  • BMI Dashboard   │
   │ 85-D vector    │        │  • Silhouette Plot │
   └───────┬────────┘        │  • Confidence Bar  │
           │                 │  • RAG Rationale   │
   ┌───────▼────────┐        │  • HitL Confirm    │
   │ simulate_nn    │        └────────────────────┘
   │ Landmark()     │                  ▲
   │ 3D coords      │                  │
   └───────┬────────┘        ┌─────────┴──────────┐
           │                 │  reasoning_agent()  │
           └────────────────►│  Z-start / Z-end   │
                             │  calc + rationale  │
                             └─────────┬──────────┘
                                       │
                             ┌─────────▼──────────┐
                             │  LangChain RAG     │
                             │  FAISS vectorstore │
                             │  HuggingFace embed │
                             └────────────────────┘
```

---

## Key Modules

| Function | Description |
|---|---|
| `simulate_skel_model(h, w)` | Returns 85-D numpy float32 vector encoding BMI + body proportions |
| `simulate_nnLandmark(h, w)` | Returns 3D (x,y,z) mm coords for Sternum, T12, Pubic Symphysis |
| `build_rag_pipeline()` | Builds FAISS index from CT_PROTOCOL_KB using MiniLM embeddings |
| `retrieve_protocol(query)` | Similarity-searches the FAISS index for relevant protocol chunks |
| `reasoning_agent(...)` | Computes Z-boundaries and medical rationale from landmarks + protocol |
| `plot_human_silhouette(...)` | Matplotlib figure: schematic body outline + scan range markers |

---

## Extending to Real Inference

| Component | Mock (Prototype) | Production Upgrade |
|---|---|---|
| Skeletal model | `simulate_skel_model()` | 3D-SMPL / STAR mesh regressor |
| Landmark detector | `simulate_nnLandmark()` | PointNet / DNN on CT/surface scan |
| Embeddings | `all-MiniLM-L6-v2` | OpenAI `text-embedding-3-small` or MedCPT |
| Reasoning LLM | Rule-based Python | Llama 3 via Ollama (`langchain-ollama`) |
| Vector store | FAISS in-memory | Chroma / Weaviate / Pinecone |
| Knowledge base | Hard-coded string | Full DICOM SR / ACR protocol library |

### Enabling Llama 3 (optional)
```bash
# Install Ollama: https://ollama.com
ollama pull llama3
pip install langchain-ollama
```
Then replace `reasoning_agent()` with a `ChatOllama` chain call.

---

## Audit Log
Each confirmed scan writes a JSONL entry to `/tmp/ct_planner_audit.jsonl`:
```json
{
  "timestamp": "2024-01-15 14:32:01",
  "action": "APPROVED",
  "protocol": "CHEST-001",
  "z_start": 312.5,
  "z_end": 634.2,
  "confidence": 0.91,
  "patient": {"height": 170, "weight": 75, "bmi": 25.95, "age": 45},
  "scan_type": "Chest CT"
}
```

---

## Supported Scan Types

| Protocol ID | Name | Coverage |
|---|---|---|
| CHEST-001 | Chest CT | Lung apex → T12 |
| ABD-001 | Abdomen CT | Diaphragm → Pubic symphysis |
| PELVIS-001 | Pelvis CT | Iliac crest → Ischial tuberosities |
| CHEST-ABD-001 | CAP CT | Lung apex → Pubic symphysis |

---

*Prototype for academic/research use only. Not for clinical deployment.*
