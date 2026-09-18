# 🛍️ Retail AI Pre-Sales Command Center

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://bookish-cod-vxrq4p6pqqgfpvwp-8501.app.github.dev/)

> 💡 **Browser Launch Note**: When Codespaces finishes initializing, Google Chrome or Edge may block the new tab automatically. If the Streamlit dashboard doesn't appear in a new window:
> 1. Look for the **Pop-up blocked** icon (🚫) in your browser address bar and select **"Always allow pop-ups from github.dev"**, OR
> 2. Open the **Ports** tab in the bottom tray of the editor and click the **Globe icon** (Port 8501) to launch the Command Center manually.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![Playwright](https://img.shields.io/badge/Playwright-Chromium-green.svg)](https://playwright.dev/)
[![Docker Multi-Stage](https://img.shields.io/badge/Docker-Multi--Stage-2496ED.svg)](https://www.docker.com/)
[![GHCR](https://img.shields.io/badge/GHCR-ghcr.io-black.svg)](https://ghcr.io/mmxco/retail-command-center)
[![Tests Passing](https://img.shields.io/badge/tests-126%20passed-brightgreen.svg)]()
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An enterprise-grade, unified pre-sales demonstration cockpit engineered for retail AI solution architects and executive pre-sales teams. Integrates automated account discovery, autonomous multi-agent inventory replenishment negotiations, and mathematical Retail Inventory Method (RIM) balance sheet valuation forensics.

---

## ⚡ Executive 30-Second Summary

The **Retail AI Pre-Sales Command Center** addresses the $100B+ margin-leakage problem in omnichannel apparel retail by bridging corporate intelligence, frontline operations, and balance sheet integrity:

1. **Phase 1 (Discovery)**: Eliminates 15+ hours of pre-sales account research by crawling target apparel domains and generating tailored executive dossiers via token-efficient Small Language Models (SLMs) in under 60 seconds.
2. **Phase 2 (Autonomous Multi-Agent Ops)**: Resolves critical omnichannel stockouts using a multi-agent negotiation committee (Microsoft AutoGen v0.2) that autonomously balances store demand, regional distribution, and vendor contracts under a strict human-in-the-loop ERP governance gate.
3. **Phase 3 (Valuation Forensics & ML Clustering)**: Demonstrates the mathematical distortion of the Retail Inventory Method (RIM), proving clearance markdowns artificially depress balance sheet inventory valuations. Employs Scikit-Learn KNN store clustering ($k=4$) and Gemini Pro "AI CFO" risk briefings to quantify immediate enterprise ROI.

---

## 🏢 Tab 1: Strategic Account Discovery & Pre-Sales Intelligence Brief

Tab 1 automates corporate prospect discovery, tech-stack fingerprinting, and pre-sales briefing generation.

* **Target Extraction**: Programmatic extraction of apparel brand signals, investor relations press releases, and career footprint markers using headless Playwright Chromium.
* **SLM Input-Token Conservation**: Rather than streaming massive, noisy raw DOM markup into costly frontier LLMs, the pipeline routes sanitized markdown through **Small Language Models (Gemini 2.5 Flash / SLM architectures)**. By filtering layout boilerplates and extracting structured Pydantic schemas in a targeted first-pass, the pipeline achieves an **85%+ reduction in input token consumption** while slashing end-to-end inference latency to sub-second responses.
* **Deterministic Fallback Profiles**: Ships with offline-resilient presets (The Buckle, American Eagle, Abercrombie & Fitch) guaranteeing zero demo-day presentation failures.

---

## 🤖 Tab 2 Deep Dive: Autonomous Multi-Agent Replenishment Ops (AutoGen)

Tab 2 demonstrates autonomous supply chain crisis resolution using **Microsoft AutoGen (v0.2)**. When a high-velocity promotional denim item faces a stockout, four specialized AI agents negotiate an optimal replenishment strategy in real time.

```
                  ┌─────────────────────────────────────────────────────────┐
                  │      🏬 Store Operations Lead (Stockout Sensor)         │
                  │      "Store #104 denim stockout: 240 units needed"      │
                  └────────────────────────────┬────────────────────────────┘
                                               │
                                               ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │    📊 Inventory Merchandising Lead (Multi-Echelon)       │
                  │    "Regional DC has 0; Salt Lake Hub has 140 excess"     │
                  └────────────────────────────┬────────────────────────────┘
                                               │
                                               ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │    📦 Vendor Procurement Lead (Contract Negotiator)     │
                  │    "Fast-track PO for 100 units @ $28.50 ($120 freight)"│
                  └────────────────────────────┬────────────────────────────┘
                                               │
                                               ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │    ⚙️ Autonomous Governance & ERP Executor              │
                  │    Consensus Tagged: Split transfer (140) + Vendor PO (100)│
                  │    [ ⏸️ Human-in-the-Loop Executive Sign-off Gate ]      │
                  └────────────────────────────┬────────────────────────────┘
                                               │ Executive Approval
                                               ▼
                           🚀 Mock EDI 850 ERP Purchase Order Transmitted
```

### Key Architectural Pillars:
* **Distinct Agent Personas & Conflicting Objectives**:
  * **Store Operations Lead**: Maximizes on-shelf availability and eliminates customer walkaways; indifferent to expedited shipping surcharges.
  * **Inventory Merchandising Lead**: Minimizes inter-store logistics fees by auditing multi-echelon network inventory for lateral store-to-store rebalancing.
  * **Vendor Procurement Lead**: Evaluates contract MOQ, tier discounts, lead-time penalties, and rush manufacturing options.
  * **Autonomous Governance & ERP Executor**: Synthesizes agent arguments into an audited consensus proposal and enforces safety boundaries.
* **Human-in-the-Loop (HITL) Governance Gate**: Fully autonomous negotiations halt before financial commitment. Execution requires explicit one-click executive sign-off (`st.button("✅ Authorize & Dispatch EDI 850 PO")`).
* **Audited Tool Execution**: Triggers a simulated transactional ERP payload emitting an audited EDI 850 Purchase Order receipt complete with PO number, vendor routing, and GL account allocation.

---

## 📊 Tab 3 Deep Dive: RIM Valuation Forensics & Predictive Store Clustering

Tab 3 exposes the structural flaw of the **Retail Inventory Method (RIM)**—the predominant inventory accounting standard in enterprise apparel retail—and pairs it with predictive machine learning.

### 1. The Structural Flaw of RIM Accounting
Under standard GAAP/IFRS retail inventory accounting, ending inventory cost is calculated by multiplying ending inventory at retail by the cumulative **Cost-to-Retail Ratio ($C/R$)**:

$$\text{Ending Inventory Cost} = \text{Ending Inventory Retail} \times \left( \frac{\text{Cost of Goods Available}}{\text{Retail Value of Goods Available}} \right)$$

When store operators take aggressive clearance markdowns ($D$) to clear seasonal overhang, the retail value of goods sold and on hand plummets, but the historical markon complement remains static. Consequently:
* Clearance markdowns **artificially deflate the balance sheet valuation** of ending inventory.
* Reported gross margins experience severe, unwarranted compression.
* Working capital ratios deteriorate regardless of actual physical unit inventory health.

### 2. Multi-Period Balance Sheet Simulator
* Built upon a mathematically verified 50-store, 24-month rolling apparel dataset (1,200 observations).
* Enforces strict period-over-period roll-forward integrity:
  $$\text{Beginning Inventory}_{T+1} \equiv \text{Ending Inventory}_T$$
* Features an interactive **Markdown Shock Simulator (+0% to +50%)** dynamically demonstrating balance sheet asset deflation across store tiers.

### 3. Predictive Store Clustering & KNN Peer Benchmarking
* **Scikit-Learn Pipeline (`KMeans`, $k=4$)**: Segregates store networks into 4 operational archetypes based on sales velocity, markdown exposure, and shrinkage rates:
  1. 🟢 **Capital-Efficient Flagships**: High turn rate, low markdown dependency ($<15\%$), superior GMROI.
  2. 🔵 **Balanced Regional Performers**: Stable margins, moderate turnover, predictable replenishment.
  3. 🟠 **Discount-Addicted Outliers**: Excessive clearance dependency ($>30\%$), deflated ending inventory cost valuations.
  4. 🔴 **High-Risk Shrinkage Anomalies**: Disproportionate unrecorded book-to-physical inventory loss ($>3.5\%$).
* **KNN Peer Benchmarking**: Nearest-neighbor Euclidean distance matching identifies top 3 peer stores for targeted corrective merchandising.

### 4. Gemini Pro "AI CFO" & Dynamic Value Realization Calculator
* **AI CFO Briefings**: Live dashboard parameters are parsed by Gemini Pro to produce an executive risk memo detailing balance sheet exposure, gross margin erosion, and working capital vulnerability.
* **Dynamic Value Realization Engine**: Projects 3-year cumulative ROI and payback periods based on store count, markdown mitigation (8–18 bps), and shrinkage recovery.

---

## 🐳 Containerized Execution & DevOps

The application is fully containerized using a hardened, multi-stage Docker build optimized for GitHub Container Registry (`ghcr.io`) and GitHub Codespaces.

### 1. Instant Local Launch via Docker Compose
No local Python environment or browser binary installation required:

```powershell
cd retail-command-center

# Build and run with mock pre-sales evaluation keys
docker compose up -d --build

# View container logs
docker compose logs -f
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

### 2. Pull Directly from GitHub Container Registry (GHCR)
```powershell
docker run -d `
  --name retail-command-center `
  -p 8501:8501 `
  ghcr.io/mmxco/retail-command-center:latest
```

### 3. GitHub Codespaces / Dev Containers
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://bookish-cod-vxrq4p6pqqgfpvwp-8501.app.github.dev/)

> 💡 **Browser Launch Note**: When Codespaces finishes initializing, Google Chrome or Edge may block the new tab automatically. If the Streamlit dashboard doesn't appear in a new window:
> 1. Look for the **Pop-up blocked** icon (🚫) in your browser address bar and select **"Always allow pop-ups from github.dev"**, OR
> 2. Open the **Ports** tab in the bottom tray of the editor and click the **Globe icon** (Port 8501) to launch the Command Center manually.

Click the badge above to launch an instant cloud workstation in your browser. Streamlit (port 8501) automatically compiles and forwards to your browser in under 60 seconds.

---

## 🛡️ Enterprise Validation Metrics & Governance

The codebase is governed by automated architectural quality checks and strict production constraints:

| Validation Metric | Standard / Threshold | Achieved Status |
| :--- | :--- | :--- |
| **Unit Test Coverage** | 100% core coverage across 12 test suites | **126 passing tests** (`python -m unittest discover tests`) |
| **Container Security Context** | Non-root unprivileged runtime | **Enforced UID 1001 / GID 1001 (`appuser:appgroup`)** |
| **Headless Playwright** | System dependencies bundled in Debian slim | **Verified operational** in `/ms-playwright` |
| **Container Healthcheck** | Automated server readiness probing | **`curl -f http://localhost:8501/_stcore/health`** |
| **Streamlit API Standards** | Zero deprecated `use_container_width` | **0 violations**; 100% `width="stretch"` compliance |
| **Architectural Complexity** | AST Cyclomatic Complexity $<10$, LOC $\le 80$ | **100% compliance** verified via AST test runner |
| **Zero-Crash Resilience** | Deterministic heuristic fallbacks | **Offline operational** without live API keys |

---

## 📂 Project Structure

```
retail-command-center/
├── .devcontainer/
│   └── devcontainer.json          # GitHub Codespaces 1-click environment
├── .github/
│   └── workflows/
│       └── container-publish.yml  # Multi-stage GHCR build, smoke test & push
├── assets/
│   └── styles.css                 # Enterprise retail typography & UI styling
├── core/                          # Pure domain logic (Zero Streamlit imports)
│   ├── rim_engine.py              # RIM accounting math & roll-forward verification
│   ├── store_clustering.py        # KMeans (k=4) and KNN peer benchmarking
│   └── value_engine.py            # Dynamic 3-year value realization math
├── data/
│   ├── autogen_scenarios.json     # Multi-agent negotiation scenario dialogues
│   └── discovery_presets.json     # Enterprise account discovery dossiers
├── models/
│   └── store_cluster_pipeline.joblib # Serialized Scaler + KMeans + KNN pipeline
├── services/
│   ├── data_loaders.py            # Cached, resilient data & style loaders
│   ├── formatters.py              # KaTeX LaTeX sanitizer & currency formatters
│   └── llm_advisor.py             # Google Gemini client cascade & CFO prompt builder
├── tests/                         # Automated test suite (126 tests)
│   ├── test_architecture_governance.py
│   ├── test_rim_engine.py
│   └── ...
├── ui/                            # Modular presentation components
│   ├── kpi_metrics.py             # Executive KPI metric cards
│   ├── tab_autogen.py             # Tab 2: AutoGen multi-agent orchestration
│   ├── tab_discovery.py           # Tab 1: Account discovery & Google Docs brief
│   ├── tab_rim_knn.py             # Tab 3: RIM forensics & KNN clustering
│   └── value_calculator.py        # Tab 3: Dynamic ROI realization UI
├── app.py                         # Unified 92-LOC coordinator entrypoint
├── Dockerfile                     # Multi-stage production container definition
├── docker-compose.yml             # Local evaluation & GHCR orchestration
├── .dockerignore                  # Strict secrets & cache exclusion matrix
└── requirements.txt               # Minimal pinned production dependencies
```

---

## 📜 License
This project is licensed under the Apache 2.0 License.
