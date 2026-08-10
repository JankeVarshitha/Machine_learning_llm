# Hugging Face Structured Data Benchmarks: Foundation Models vs. GBDTs

A high-performance experimentation suite evaluating the production trade-offs between modern **Zero-Shot Numerical Foundation Models** on the Hugging Face Hub vs. traditional **Gradient Boosted Decision Trees (GBDTs)**.

This repository implements cross-modality architectural benchmarks across two core data domains:
1. **Tabular Classification**: `google/tabfm-1.0.0-pytorch` vs. **XGBoost** (Dataset: California Housing)
2. **Time-Series Forecasting**: `google/timesfm-2.5-200m-pytorch` vs. **LightGBM** (Dataset: AEP Hourly Energy Consumption)

---

Modern numerical foundation models bypass the traditional text-serialization limits of early architectures like TabLLM or TimeLLM. Instead of flattening data into string templates (which destroys spatial relationships and blows up token counts), **Google TabFM** applies token-free structural cross-attention across row/column dimensions. Similarly, **TimesFM 2.5** leverages input patching and native scalar quantization to predict multi-seasonal trends in a single forward pass without dataset-specific fine-tuning.

This framework measures whether the cross-domain generalization of massive transformer backbones justifies the computational hardware latency tax compared to highly lightweight, compiled tree models.

---

##  Quick Start & Installation

### 1. Clone and Initialize Environment
```bash
# Clone the repository
git clone https://github.com
cd hf-foundation-benchmarks

# Create a clean virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install unified Hugging Face and ML dependencies
pip install -r requirements.txt
```

### 2. Execute Tabular Experimentation Protocol
```bash
python bench_tabular.py
```

### 3. Execute Time-Series Experimentation Protocol
```bash
python bench_timeseries.py
```

---

##  Benchmarking Vectors

The system automatically runs evaluation loops and logs data to stdout across two major criteria:

* **Predictive Performance**: Tracking Accuracy/F1-Score for tabular classification matrices, and Mean Absolute Error (MAE) for continuous time-series arrays.
* **MLOps Compute Latency**: Calculating exact execution overhead in milliseconds (`ms`) per batch on the localized CPU/GPU runtime thread.

*Note: Running the benchmark will automatically generate and cache comparison tracking plots (`hf_ts_benchmark.png`) in the root directory.*

---

##  Requirements Stack
The pipeline is verified on Python 3.10+ with the core integration components:
* `transformers` & `huggingface_hub` — Low-level model orchestration and layer weight caching.
* `xgboost` & `lightgbm` — Compiled baseline matrix operators.
* `torch` — Tensor computational backbone execution engine.

---