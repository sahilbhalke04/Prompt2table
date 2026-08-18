# Text-to-Dataset Converter

Turn messy unstructured text, bullet points, or raw notes into clean, structured tables (CSV / JSON) using LLMs.

---

## What It Does

* **Auto or Custom Schema:** Let the LLM detect columns automatically, or specify the exact column names you want.
* **Auto-Fill Missing Data:** Uses AI to impute missing fields without breaking existing values.
* **Model Agnostic:** Works with Google Gemini and OpenAI API keys.
* **Clean Data Types:** Automatically strips messy characters (`$`, `₹`, `GB`, `mAh`) into raw numbers and handles `null` values gracefully.
* **One-Click Export:** Download results instantly as CSV or JSON.

---

## Quick Start

### 1. Clone & Set Up
```bash
git clone [https://github.com/sahilbhalke04/text-to-dataset-converter.git](https://github.com/sahilbhalke04/text-to-dataset-converter.git)
cd text-to-dataset-converter
```

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # On Windows
# source .venv/bin/activate # On Mac/Linux

pip install -r requirements.txt

### 2. Run the AppBashstreamlit run app.py
Project StructurePlaintexttext_to_dataset/
├── auto_extractor.py      # Auto-detects columns from unstructured text
├── custom_extractor.py    # Extracts user-defined columns
├── exporter.py            # Data cleaning, null handling, & file downloads
├── app.py                 # Streamlit UI
└── requirements.txt

Contributors:Sahil Bhalke — Architecture, Auto Extraction & UIManasi Ghadge — Custom Extraction, Data Cleaning & Exporter
