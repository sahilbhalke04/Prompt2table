# 📊 Unstructured Text-to-Dataset Converter

An automated information extraction and data engineering utility that transforms raw, messy text (such as AI chat transcripts, conversational bullet points, articles, and research summaries) into clean, schema-validated tabular datasets (`CSV` / `JSON`).

---

## 📌 Problem & Motivation

Large Language Models (LLMs) excel at generating synthesized textual information, but their outputs are typically conversational, narrative, or inconsistently formatted. Manually copying, formatting, and restructuring these answers into clean datasets for data analytics, machine learning, or spreadsheet workflows is repetitive and time-consuming.

This project solves that friction point by providing an end-to-end extraction pipeline that:
1. Accepts raw, unstructured text directly via copy-paste.
2. Infers or strictly maps entity attributes using LLMs and **Pydantic** structured schemas.
3. Automatically sanitizes numeric values and standardizes schema keys.
4. Produces validated, exportable datasets with zero manual data-entry overhead.

---

## ✨ Key Features

* **Dual Extraction Workflows:**
  * **Auto-Detect Schema:** Analyzes arbitrary paragraphs and autonomously determines relevant entity attributes and column headers.
  * **Custom Column Mode:** Enforces user-defined schema specifications (e.g., `model_name, price, engine_cc`).
* **Multi-Provider LLM Support:** Seamlessly switch between **Google Gemini** (`gemini-2.5-flash`) and **OpenAI** (`gpt-4o-mini`) via the sidebar interface.
* **Strict Type & Null Handling:** Enforces field types using Pydantic models to gracefully manage missing values (`None`/`null`) without pipeline crashes.
* **Automated Data Sanitization:**
  * Standardizes attribute headers to lowercase `snake_case`.
  * Sanitizes formatted strings containing currency and units (e.g., `"$1,299"` $\rightarrow$ `1299`, `"6.9-inch"` $\rightarrow$ `6.9`).
* **Interactive UI & Multi-Format Export:** Real-time DataFrame inspection in Streamlit with instant download options for **CSV** and **JSON**.

---

## 🏗️ System Architecture

```text
               ┌───────────────────────────────┐
               │    User Pastes Raw Text       │
               └───────────────┬───────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
   [ Auto-Detect Mode ]                  [ Custom Columns Mode ]
   (LLM infers attributes)               (Enforces user-specified keys)
            │                                     │
            └──────────────────┬──────────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │  Structured LLM Inference │
                 │    (Gemini 2.5 / GPT-4o)  │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │   Data Sanitization Engine │
                 │  (Snake_case & Num Parse) │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌───────────────────────────
                 │     Streamlit Preview     │
                 │   & CSV / JSON Exporters  │
                 └───────────────────────────┘
```
📁 Repository Structure
Plaintext
text_to_dataset/
│
├── auto_extractor.py      # Backend engine for dynamic schema inference
├── custom_extractor.py    # Backend engine for user-defined column extraction
├── exporter.py            # Data cleaning, DataFrame transformation, & export utilities
├── app.py                 # Streamlit web application & user interface
├── requirements.txt       # Project dependencies
└── .gitignore             # Ignored local files, environments, and secrets

🛠️ Tech Stack
Language: Python 3.10+

Frontend / Dashboard: Streamlit

Data Processing & Validation: Pandas, Pydantic

AI & LLM Providers: Google GenAI SDK, OpenAI API

🚀 Getting Started
1. Prerequisites
2. Python 3.10 or higher installed
3. An active API key from Google AI Studio or OpenAI.
4. Clone the Repository:
   Bashgit clone [https://github.com/sahilbhalke04/text-to-dataset-converter.git](https://github.com/sahilbhalke04/text-to-dataset-converter.git)
   cd text-to-dataset-converter
5. Set Up a Virtual Environment
   Bash# Create virtual environment
   python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Activate on macOS/Linux:
source .venv/bin/activate
4. Install DependenciesBashpip install -r requirements.txt
5. Launch the ApplicationBashstreamlit run app.py

💡 Example UsageInput (Raw Conversational Text)"The Tesla Model S Plaid comes with a range of 396 miles, costs roughly $89,990, and hits 0-60 mph in 1.99 seconds. Meanwhile, the Lucid Air Grand Touring offers 516 miles of range, retails for $109,900, and does 0-60 in about 3.0 seconds."Output (Extracted Tabular DataFrame)model_namerange_milesprice_usdzero_to_sixty_secTesla Model S Plaid396899901.99Lucid Air Grand Touring5161099003.0

👥 Contributors
Sahil Bhalke - Core Architecture, Schema Inference Engine & UI Integration Project 
Manasi Ghadge - Custom Column Logic, Data Sanitization & Export Utilities
