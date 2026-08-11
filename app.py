"""
app.py  —  Text-to-Dataset Converter  (Streamlit UI)
─────────────────────────────────────────────────────
Person A: Core UI shell — sidebar, text input area, mode toggle.
Person B: Custom column input, results table, download buttons.

Run:  streamlit run app.py
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os

# Load .env file if present
load_dotenv()

# ── Person A imports ──────────────────────────────────────────────────────────
from auto_extractor import extract_auto_dataset

# ── Person B imports (graceful fallback if not yet implemented) ────────────────
try:
    from custom_extractor import extract_custom_dataset
    _CUSTOM_AVAILABLE = True
except (ImportError, Exception):
    _CUSTOM_AVAILABLE = False

try:
    from exporter import convert_to_csv, convert_to_json
    _EXPORTER_AVAILABLE = True
except (ImportError, Exception):
    _EXPORTER_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Text → Dataset Converter",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — Premium Dark Theme
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Google Fonts ────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Root Variables ──────────────────────────────────────────────── */
    :root {
        --bg-primary:    #0a0e1a;
        --bg-secondary:  #111827;
        --bg-card:       #1a1f36;
        --border-color:  #2a2f45;
        --text-primary:  #e2e8f0;
        --text-secondary:#94a3b8;
        --accent-1:      #818cf8;
        --accent-2:      #6366f1;
        --accent-grad:   linear-gradient(135deg, #818cf8 0%, #6366f1 50%, #4f46e5 100%);
        --success:       #34d399;
        --warning:       #fbbf24;
        --error:         #f87171;
    }

    /* ── Global ──────────────────────────────────────────────────────── */
    html, body, .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    .stApp {
        background: var(--bg-primary) !important;
    }

    /* ── Sidebar ─────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--text-primary) !important;
    }

    /* ── Header banner ───────────────────────────────────────────────── */
    .hero-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #1e1b4b 100%);
        border: 1px solid rgba(129,140,248,.25);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .hero-banner h1 {
        background: linear-gradient(135deg, #c7d2fe, #818cf8, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .hero-banner p {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 300;
    }

    /* ── Cards ────────────────────────────────────────────────────────── */
    .ui-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .ui-card h3 {
        color: var(--accent-1) !important;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }

    /* ── Success / Error / Warning boxes ─────────────────────────────── */
    .success-box {
        background: rgba(52,211,153,.08);
        border: 1px solid rgba(52,211,153,.3);
        border-left: 4px solid var(--success);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        color: var(--success);
        margin: 0.5rem 0;
    }
    .error-box {
        background: rgba(248,113,113,.08);
        border: 1px solid rgba(248,113,113,.3);
        border-left: 4px solid var(--error);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        color: var(--error);
        margin: 0.5rem 0;
    }
    .warning-box {
        background: rgba(251,191,36,.08);
        border: 1px solid rgba(251,191,36,.3);
        border-left: 4px solid var(--warning);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        color: var(--warning);
        margin: 0.5rem 0;
    }
    .info-box {
        background: rgba(129,140,248,.08);
        border: 1px solid rgba(129,140,248,.3);
        border-left: 4px solid var(--accent-1);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        color: var(--accent-1);
        margin: 0.5rem 0;
    }

    /* ── Text area styling ───────────────────────────────────────────── */
    .stTextArea textarea {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', monospace !important;
        font-size: 0.9rem !important;
        transition: border-color 0.2s ease;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent-1) !important;
        box-shadow: 0 0 0 2px rgba(129,140,248,.2) !important;
    }

    /* ── Buttons ──────────────────────────────────────────────────────── */
    .stButton > button[kind="primary"],
    div.stButton > button:first-child {
        background: var(--accent-grad) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button:hover,
    div.stButton > button:first-child:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(99,102,241,.35) !important;
    }

    /* ── Radio buttons ───────────────────────────────────────────────── */
    .stRadio > label {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }

    /* ── Metrics ─────────────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1rem;
    }
    [data-testid="stMetricValue"] {
        color: var(--accent-1) !important;
    }

    /* ── DataFrames ──────────────────────────────────────────────────── */
    .stDataFrame {
        border-radius: 10px !important;
        overflow: hidden;
    }

    /* ── Expander ────────────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        color: var(--text-secondary) !important;
    }

    /* ── Footer ──────────────────────────────────────────────────────── */
    .app-footer {
        text-align: center;
        color: #475569;
        font-size: 0.75rem;
        padding: 1.5rem 0 0.5rem;
        border-top: 1px solid var(--border-color);
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialisation
# ─────────────────────────────────────────────────────────────────────────────

if "extraction_df" not in st.session_state:
    st.session_state.extraction_df = None
if "run_count" not in st.session_state:
    st.session_state.run_count = 0


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Person A's UI (Provider, Model, API Key)
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # ── Provider Selection ────────────────────────────────────────────────
    st.markdown("**1. LLM Provider**")
    provider = st.radio(
        "Choose Provider",
        ["🔵 Gemini (Google)", "🟢 OpenAI (GPT)"],
        index=0,
        key="provider_radio",
        label_visibility="collapsed",
    )
    provider = "gemini" if "Gemini" in provider else "openai"

    # ── Model Selection ──────────────────────────────────────────────────
    st.markdown("**2. Model**")
    if provider == "openai":
        model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
    else:
        model_options = [
            "gemini-3.6-flash",       # ✅ Active default model
            "gemini-flash-latest",    # ✅ Active alias
            "gemini-3.5-flash",       # ✅ Active model
            "gemini-pro-latest",      # ✅ Active high-quality model
        ]

    selected_model = st.selectbox(
        "Select Model",
        model_options,
        key="model_select",
        label_visibility="collapsed",
    )

    # ── API Keys ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**3. API Keys**")

    gemini_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AQ.Ab... or AIzaSy...",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Get a free key at https://aistudio.google.com/app/apikey",
        key="gemini_key_input",
    )
    st.caption("🔑 [Get Gemini Key](https://aistudio.google.com/app/apikey)")

    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        value=os.environ.get("OPENAI_API_KEY", ""),
        help="Get your key at https://platform.openai.com/api-keys",
        key="openai_key_input",
    )
    st.caption("🔑 [Get OpenAI Key](https://platform.openai.com/api-keys)")

    # ── Integration Status ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Module Status**")
    st.markdown(
        f"- Auto-Detect: ✅ Ready\n"
        f"- Custom Columns: {'✅ Ready' if _CUSTOM_AVAILABLE else '⏳ Pending (Person B)'}\n"
        f"- Exporter: {'✅ Ready' if _EXPORTER_AVAILABLE else '⏳ Pending (Person B)'}"
    )


# ═════════════════════════════════════════════════════════════════════════════
# MAIN AREA — Header
# ═════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class="hero-banner">
        <h1>🧬 Text → Dataset Converter</h1>
        <p>Paste raw unstructured text · Get clean, downloadable tables</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN AREA — Person A: Text Input & Mode Toggle
# ═════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="ui-card"><h3>📝 Paste Your Raw Text</h3></div>',
    unsafe_allow_html=True,
)

raw_text = st.text_area(
    "Paste your raw text here",
    height=280,
    placeholder=(
        "Paste any unstructured text here...\n\n"
        "Example: Copy a ChatGPT/Gemini response listing the top 10 fastest "
        "cars with model name, top speed, and price — then let the AI convert "
        "it into a clean dataset!"
    ),
    label_visibility="collapsed",
    key="raw_text_input",
)

# Character counter
if raw_text:
    char_count = len(raw_text)
    word_count = len(raw_text.split())
    st.caption(f"📊 {char_count:,} characters · {word_count:,} words")


# ── Extraction Mode Toggle (Person A's UI) ────────────────────────────────

st.markdown("---")

mode = st.radio(
    "🔀 Extraction Mode",
    ["🤖 Auto-Detect Columns", "✏️ Custom Columns"],
    index=0,
    horizontal=True,
    key="mode_radio",
    help=(
        "**Auto-Detect:** The AI figures out the best column names from your text.\n\n"
        "**Custom Columns:** You specify exactly which columns you want extracted."
    ),
)

# ═════════════════════════════════════════════════════════════════════════════
# Person B's UI Slot — Custom Column Input
# ═════════════════════════════════════════════════════════════════════════════

user_cols = ""
if "Custom" in mode:
    st.markdown(
        '<div class="ui-card">'
        '<h3>✏️ Define Your Columns</h3>'
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Person B: Replace or enhance this input field ─────────────────────
    user_cols = st.text_input(
        "Enter column names (comma-separated)",
        placeholder="e.g.  Name, Price, Top Speed, 0-60 Time",
        key="custom_cols_input",
        help="Type the exact column names you want in the output table.",
    )

    if not _CUSTOM_AVAILABLE:
        st.markdown(
            '<div class="warning-box">'
            "⏳ <strong>Custom Columns module not yet implemented.</strong> "
            "Person B: implement <code>custom_extractor.py</code> to enable this mode."
            "</div>",
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Convert Button & Pipeline
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("")  # spacing

convert_clicked = st.button(
    "🚀 Convert to Dataset",
    type="primary",
    use_container_width=True,
    key="convert_btn",
)

if convert_clicked:
    # ── Validation ────────────────────────────────────────────────────────
    if not raw_text or not raw_text.strip():
        st.markdown(
            '<div class="warning-box">⚠️ Please paste some text first!</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    # Check API key
    api_key = gemini_key if provider == "gemini" else openai_key
    if not api_key or not api_key.strip():
        st.markdown(
            f'<div class="warning-box">⚠️ Please enter your '
            f'{"Gemini" if provider == "gemini" else "OpenAI"} API key '
            f"in the sidebar.</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    # Check custom mode availability
    if "Custom" in mode and not _CUSTOM_AVAILABLE:
        st.markdown(
            '<div class="error-box">❌ Custom Columns mode is not yet available. '
            "Person B needs to implement <code>custom_extractor.py</code>.</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    if "Custom" in mode and not user_cols.strip():
        st.markdown(
            '<div class="warning-box">⚠️ Please enter column names for Custom mode.</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    # ── Run Pipeline ─────────────────────────────────────────────────────
    progress_bar = st.progress(0, text="Initialising...")
    status = st.empty()

    try:
        t_start = time.time()

        if "Auto" in mode:
            # ── Person A's Backend ────────────────────────────────────────
            status.info("🤖 Analysing text and auto-detecting schema...")
            progress_bar.progress(30, text="Calling LLM...")

            df = extract_auto_dataset(
                raw_text=raw_text,
                api_key=api_key,
                provider=provider,
                model=selected_model,
            )
        else:
            # ── Person B's Backend ────────────────────────────────────────
            status.info("✏️ Extracting with custom columns...")
            progress_bar.progress(30, text="Calling LLM...")

            df = extract_custom_dataset(
                raw_text=raw_text,
                columns=user_cols,
                api_key=api_key,
                provider=provider,
                model=selected_model,
            )

        elapsed = time.time() - t_start
        progress_bar.progress(100, text="Done!")
        status.empty()

        # ── Store results in session state ────────────────────────────────
        st.session_state.extraction_df = df
        st.session_state.run_count += 1

        if df is not None and not df.empty:
            st.markdown(
                f'<div class="success-box">'
                f"✅ Extracted <strong>{len(df)}</strong> records "
                f"with <strong>{len(df.columns)}</strong> columns "
                f"in <strong>{elapsed:.1f}s</strong>."
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="warning-box">'
                "⚠️ No structured data could be extracted from the text. "
                "Try pasting text with clearer data patterns."
                "</div>",
                unsafe_allow_html=True,
            )

    except NotImplementedError as exc:
        progress_bar.empty()
        status.empty()
        st.markdown(
            f'<div class="error-box">🔧 <strong>Not Implemented:</strong> {exc}</div>',
            unsafe_allow_html=True,
        )

    except Exception as exc:
        progress_bar.empty()
        status.empty()
        error_str = str(exc)
        error_detail = traceback.format_exc()

        # ── Smart error messages ──────────────────────────────────────────
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
            st.markdown(
                '<div class="error-box">'
                "🚫 <strong>Quota / Rate Limit Exceeded</strong><br><br>"
                "Your API key has hit its quota. Try:<br>"
                "• Wait 1 minute and retry<br>"
                "• Switch to a different model in the sidebar<br>"
                '• <a href="https://aistudio.google.com/app/apikey" '
                'target="_blank" style="color:#f59e0b;">Check your quota</a>'
                "</div>",
                unsafe_allow_html=True,
            )
        elif "404" in error_str or "NOT_FOUND" in error_str:
            st.markdown(
                '<div class="error-box">'
                "🔑 <strong>API Key / Model Error (404)</strong><br><br>"
                "The API endpoint returned 404. Try:<br>"
                "• Double-check your API key in the sidebar<br>"
                '• Get a free key at <a href="https://aistudio.google.com/app/apikey" target="_blank" style="color:#f59e0b;">aistudio.google.com</a><br>'
                "• Switch provider to 🟢 OpenAI if you have an OpenAI key"
                "</div>",
                unsafe_allow_html=True,
            )
        elif "401" in error_str or "invalid" in error_str.lower():
            st.markdown(
                '<div class="error-box">'
                "🔑 <strong>Invalid API Key</strong> — "
                "Double-check the key in the sidebar."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="error-box">❌ <strong>Error:</strong> {exc}</div>',
                unsafe_allow_html=True,
            )

        with st.expander("🔍 Error Details"):
            st.code(error_detail, language="python")


# ═════════════════════════════════════════════════════════════════════════════
# RESULTS SECTION
# ═════════════════════════════════════════════════════════════════════════════
# Person B will enhance this section with styled st.dataframe, download
# buttons for CSV/JSON, and data preview.  The basic display below ensures
# the app is testable before Person B's code is integrated.

df = st.session_state.extraction_df

if df is not None and not df.empty:
    st.markdown("---")
    st.markdown(
        '<div class="ui-card"><h3>📊 Extracted Dataset</h3></div>',
        unsafe_allow_html=True,
    )

    # ── Summary metrics ───────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Records", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        null_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Completeness", f"{100 - null_pct:.0f}%")

    # ── Data preview ──────────────────────────────────────────────────────
    st.dataframe(df, use_container_width=True, height=400)

    # ── Download buttons (basic fallback — Person B will replace) ─────────
    st.markdown("")
    dl_col1, dl_col2 = st.columns(2)

    if _EXPORTER_AVAILABLE:
        # Person B's export functions are available
        with dl_col1:
            csv_data = convert_to_csv(df)
            st.download_button(
                "📥 Download CSV",
                data=csv_data,
                file_name="dataset.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl_col2:
            json_data = convert_to_json(df)
            st.download_button(
                "📥 Download JSON",
                data=json_data,
                file_name="dataset.json",
                mime="application/json",
                use_container_width=True,
            )
    else:
        # Basic fallback until Person B implements exporter.py
        with dl_col1:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download CSV",
                data=csv_bytes,
                file_name="dataset.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl_col2:
            json_str = df.to_json(orient="records", indent=2)
            st.download_button(
                "📥 Download JSON",
                data=json_str,
                file_name="dataset.json",
                mime="application/json",
                use_container_width=True,
            )

    # ── Raw JSON preview (collapsible) ────────────────────────────────────
    with st.expander("🔍 Raw JSON Preview"):
        st.json(df.to_dict(orient="records"))


# ═════════════════════════════════════════════════════════════════════════════
# Footer
# ═════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class="app-footer">
        🧬 Text-to-Dataset Converter &nbsp;|&nbsp;
        Powered by Gemini · OpenAI · Streamlit · Pandas
        &nbsp;|&nbsp; Person A + Person B Collab
    </div>
    """,
    unsafe_allow_html=True,
)
