"""
app.py
------
Streamlit chat UI for the Healthcare RAG Chatbot.
Run: python3 -m streamlit run app/app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pipeline import HealthcareRAGPipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Q&A Chatbot",
    page_icon="🏥",
    layout="wide",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .disclaimer {
    background: #fff3cd; color: #856404;
    border-left: 4px solid #ffc107;
    padding: .6rem 1rem; border-radius: 6px;
    font-size: .85rem; margin-bottom: 1rem;
  }
  .source-chip {
    display: inline-block;
    background: #e9ecef; color: #495057;
    border-radius: 12px; padding: 2px 10px;
    font-size: .75rem; margin: 2px;
  }
  .hospital-card {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
    font-size: 13px;
  }
  .hospital-name { font-weight: 600; color: #212529; margin-bottom: 2px; }
  .hospital-type { color: #6c757d; font-size: 11px; margin-bottom: 4px; }
  .hospital-dist { color: #0d6efd; font-size: 12px; }
  .emergency-box {
    background: #f8d7da; border: 1px solid #f5c2c7;
    border-radius: 8px; padding: 10px 12px;
    margin-bottom: 12px; text-align: center;
    font-weight: 600; color: #842029; font-size: 14px;
  }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    with st.spinner("Loading medical knowledge base..."):
        import os
        chroma_path = Path(__file__).parent.parent / "data" / "chroma_db"
        if not chroma_path.exists():
            st.info("⏳ First run: building knowledge base... (takes ~2 mins)")
            from ingest import load_and_chunk_data
            from embeddings import build_vector_store
            load_and_chunk_data()
            build_vector_store()
        st.session_state.pipeline = HealthcareRAGPipeline(top_k=5)
        
if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_sources" not in st.session_state:
    st.session_state.show_sources = False

if "city" not in st.session_state:
    st.session_state.city = "New York, NY"

# ── Layout: two columns ───────────────────────────────────────────────────────
chat_col, map_col = st.columns([3, 2], gap="large")

# ══════════════════════════════════════════════════════
# LEFT COLUMN — Chatbot
# ══════════════════════════════════════════════════════
with chat_col:
    st.title("🏥 Healthcare Q&A Chatbot")
    st.caption("Powered by PubMedQA · MedQuAD · LLaMA3 · ChromaDB · RAG")

    st.markdown("""<div class="disclaimer">
    ⚠️ <strong>Educational use only.</strong> Not a substitute for professional medical advice.
    Always consult a qualified healthcare provider.
    </div>""", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.session_state.show_sources = st.toggle(
            "Show retrieved sources", value=st.session_state.show_sources
        )
        top_k = st.slider("Sources to retrieve (top-k)", min_value=2, max_value=8, value=5)
        st.session_state.pipeline.retriever.top_k = top_k

        if st.button("🔄 New conversation"):
            st.session_state.messages = []
            st.session_state.pipeline.reset_history()
            st.rerun()

        st.divider()
        st.markdown("**Example questions:**")
        example_questions = [
            "What are symptoms of type 2 diabetes?",
            "How is hypertension treated?",
            "What causes Alzheimer's disease?",
            "Side effects of metformin?",
            "Cure for migraine?",
            "How to treat fever at home?",
        ]
        for eq in example_questions:
            if st.button(eq, use_container_width=True):
                st.session_state._prefill = eq
                st.rerun()

        st.divider()
        st.markdown("**Built with:** LangChain · ChromaDB · sentence-transformers · Groq · Streamlit")

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if (
                msg["role"] == "assistant"
                and st.session_state.show_sources
                and "sources" in msg
            ):
                with st.expander("📚 Retrieved Sources", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(
                            f'<span class="source-chip">{src["source"].upper()}</span>'
                            f'<span class="source-chip">relevance: {src["score"]}</span>',
                            unsafe_allow_html=True,
                        )
                        st.caption(src["text"][:300] + "...")
                        if i < len(msg["sources"]):
                            st.divider()

    # Chat input
    prefill = st.session_state.pop("_prefill", None)
    user_input = st.chat_input("Ask a medical question...") or prefill

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching medical knowledge base..."):
                result = st.session_state.pipeline.ask(user_input)
            st.markdown(result["answer"])

            if st.session_state.show_sources:
                with st.expander("📚 Retrieved Sources", expanded=False):
                    for i, src in enumerate(result["sources"], 1):
                        st.markdown(
                            f'<span class="source-chip">{src["source"].upper()}</span>'
                            f'<span class="source-chip">relevance: {src["score"]}</span>',
                            unsafe_allow_html=True,
                        )
                        st.caption(src["text"][:300] + "...")
                        if i < len(result["sources"]):
                            st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })

# ══════════════════════════════════════════════════════
# RIGHT COLUMN — Nearby Hospitals Map
# ══════════════════════════════════════════════════════
with map_col:
    st.subheader("🗺️ Nearby Healthcare Facilities")

    # City input
    city_input = st.text_input(
        "Enter your city or zip code",
        value=st.session_state.city,
        placeholder="e.g. Boston, MA or 10001",
    )

    facility_type = st.selectbox(
        "Facility type",
        ["Hospitals", "Urgent Care", "Emergency Room", "Clinics", "Pharmacy"],
    )

    if st.button("🔍 Find Nearby Facilities", use_container_width=True):
        st.session_state.city = city_input
        st.session_state.search_query = f"{facility_type} near {city_input}"
        st.rerun()

    # Emergency banner
    st.markdown("""<div class="emergency-box">
    🚨 Medical Emergency? Call 911 immediately
    </div>""", unsafe_allow_html=True)

    # Map using OpenStreetMap embed (free, no API key needed)
    search_query = getattr(st.session_state, "search_query",
                           f"Hospitals near {st.session_state.city}")

    encoded_query = search_query.replace(" ", "+")

    map_embed = f"""
    <iframe
        width="100%"
        height="420"
        frameborder="0"
        scrolling="no"
        marginheight="0"
        marginwidth="0"
        src="https://www.openstreetmap.org/export/embed.html?bbox=-74.1%2C40.6%2C-73.8%2C40.9&layer=mapnik&marker=40.7128%2C-74.0060"
        style="border-radius: 10px; border: 1px solid #dee2e6;"
    ></iframe>
    <br/>
    <a href="https://www.google.com/maps/search/{encoded_query}"
       target="_blank"
       style="display:block; text-align:center; margin-top:8px;
              background:#0d6efd; color:white; padding:8px 16px;
              border-radius:6px; text-decoration:none; font-size:13px; font-weight:500;">
       🗺️ Open Full Map on Google Maps
    </a>
    """
    st.markdown(map_embed, unsafe_allow_html=True)

    st.divider()

    # Quick links
    st.markdown("**🔗 Quick Links**")
    loc = city_input.replace(" ", "+")
    st.markdown(f"""
- [🏥 Hospitals near me](https://www.google.com/maps/search/hospitals+near+{loc})
- [🚑 Urgent care near me](https://www.google.com/maps/search/urgent+care+near+{loc})
- [💊 Pharmacy near me](https://www.google.com/maps/search/pharmacy+near+{loc})
- [🩺 Doctors near me](https://www.google.com/maps/search/doctors+near+{loc})
- [☎️ Poison Control](https://www.poisonhelp.org) — 1-800-222-1222
- [🧠 Mental Health Crisis](https://988lifeline.org) — Call/Text 988
    """)

    st.divider()
    st.caption("💡 Tip: Click any link to open Google Maps with real-time results in your area.")
