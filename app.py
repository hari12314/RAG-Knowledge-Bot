import streamlit as st
import os
import json
import time
import tempfile
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(
    page_title="SummarAI - RAG Summarizer",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@300;400;600;700;900&family=Karla:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400&display=swap');

:root {
    --bg:     #f7f3ee;
    --s1:     #efe9e0;
    --s2:     #e5ddd2;
    --border: #d4cab8;
    --ink:    #1c1a16;
    --acc:    #c0392b;
    --acc2:   #2c6e49;
    --gold:   #c77d19;
    --muted:  #7a7060;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--ink) !important;
    font-family: 'Karla', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: var(--ink) !important;
    border-right: none !important;
}

[data-testid="stSidebar"] * { color: #e8e4dc !important; }

[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #2a2820 !important;
    border: 1px solid #3a3830 !important;
    color: #e8e4dc !important;
    border-radius: 6px !important;
}

.hero {
    border-bottom: 3px solid var(--ink);
    padding-bottom: 1.2rem;
    margin-bottom: 2rem;
}

.hero-title {
    font-family: 'Fraunces', serif;
    font-size: 3.5rem;
    font-weight: 900;
    color: var(--ink);
    letter-spacing: -2px;
    line-height: 1;
    margin: 0;
}

.hero-title em { color: var(--acc); font-style: normal; }

.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 0.5rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.pipeline-step {
    background: var(--s1);
    border: 1px solid var(--border);
    border-top: 3px solid var(--acc);
    border-radius: 0 0 10px 10px;
    padding: 0.9rem 1rem;
    text-align: center;
    font-size: 0.82rem;
    color: var(--muted);
}

.pipeline-step .num {
    font-family: 'Fraunces', serif;
    font-size: 1.6rem;
    font-weight: 900;
    color: var(--acc);
    display: block;
    line-height: 1;
}

.pipeline-step strong { color: var(--ink); display: block; font-size: 0.88rem; }

.doc-row {
    background: var(--s1);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    font-size: 0.84rem;
}

.result-section {
    background: white;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.6rem;
    margin-bottom: 1rem;
}

.result-section.one-line {
    background: var(--ink);
    color: #f0ece4;
    border: none;
}

.result-section.one-line .rlabel { color: var(--gold) !important; }
.result-section.one-line .rtext  { color: #f0ece4; font-size: 1.1rem; font-weight: 600; }

.result-section.executive { border-left: 4px solid var(--acc2); }
.result-section.keypoints { border-left: 4px solid var(--gold); }
.result-section.topics    { border-left: 4px solid var(--acc); }

.rlabel {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.7rem;
}

.rtext { font-size: 0.95rem; line-height: 1.85; color: var(--ink); }

.kp-item {
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.92rem;
    color: var(--ink);
    line-height: 1.6;
    display: flex;
    gap: 0.6rem;
    align-items: flex-start;
}

.kp-item:last-child { border-bottom: none; }

.topic-tag {
    display: inline-block;
    background: var(--s2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    color: var(--ink);
    margin: 3px 3px 0 0;
    font-family: 'JetBrains Mono', monospace;
}

.meta-row {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-top: 0.8rem;
}

.meta-chip {
    background: var(--s1);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
}

.meta-chip span { color: var(--ink); font-weight: 600; }

.chunk-preview {
    background: var(--s1);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
    line-height: 1.7;
    white-space: pre-wrap;
    max-height: 150px;
    overflow-y: auto;
}

.master-box {
    background: var(--acc2);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    color: white;
    margin-bottom: 1rem;
}

.stButton > button {
    background: var(--ink) !important;
    color: #f0ece4 !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Karla', sans-serif !important;
    font-weight: 600 !important;
    width: 100% !important;
    letter-spacing: 0.03em !important;
}

.stButton > button:hover { background: var(--acc) !important; }

.stTextArea textarea {
    background: white !important;
    border: 1px solid var(--border) !important;
    color: var(--ink) !important;
    border-radius: 8px !important;
    font-family: 'Karla', sans-serif !important;
}

.stTextArea textarea:focus { border-color: var(--ink) !important; }

.stSelectbox > div > div {
    background: white !important;
    border: 1px solid var(--border) !important;
    color: var(--ink) !important;
    border-radius: 8px !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)

DOC_COLORS = ["#c0392b","#2c6e49","#c77d19","#2563eb","#7c3aed","#0891b2"]

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device":"cpu"})

def load_file(uploaded_file):
    suffix = ".pdf" if uploaded_file.type == "application/pdf" else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name
    loader = PyPDFLoader(path) if suffix == ".pdf" else TextLoader(path, encoding="utf-8")
    pages = loader.load()
    for p in pages:
        p.metadata["source"] = uploaded_file.name
    os.unlink(path)
    return pages

def build_index(all_docs, chunk_size, chunk_overlap, emb):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n","\n",". "," ",""],
    )
    chunks = splitter.split_documents(all_docs)
    vs = FAISS.from_documents(chunks, emb)
    return vs, chunks

def retrieve_chunks(vs, doc_name, top_k):
    query = "main topics key points important information overview summary"
    results = vs.similarity_search(query, k=top_k * 3)
    filtered = [r for r in results if r.metadata.get("source") == doc_name]
    return filtered[:top_k]

def map_chunk(client, model, chunk_text, source):
    prompt = (
        "Summarize this text section into 2-4 concise bullet points using - symbol.\n"
        "Use only information from the text. No external knowledge.\n\n"
        "TEXT (from " + source + "):\n" + chunk_text + "\n\nBULLET POINTS:"
    )
    r = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=0.1, max_tokens=300,
    )
    return r.choices[0].message.content.strip()

def reduce_to_json(client, model, chunk_summaries, doc_name):
    combined = "\n\n".join([
        "Section " + str(i+1) + ":\n" + s["summary"]
        for i, s in enumerate(chunk_summaries)
    ])
    prompt = (
        "Based on these section summaries from '" + doc_name + "', generate a structured summary.\n"
        "Use ONLY the provided information. Return valid JSON only:\n"
        "{\n"
        "  \"one_line_summary\": \"one sentence\",\n"
        "  \"executive_summary\": \"2-3 paragraph summary\",\n"
        "  \"key_points\": [\"point1\", \"point2\", \"point3\", \"point4\", \"point5\"],\n"
        "  \"topics_covered\": [\"topic1\", \"topic2\", \"topic3\"]\n"
        "}\n\n"
        "SECTION SUMMARIES:\n\n" + combined + "\n\nJSON:"
    )
    r = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=0.1, max_tokens=1000,
    )
    raw = r.choices[0].message.content.strip()
    try:
        if "```" in raw:
            for part in raw.split("```"):
                part = part.strip()
                if part.startswith("json"): part = part[4:].strip()
                if part.startswith("{"): raw = part; break
        return json.loads(raw)
    except Exception:
        return {"one_line_summary":"","executive_summary":raw,"key_points":[],"topics_covered":[]}

def summarize_doc(client, model, vs, doc_name, top_k):
    chunks = retrieve_chunks(vs, doc_name, top_k)
    chunk_sums = []
    for c in chunks:
        s = map_chunk(client, model, c.page_content, c.metadata.get("source",""))
        chunk_sums.append({"source": c.metadata.get("source",""), "summary": s})
    result = reduce_to_json(client, model, chunk_sums, doc_name)
    result["chunks_used"] = len(chunks)
    result["chunk_summaries"] = chunk_sums
    result["retrieved_chunks"] = chunks
    return result

def master_summary(client, model, all_results):
    liners = "\n".join(["- " + r["source"] + ": " + r.get("one_line_summary","") for r in all_results])
    prompt = (
        "Write a 3-sentence master overview capturing themes across all these documents.\n"
        "Use only the provided summaries.\n\n"
        "DOCUMENTS:\n" + liners + "\n\nMASTER OVERVIEW:"
    )
    r = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=0.2, max_tokens=300,
    )
    return r.choices[0].message.content.strip()

# SIDEBAR
with st.sidebar:
    st.markdown(
        '<div style="font-family:Fraunces,serif;font-size:1.5rem;font-weight:900;'
        'color:#f0ece4;letter-spacing:-0.5px;">Summar<em style="color:#c0392b;">AI</em></div>',
        unsafe_allow_html=True)
    st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#6b6858;margin-bottom:1rem;">RAG Document Summarizer</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**API Key**")
    st.caption("Free at: https://console.groq.com")
    api_key = st.text_input("", type="password", placeholder="gsk_...", label_visibility="collapsed")

    st.markdown("**Model**")
    model = st.selectbox("", ["llama-3.3-70b-versatile","llama-3.1-8b-instant","mixtral-8x7b-32768","gemma2-9b-it"], label_visibility="collapsed")

    st.markdown("**Chunk Settings**")
    chunk_size    = st.slider("Chunk Size", 400, 1500, 800, 100)
    chunk_overlap = st.slider("Overlap", 0, 200, 100, 20)

    st.markdown("**Retrieval**")
    top_k = st.slider("Top K per Document", 2, 8, 4)

    st.markdown("---")
    st.markdown(
        '<div style="font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#4a4838;line-height:1.9;">'
        'Method: Map-Reduce<br>Embeddings: MiniLM-L6-v2<br>Vector DB: FAISS<br>'
        'Output: Structured JSON'
        '</div>', unsafe_allow_html=True)

# HEADER
st.markdown(
    '<div class="hero"><h1 class="hero-title">Summar<em>AI</em></h1>'
    '<p class="hero-sub">RAG SUMMARIZATION &nbsp;|&nbsp; MAP-REDUCE PIPELINE &nbsp;|&nbsp; STRUCTURED JSON OUTPUT</p></div>',
    unsafe_allow_html=True)

# Pipeline steps
cols = st.columns(5)
steps = [("1","Upload","PDF or TXT"),("2","Chunk","Split text"),("3","Index","FAISS"),("4","Map","Per-chunk summary"),("5","Reduce","Final JSON")]
for i, (num, title, sub) in enumerate(steps):
    with cols[i]:
        st.markdown(
            '<div class="pipeline-step"><span class="num">' + num + '</span>'
            '<strong>' + title + '</strong>' + sub + '</div>',
            unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

# LEFT
with left:
    st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#c0392b;text-transform:uppercase;letter-spacing:0.12em;">Step 1 — Upload Documents</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["pdf","txt"], accept_multiple_files=True, label_visibility="collapsed")

    if uploaded:
        for i, f in enumerate(uploaded):
            color = DOC_COLORS[i % len(DOC_COLORS)]
            st.markdown(
                '<div class="doc-row">'
                '<div style="width:8px;height:8px;border-radius:50%;background:' + color + ';flex-shrink:0;"></div>'
                '<span style="flex:1;">' + f.name + '</span>'
                '<span style="font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#7a7060;">'
                + str(round(f.size/1024,1)) + 'KB</span></div>',
                unsafe_allow_html=True)

        build_btn = st.button("Build Index (" + str(len(uploaded)) + " file" + ("s" if len(uploaded)>1 else "") + ")")

        if build_btn:
            if not api_key:
                st.error("Enter your Groq API key in the sidebar.")
            else:
                with st.spinner("Loading embedding model..."):
                    emb = get_embeddings()
                all_pages = []
                with st.spinner("Reading documents..."):
                    for f in uploaded:
                        all_pages.extend(load_file(f))
                with st.spinner("Chunking and indexing..."):
                    vs, chunks = build_index(all_pages, chunk_size, chunk_overlap, emb)

                st.session_state["vs"]        = vs
                st.session_state["chunks"]    = chunks
                st.session_state["doc_names"] = [f.name for f in uploaded]
                st.session_state["results"]   = {}
                st.success("Index ready! " + str(len(chunks)) + " chunks indexed.")

    if "vs" in st.session_state:
        docs = st.session_state["doc_names"]
        n_chunks = len(st.session_state["chunks"])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="pipeline-step"><span class="num">' + str(len(docs)) + '</span><strong>Documents</strong>indexed</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="pipeline-step"><span class="num">' + str(n_chunks) + '</span><strong>Chunks</strong>total</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Select document to summarize:**")
        selected = st.selectbox("", docs, label_visibility="collapsed")

        sum_btn = st.button("Summarize: " + selected)

        if sum_btn:
            if not api_key:
                st.error("Enter your Groq API key.")
            else:
                groq_client = Groq(api_key=api_key)
                progress = st.progress(0, text="Retrieving chunks...")
                try:
                    chunks_for_doc = retrieve_chunks(st.session_state["vs"], selected, top_k)
                    progress.progress(25, text="MAP: Summarizing " + str(len(chunks_for_doc)) + " chunks...")
                    chunk_sums = []
                    for i, c in enumerate(chunks_for_doc):
                        s = map_chunk(groq_client, model, c.page_content, c.metadata.get("source",""))
                        chunk_sums.append({"source": c.metadata.get("source",""), "summary": s})
                        progress.progress(25 + int(50 * (i+1) / len(chunks_for_doc)),
                                          text="MAP: chunk " + str(i+1) + "/" + str(len(chunks_for_doc)))
                    progress.progress(80, text="REDUCE: Generating structured summary...")
                    result = reduce_to_json(groq_client, model, chunk_sums, selected)
                    result["chunks_used"]     = len(chunks_for_doc)
                    result["chunk_summaries"] = chunk_sums
                    result["retrieved_chunks"]= chunks_for_doc
                    st.session_state["results"][selected] = result
                    st.session_state["active_doc"] = selected
                    progress.progress(100, text="Done!")
                    time.sleep(0.5)
                    progress.empty()
                    st.success("Summary ready!")
                except Exception as e:
                    progress.empty()
                    st.error("Error: " + str(e))

        # Multi-doc summarize
        if len(docs) > 1 and "vs" in st.session_state:
            st.markdown("<br>", unsafe_allow_html=True)
            all_btn = st.button("Summarize ALL " + str(len(docs)) + " Documents")
            if all_btn:
                if not api_key:
                    st.error("Enter your Groq API key.")
                else:
                    groq_client = Groq(api_key=api_key)
                    all_results = []
                    for doc in docs:
                        with st.spinner("Summarizing " + doc + "..."):
                            try:
                                chunks_d = retrieve_chunks(st.session_state["vs"], doc, top_k)
                                chunk_sums_d = [{"source": c.metadata.get("source",""),
                                                 "summary": map_chunk(groq_client, model, c.page_content, c.metadata.get("source",""))}
                                                for c in chunks_d]
                                res = reduce_to_json(groq_client, model, chunk_sums_d, doc)
                                res["chunks_used"] = len(chunks_d)
                                res["chunk_summaries"] = chunk_sums_d
                                res["retrieved_chunks"] = chunks_d
                                st.session_state["results"][doc] = res
                                all_results.append({**res, "source": doc})
                            except Exception as e:
                                st.error("Error on " + doc + ": " + str(e))
                    with st.spinner("Generating master summary..."):
                        ms = master_summary(groq_client, model, all_results)
                        st.session_state["master_summary"] = ms
                    st.session_state["active_doc"] = docs[0]
                    st.success("All documents summarized!")
    else:
        st.markdown(
            '<div style="background:var(--s1);border:1px solid var(--border);border-radius:10px;'
            'padding:2.5rem;text-align:center;color:var(--muted);font-family:JetBrains Mono,monospace;font-size:0.82rem;">'
            'Upload PDF or TXT files above<br>'
            '<span style="font-size:0.72rem;opacity:0.6;">then click Build Index</span>'
            '</div>', unsafe_allow_html=True)

# RIGHT
with right:
    st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#c0392b;text-transform:uppercase;letter-spacing:0.12em;">Step 2 — Summary Output</p>', unsafe_allow_html=True)

    active = st.session_state.get("active_doc")
    results = st.session_state.get("results", {})

    if "master_summary" in st.session_state and len(st.session_state.get("doc_names",[])) > 1:
        st.markdown(
            '<div class="master-box">'
            '<div style="font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#a8dfc0;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.6rem;">Master Cross-Document Summary</div>'
            '<div style="font-size:0.95rem;line-height:1.8;">' + st.session_state["master_summary"] + '</div>'
            '</div>', unsafe_allow_html=True)

        if len(results) > 1:
            shown = st.selectbox("View individual doc:", list(results.keys()), key="doc_selector")
            active = shown

    if active and active in results:
        r = results[active]

        # One-line
        st.markdown(
            '<div class="result-section one-line">'
            '<div class="rlabel">One-Line Summary</div>'
            '<div class="rtext">' + r.get("one_line_summary","") + '</div>'
            '</div>', unsafe_allow_html=True)

        # Executive summary
        st.markdown(
            '<div class="result-section executive">'
            '<div class="rlabel">Executive Summary</div>'
            '<div class="rtext">' + r.get("executive_summary","").replace("\n","<br>") + '</div>'
            '</div>', unsafe_allow_html=True)

        # Key points
        kp_html = ""
        for p in r.get("key_points",[]):
            kp_html += '<div class="kp-item"><span style="color:#c77d19;font-weight:700;flex-shrink:0;">-</span><span>' + p + '</span></div>'

        st.markdown(
            '<div class="result-section keypoints">'
            '<div class="rlabel">Key Points</div>'
            + kp_html +
            '</div>', unsafe_allow_html=True)

        # Topics
        topics_html = "".join(['<span class="topic-tag">' + t + '</span>' for t in r.get("topics_covered",[])])
        st.markdown(
            '<div class="result-section topics">'
            '<div class="rlabel">Topics Covered</div>'
            + topics_html +
            '</div>', unsafe_allow_html=True)

        # Meta
        st.markdown(
            '<div class="meta-row">'
            '<div class="meta-chip">Chunks used <span>' + str(r.get("chunks_used","")) + '</span></div>'
            '<div class="meta-chip">Key points <span>' + str(len(r.get("key_points",[]))) + '</span></div>'
            '<div class="meta-chip">Topics <span>' + str(len(r.get("topics_covered",[]))) + '</span></div>'
            '</div>', unsafe_allow_html=True)

        # Download
        st.markdown("<br>", unsafe_allow_html=True)
        export = {k: v for k, v in r.items() if k not in ["retrieved_chunks","chunk_summaries"]}
        export["document"] = active
        st.download_button(
            "Download Summary (JSON)",
            json.dumps(export, indent=2),
            file_name=active.replace(".","_") + "_summary.json",
            mime="application/json",
        )

        # Chunk inspector
        with st.expander("View Map Step (chunk-by-chunk summaries)"):
            for i, cs in enumerate(r.get("chunk_summaries",[])):
                st.markdown("**Chunk " + str(i+1) + "** — " + cs["source"])
                st.markdown('<div class="chunk-preview">' + cs["summary"] + '</div>', unsafe_allow_html=True)

        with st.expander("View Retrieved Chunks (raw text)"):
            for i, c in enumerate(r.get("retrieved_chunks",[])):
                st.markdown("**Chunk " + str(i+1) + "** (" + str(len(c.page_content)) + " chars)")
                st.markdown('<div class="chunk-preview">' + c.page_content + '</div>', unsafe_allow_html=True)

    else:
        st.markdown(
            '<div style="background:var(--s1);border:1px solid var(--border);border-radius:12px;'
            'padding:3rem;text-align:center;color:var(--muted);font-family:JetBrains Mono,monospace;'
            'font-size:0.82rem;min-height:400px;display:flex;align-items:center;'
            'justify-content:center;flex-direction:column;">'
            '<div style="font-family:Fraunces,serif;font-size:2rem;font-weight:900;color:#d4cab8;">S</div>'
            '<div style="margin-top:0.5rem;">Summary will appear here</div>'
            '</div>', unsafe_allow_html=True)
