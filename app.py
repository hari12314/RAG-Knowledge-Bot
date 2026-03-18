import streamlit as st
import os
import time
import tempfile
from datetime import datetime
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(
    page_title="KnowBot - RAG Chatbot",
    page_icon="K",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Fira+Code:wght@300;400;500&display=swap');

:root {
    --bg:      #0f0f17;
    --s1:      #16161f;
    --s2:      #1e1e2a;
    --s3:      #252533;
    --border:  #2e2e3e;
    --acc:     #7c6fef;
    --acc2:    #ef6f9f;
    --green:   #4eca8b;
    --yellow:  #f0c060;
    --text:    #e8e8f2;
    --muted:   #686880;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: var(--s1) !important;
    border-right: 1px solid var(--border) !important;
}

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid var(--border);
    position: relative;
    overflow: hidden;
}

.hero::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(124,111,239,0.15) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: white;
    letter-spacing: -1px;
    margin: 0;
    line-height: 1;
}

.hero-title span { color: #7c6fef; }

.hero-sub {
    font-family: 'Fira Code', monospace;
    font-size: 0.72rem;
    color: #686880;
    margin-top: 0.5rem;
    letter-spacing: 0.05em;
}

.chat-container {
    height: 520px;
    overflow-y: auto;
    padding: 1rem;
    background: var(--s1);
    border: 1px solid var(--border);
    border-radius: 12px;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    scroll-behavior: smooth;
}

.msg-user {
    display: flex;
    justify-content: flex-end;
}

.msg-bot {
    display: flex;
    justify-content: flex-start;
    gap: 0.7rem;
    align-items: flex-start;
}

.bubble-user {
    background: var(--acc);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 0.75rem 1.1rem;
    max-width: 75%;
    font-size: 0.9rem;
    line-height: 1.6;
}

.bubble-bot {
    background: var(--s2);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 18px 18px 18px 4px;
    padding: 0.75rem 1.1rem;
    max-width: 78%;
    font-size: 0.9rem;
    line-height: 1.7;
}

.bot-avatar {
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, var(--acc), var(--acc2));
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: 700;
    color: white;
    flex-shrink: 0;
    margin-top: 2px;
}

.msg-meta {
    font-family: 'Fira Code', monospace;
    font-size: 0.62rem;
    color: var(--muted);
    margin-top: 0.3rem;
    padding: 0 0.3rem;
}

.source-pill {
    display: inline-block;
    background: var(--s3);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2px 8px;
    font-family: 'Fira Code', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    margin: 2px 2px 0 0;
}

.stat-box {
    background: var(--s2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem;
    text-align: center;
}

.stat-n { font-size: 1.5rem; font-weight: 800; color: var(--acc); line-height: 1; }
.stat-l { font-family: 'Fira Code', monospace; font-size: 0.62rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.2rem; }

.kb-doc {
    background: var(--s2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.7rem;
    font-size: 0.82rem;
    color: var(--text);
}

.dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }

.welcome-msg {
    background: linear-gradient(135deg, #1a1a2e, #1e2040);
    border: 1px solid var(--border);
    border-radius: 18px 18px 18px 4px;
    padding: 1.2rem 1.4rem;
    max-width: 85%;
    font-size: 0.9rem;
    line-height: 1.8;
    color: var(--text);
}

.thinking {
    display: flex;
    gap: 4px;
    padding: 0.5rem 0;
}

.thinking span {
    width: 6px; height: 6px;
    background: var(--acc);
    border-radius: 50%;
    display: inline-block;
    animation: bounce 1.2s infinite;
}

.thinking span:nth-child(2) { animation-delay: 0.2s; }
.thinking span:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
    0%, 80%, 100% { transform: translateY(0); }
    40% { transform: translateY(-6px); }
}

.stTextInput input {
    background: var(--s2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.92rem !important;
}

.stTextInput input:focus { border-color: var(--acc) !important; }

[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: var(--s2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

[data-testid="stSidebar"] * { color: var(--text) !important; }

.stSelectbox > div > div {
    background: var(--s2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

.stButton > button {
    background: var(--acc) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    width: 100% !important;
}

.stButton > button:hover { background: var(--acc2) !important; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)

DOC_COLORS = ["#7c6fef","#ef6f9f","#4eca8b","#f0c060","#38bdf8","#fb923c"]

# Session state init
if "messages"    not in st.session_state: st.session_state["messages"]    = []
if "vs"          not in st.session_state: st.session_state["vs"]          = None
if "doc_names"   not in st.session_state: st.session_state["doc_names"]   = []
if "kb_built"    not in st.session_state: st.session_state["kb_built"]    = False
if "total_chunks" not in st.session_state: st.session_state["total_chunks"] = 0

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device":"cpu"})

def load_file(f):
    suffix = ".pdf" if f.type == "application/pdf" else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(f.read())
        path = tmp.name
    loader = PyPDFLoader(path) if suffix == ".pdf" else TextLoader(path, encoding="utf-8")
    pages = loader.load()
    for p in pages: p.metadata["source"] = f.name
    os.unlink(path)
    return pages

def build_kb(all_docs, chunk_size, chunk_overlap, emb):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n","\n",". "," ",""],
    )
    chunks = splitter.split_documents(all_docs)
    vs = FAISS.from_documents(chunks, emb)
    return vs, chunks

def build_system_prompt(bot_name, role, company, tone):
    return (
        "You are " + bot_name + ", a " + role + " for " + company + ".\n"
        "Your tone is " + tone + ".\n\n"
        "STRICT RULES:\n"
        "1. Answer ONLY using the CONTEXT provided below.\n"
        "2. If the answer is not in the context, say: "
        "\"I don't have information about that in my knowledge base. "
        "Please contact our support team for further assistance.\"\n"
        "3. Never make up facts, prices, or features not in the context.\n"
        "4. Never discuss competitors.\n"
        "5. Keep answers concise (2-4 sentences) unless detail is needed.\n"
        "6. Always suggest next steps when possible.\n"
        "7. Use conversation history to understand follow-up questions.\n"
    )

def rephrase_question(client, question, history_text, model):
    if not history_text or history_text == "No previous conversation.":
        return question
    prompt = (
        "Given this conversation history, rephrase the follow-up question as standalone.\n"
        "If already standalone, return unchanged. Return ONLY the question.\n\n"
        "HISTORY:\n" + history_text + "\n\n"
        "FOLLOW-UP: " + question + "\n\nSTANDALONE:"
    )
    r = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=0.0, max_tokens=80,
    )
    return r.choices[0].message.content.strip()

def get_history_text(messages, n=6):
    if not messages: return "No previous conversation."
    recent = messages[-n:]
    lines = []
    for m in recent:
        prefix = "User: " if m["role"] == "user" else "Bot: "
        lines.append(prefix + m["content"][:200])
    return "\n".join(lines)

def chat_with_bot(client, model, vs, system_prompt, messages, user_message, top_k, temperature):
    history_text = get_history_text(messages)
    search_q     = rephrase_question(client, user_message, history_text, model)
    chunks       = vs.similarity_search(search_q, k=top_k)

    context = "\n\n".join([
        "SOURCE [" + c.metadata.get("source","?") + "]:\n" + c.page_content
        for i, c in enumerate(chunks)
    ])

    user_prompt = (
        "KNOWLEDGE BASE CONTEXT:\n\n" + context +
        "\n\n---\n\nCONVERSATION HISTORY:\n" + history_text +
        "\n\n---\n\nCURRENT QUESTION: " + user_message + "\n\nANSWER:"
    )

    api_messages = [{"role":"system","content":system_prompt}]
    for m in messages[-6:]:
        api_messages.append({"role": m["role"], "content": m["content"]})
    api_messages.append({"role":"user","content":user_prompt})

    r = client.chat.completions.create(
        model=model,
        messages=api_messages,
        temperature=temperature,
        max_tokens=600,
    )
    answer  = r.choices[0].message.content.strip()
    sources = list(set(c.metadata.get("source","") for c in chunks))
    return answer, sources, search_q

# SIDEBAR
with st.sidebar:
    st.markdown(
        '<div style="font-family:Plus Jakarta Sans,sans-serif;font-size:1.3rem;font-weight:800;'
        'color:#e8e8f2;letter-spacing:-0.5px;">Know<span style="color:#7c6fef;">Bot</span></div>',
        unsafe_allow_html=True)
    st.markdown('<div style="font-family:Fira Code,monospace;font-size:0.7rem;color:#686880;margin-bottom:1rem;">RAG Knowledge Chatbot</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**API Key**")
    st.caption("Free at: https://console.groq.com")
    api_key = st.text_input("", type="password", placeholder="gsk_...", label_visibility="collapsed")

    st.markdown("**Model**")
    model = st.selectbox("", ["llama-3.3-70b-versatile","llama-3.1-8b-instant","mixtral-8x7b-32768","gemma2-9b-it"], label_visibility="collapsed")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)

    st.markdown("**Bot Persona**")
    bot_name = st.text_input("Bot Name", value="KnowBot")
    company  = st.text_input("Company", value="TechCorp")
    role     = st.selectbox("Role", ["customer support assistant","sales assistant","technical support engineer","HR assistant","general assistant"])
    tone     = st.selectbox("Tone", ["professional and friendly","empathetic","precise and technical","enthusiastic","concise"])

    st.markdown("**Retrieval**")
    chunk_size    = st.slider("Chunk Size",    200, 1000, 400, 50)
    chunk_overlap = st.slider("Chunk Overlap",   0,  200,  60, 10)
    top_k         = st.slider("Top K Chunks",    1,    8,   4)

    st.markdown("---")
    if st.button("Clear Conversation"):
        st.session_state["messages"] = []
        st.rerun()

    if st.session_state["kb_built"]:
        st.markdown("---")
        st.markdown("**Knowledge Base**")
        for i, doc in enumerate(st.session_state["doc_names"]):
            color = DOC_COLORS[i % len(DOC_COLORS)]
            st.markdown(
                '<div class="kb-doc">'
                '<div class="dot" style="background:' + color + '"></div>'
                + doc + '</div>', unsafe_allow_html=True)

# HEADER
st.markdown(
    '<div class="hero"><h1 class="hero-title">Know<span>Bot</span></h1>'
    '<p class="hero-sub">CONVERSATIONAL RAG &nbsp;|&nbsp; SESSION MEMORY &nbsp;|&nbsp; '
    'ROLE-BASED PROMPTS &nbsp;|&nbsp; GUARDRAILS</p></div>',
    unsafe_allow_html=True)

left, right = st.columns([1, 1.4], gap="large")

# LEFT: Upload
with left:
    st.markdown('<p style="font-family:Fira Code,monospace;font-size:0.68rem;color:#7c6fef;text-transform:uppercase;letter-spacing:0.12em;">Knowledge Base</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["pdf","txt"], accept_multiple_files=True, label_visibility="collapsed")

    if uploaded:
        build_btn = st.button("Build Knowledge Base (" + str(len(uploaded)) + " files)")
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
                with st.spinner("Building FAISS index..."):
                    vs, chunks = build_kb(all_pages, chunk_size, chunk_overlap, emb)
                st.session_state["vs"]           = vs
                st.session_state["doc_names"]    = [f.name for f in uploaded]
                st.session_state["kb_built"]     = True
                st.session_state["total_chunks"] = len(chunks)
                st.session_state["messages"]     = []
                st.success("Knowledge base ready!")

    if st.session_state["kb_built"]:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="stat-box"><div class="stat-n">' + str(len(st.session_state["doc_names"])) + '</div><div class="stat-l">Docs</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="stat-box"><div class="stat-n">' + str(st.session_state["total_chunks"]) + '</div><div class="stat-l">Chunks</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="stat-box"><div class="stat-n">' + str(len(st.session_state["messages"])//2) + '</div><div class="stat-l">Turns</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p style="font-family:Fira Code,monospace;font-size:0.65rem;color:#686880;text-transform:uppercase;letter-spacing:0.1em;">How it works</p>', unsafe_allow_html=True)
        steps = [
            ("1", "Rephrase", "Follow-up → Standalone query"),
            ("2", "Retrieve", "FAISS Top-K similarity search"),
            ("3", "Prompt",   "Role + Memory + Context"),
            ("4", "Generate", "Groq LLM with guardrails"),
            ("5", "Remember", "Stored in session memory"),
        ]
        for num, title, desc in steps:
            st.markdown(
                '<div style="background:var(--s2);border:1px solid var(--border);border-radius:8px;'
                'padding:0.5rem 0.8rem;margin-bottom:0.3rem;display:flex;gap:0.7rem;align-items:center;">'
                '<span style="color:#7c6fef;font-weight:800;font-family:Fira Code,monospace;font-size:0.8rem;">' + num + '</span>'
                '<div><strong style="font-size:0.82rem;color:#e8e8f2;">' + title + '</strong>'
                '<span style="font-size:0.75rem;color:#686880;margin-left:0.5rem;">' + desc + '</span></div>'
                '</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="background:var(--s1);border:1px dashed var(--border);border-radius:12px;'
            'padding:3rem;text-align:center;color:var(--muted);font-family:Fira Code,monospace;font-size:0.8rem;">'
            'Upload your knowledge base files<br>'
            '<span style="font-size:0.7rem;opacity:0.6;">PDF or TXT files</span>'
            '</div>', unsafe_allow_html=True)

# RIGHT: Chat
with right:
    st.markdown('<p style="font-family:Fira Code,monospace;font-size:0.68rem;color:#7c6fef;text-transform:uppercase;letter-spacing:0.12em;">Chat</p>', unsafe_allow_html=True)

    # Build chat HTML
    chat_html = '<div class="chat-container" id="chat-container">'

    if not st.session_state["kb_built"]:
        chat_html += (
            '<div class="msg-bot"><div class="bot-avatar">K</div>'
            '<div class="welcome-msg">'
            'Hello! I am KnowBot.<br><br>'
            'Please upload your knowledge base documents on the left and click '
            '<strong>Build Knowledge Base</strong> to get started.<br><br>'
            'I will then answer questions based strictly on your documents.'
            '</div></div>'
        )
    elif not st.session_state["messages"]:
        chat_html += (
            '<div class="msg-bot"><div class="bot-avatar">K</div>'
            '<div class="welcome-msg">'
            'Hello! I am <strong>' + bot_name + '</strong>, your ' + role + ' for ' + company + '.<br><br>'
            'I have loaded <strong>' + str(len(st.session_state["doc_names"])) + ' document(s)</strong> into my knowledge base.<br><br>'
            'Ask me anything and I will answer strictly from the provided documents.'
            '</div></div>'
        )
    else:
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                chat_html += (
                    '<div class="msg-user"><div class="bubble-user">' + msg["content"] + '</div></div>'
                )
            else:
                src_pills = "".join(['<span class="source-pill">' + s + '</span>' for s in msg.get("sources",[])])
                chat_html += (
                    '<div class="msg-bot"><div class="bot-avatar">' + bot_name[0].upper() + '</div>'
                    '<div><div class="bubble-bot">' + msg["content"].replace("\n","<br>") + '</div>'
                    '<div class="msg-meta">' + msg.get("time","") + ' &nbsp; ' + src_pills + '</div>'
                    '</div></div>'
                )

    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)

    # Input row
    if st.session_state["kb_built"]:
        col_input, col_send = st.columns([5, 1])
        with col_input:
            user_input = st.text_input("", placeholder="Ask anything about your documents...", label_visibility="collapsed", key="chat_input")
        with col_send:
            send_btn = st.button("Send")

        if (send_btn or user_input) and user_input.strip():
            if not api_key:
                st.error("Enter your Groq API key in the sidebar.")
            else:
                sys_prompt = build_system_prompt(bot_name, role, company, tone)
                groq_client = Groq(api_key=api_key)

                with st.spinner(bot_name + " is thinking..."):
                    try:
                        t0 = time.time()
                        answer, sources, rephrased = chat_with_bot(
                            groq_client, model,
                            st.session_state["vs"],
                            sys_prompt,
                            st.session_state["messages"],
                            user_input.strip(),
                            top_k, temperature,
                        )
                        elapsed = round(time.time()-t0, 2)

                        st.session_state["messages"].append({
                            "role": "user",
                            "content": user_input.strip(),
                        })
                        st.session_state["messages"].append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "time": datetime.now().strftime("%H:%M") + " · " + str(elapsed) + "s",
                        })
                        st.rerun()
                    except Exception as e:
                        err = str(e)
                        if "401" in err or "invalid" in err.lower():
                            st.error("Invalid API key.")
                        elif "429" in err:
                            st.error("Rate limit hit. Wait a moment.")
                        else:
                            st.error("Error: " + err)

    # Suggested questions
    if st.session_state["kb_built"] and not st.session_state["messages"]:
        st.markdown('<p style="font-family:Fira Code,monospace;font-size:0.65rem;color:#686880;margin-top:0.8rem;">Suggested questions:</p>', unsafe_allow_html=True)
        suggestions = ["What can you help me with?","What products or services are available?","What are the pricing plans?"]
        scols = st.columns(len(suggestions))
        for i, (sc, q) in enumerate(zip(scols, suggestions)):
            with sc:
                if st.button(q, key="sug_" + str(i)):
                    if api_key:
                        sys_prompt = build_system_prompt(bot_name, role, company, tone)
                        groq_client = Groq(api_key=api_key)
                        with st.spinner("Thinking..."):
                            try:
                                answer, sources, _ = chat_with_bot(
                                    groq_client, model, st.session_state["vs"],
                                    sys_prompt, [], q, top_k, temperature)
                                st.session_state["messages"].append({"role":"user","content":q})
                                st.session_state["messages"].append({
                                    "role":"assistant","content":answer,"sources":sources,
                                    "time":datetime.now().strftime("%H:%M")})
                                st.rerun()
                            except Exception as e:
                                st.error("Error: " + str(e))
