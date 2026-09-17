import os
import streamlit as st


try:
    if "COHERE_API_KEY" in st.secrets:
        os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]
except Exception:
    pass  # Local development uses standard .env or shell environment variables

from src.api_client import get_ai_response
from src.prompts import (
    ROLE_ASSISTANT,
    ROLE_USER,
    MAX_HISTORY_MESSAGES,
    build_messages,
    clean_user_input,
    get_fallback_message,
    is_valid_messages,
    messages_to_text,
)

# ---------------------------------------------------------------------------
# Page Configuration & Custom CSS Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TechMaster AI | Intelligent Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling - Modern Orange / Amber Warm Theme
st.markdown("""
<style>

    .main-header {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF6B00 0%, #FF8800 50%, #FFAA00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0rem;
    }
    
    .sub-header {
        font-size: 1.05rem;
        color: #A0A0A0;
        margin-bottom: 1.5rem;
    }

    /* --------------------------------------------------
       Buttons & Chips - Orange Glow Style
    -------------------------------------------------- */
    div.stButton > button {
        border-radius: 20px;
        border: 1px solid #333333;
        background-color: #1E1E1E;
        color: #E0E0E0;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:hover {
        border-color: #FF8800;
        color: #FFFFFF;
        background-color: #FF6B00;
        transform: translateY(-2px);
    }

    /* --------------------------------------------------
       Chat Bubbles Styling
    -------------------------------------------------- */
    [data-testid="stChatMessageContent"] {
        border-radius: 14px;
        padding: 0.8rem 1.1rem !important;
    }

    /* User Message Bubble - Deep Amber / Orange Tint */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
        background-color: #2D1A00 !important;
        border: 1px solid #5C3300 !important;
        color: #FFE8D6 !important;
    }

    /* Assistant Message Bubble - Dark Card Style */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
        background-color: #1E1E1E !important;
        border: 1px solid #2A2A2A !important;
        color: #E0E0E0 !important;
    }

    /* Persona Pill Tag - Orange Badge */
    .persona-pill {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.15rem 0.65rem;
        border-radius: 999px;
        background: rgba(255, 107, 0, 0.15);
        color: #FF8800;
        margin-bottom: 0.4rem;
        border: 1px solid rgba(255, 107, 0, 0.35);
    }

    /* Progress Bar Color */
    div.stProgress > div > div > div > div {
        background-color: #FF6B00 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# API Guardrail Verification
# ---------------------------------------------------------------------------
if not os.getenv("COHERE_API_KEY"):
    st.error("⚠️ COHERE_API_KEY not found. Please add it to your local .env or Streamlit Secrets.")
    st.stop()

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "pending_input" not in st.session_state:
    st.session_state.pending_input = None

# ---------------------------------------------------------------------------
# Sidebar Settings & Customization Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/bot.png", width=70)
    st.title("Control Panel")
    
    # AI Persona selector to adjust response tone
    persona = st.selectbox(
        "🧠 AI Assistant Persona",
        ["Technical Expert", "Socratic Tutor", "ELI5 Explainer"],
        help="Select how you want the AI to structure and explain its responses."
    )
    
    st.divider()
    
    # Conversation Statistics
    turns = len(st.session_state.history) // 2
    col_stat1, col_stat2 = st.columns(2)
    col_stat1.metric("Turns Used", turns)
    col_stat2.metric("Memory Cap", f"{MAX_HISTORY_MESSAGES // 2} Turns")
    
    st.progress(min(turns / (MAX_HISTORY_MESSAGES // 2), 1.0))
    
    st.divider()
    
    # Action Controls
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------------------------
st.markdown('<p class="main-header">⚡ TechMaster Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your intelligent pair-programmer and tech learning mentor • <i>Project 04</i></p>', unsafe_allow_html=True)

# Quick Suggestion Chips for Fast Entry (shown when conversation is empty)
if not st.session_state.history:
    st.write("##### 💡 Suggested topics to start:")
    chip_col1, chip_col2, chip_col3 = st.columns(3)
    
    if chip_col1.button("🐍 Explain Async IO in Python"):
        st.session_state.pending_input = "Can you explain how Async IO works in Python with a clear code example?"
        st.rerun()
        
    if chip_col2.button("🚀 Rust vs Go comparison"):
        st.session_state.pending_input = "Compare Rust vs Go in terms of performance, concurrency, and web backend development."
        st.rerun()
        
    if chip_col3.button("🛡️ Web API Security checklist"):
        st.session_state.pending_input = "What are the essential security best practices for designing REST APIs?"
        st.rerun()
    
    st.divider()

# ---------------------------------------------------------------------------
# Render Previous Conversation History
# ---------------------------------------------------------------------------
for message in st.session_state.history:
    avatar = "🤖" if message["role"] == ROLE_ASSISTANT else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        if message.get("persona"):
            st.markdown(
                f'<span class="persona-pill">{message["persona"]}</span>',
                unsafe_allow_html=True,
            )
        st.markdown(message["content"])

# ---------------------------------------------------------------------------
# Handle Input & Response Generation
# ---------------------------------------------------------------------------
chat_input = st.chat_input("Ask a technical question, paste code, or request an architectural review...")

# Determine effective user input (either typed or chip selection)
raw_input = chat_input or st.session_state.pending_input

if raw_input:
    # Reset pending input trigger immediately
    st.session_state.pending_input = None
    
    try:
        user_input = clean_user_input(raw_input)
    except ValueError as error:
        with st.chat_message(ROLE_ASSISTANT, avatar="🤖"):
            st.warning(str(error))
    else:
        # Display user input in UI
        with st.chat_message(ROLE_USER, avatar="👤"):
            st.markdown(user_input)
            
        messages = build_messages(
            user_input=user_input,
            history=st.session_state.history,
        )

        with st.chat_message(ROLE_ASSISTANT, avatar="🤖"):
            st.markdown(
                f'<span class="persona-pill">{persona}</span>',
                unsafe_allow_html=True,
            )

            if not is_valid_messages(messages):
                reply = get_fallback_message()
                st.markdown(reply)
            else:
                status_placeholder = st.empty()
                with status_placeholder.status("⚙️ Processing prompt & reasoning...", expanded=True) as status:
                    prompt = messages_to_text(messages)
                    
                    # Inject Persona modifier to prompt
                    persona_prompt = f"[System instruction: Adopt a {persona} style for this answer]\n\n{prompt}"
                    
                    raw_reply = get_ai_response(persona_prompt)
                    status.update(label="Response generated!", state="complete", expanded=False)
                
                # Clear status indicator box before streaming text
                status_placeholder.empty()

                if not raw_reply or not raw_reply.strip():
                    reply = get_fallback_message()
                    st.markdown(reply)
                else:
                    reply = raw_reply.strip()
                    words = reply.split(" ")
                    chunks = (word + " " for word in words)
                    st.write_stream(chunks)

        # Update Session History
        st.session_state.history.append({"role": ROLE_USER, "content": user_input})
        st.session_state.history.append(
            {"role": ROLE_ASSISTANT, "content": reply, "persona": persona}
        )

        # Memory Window Management
        if len(st.session_state.history) > MAX_HISTORY_MESSAGES:
            st.session_state.history = st.session_state.history[-MAX_HISTORY_MESSAGES:]