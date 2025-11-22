"""
Streamlit UI for Bot GPT Backend

A user-friendly web interface to interact with the Bot GPT backend API.
Supports conversations, document uploads, and RAG-based chats.
"""
import streamlit as st
import requests
from typing import Dict, Any, List
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd

# Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS for professional styling
def load_custom_css():
    st.markdown("""
    <style>
    /* Main app styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Custom header styling */
    .custom-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .custom-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .custom-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    /* Status indicators */
    .status-success {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
    }
    
    .status-error {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
    }
    
    /* Document cards */
    .doc-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
        transition: transform 0.2s;
    }
    
    .doc-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

def check_backend_health() -> bool:
    """Check if the backend is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_health_info() -> Dict[str, Any]:
    """Get backend health information."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json() if response.status_code == 200 else {}
    except:
        return {}

def upload_document(user_id: int, title: str, content: str) -> Dict[str, Any]:
    """Upload a document to the backend."""
    payload = {
        "user_id": user_id,
        "title": title,
        "content": content
    }
    try:
        response = requests.post(f"{API_BASE_URL}/documents", json=payload, timeout=60)
        return {"success": response.status_code == 201, "data": response.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}

def list_documents(user_id: int) -> Dict[str, Any]:
    """List user's documents."""
    try:
        response = requests.get(f"{API_BASE_URL}/documents", params={"user_id": user_id}, timeout=10)
        return {"success": response.status_code == 200, "data": response.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}

def start_conversation(user_id: int, title: str, mode: str, first_message: str, document_ids: List[int] = None) -> Dict[str, Any]:
    """Start a new conversation."""
    payload = {
        "user_id": user_id,
        "title": title,
        "mode": mode,
        "first_message": first_message,
        "document_ids": document_ids or []
    }
    try:
        response = requests.post(f"{API_BASE_URL}/conversations", json=payload, timeout=60)
        return {"success": response.status_code == 201, "data": response.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}

def add_message(conversation_id: int, user_id: int, content: str) -> Dict[str, Any]:
    """Add a message to an existing conversation."""
    payload = {"content": content}
    try:
        response = requests.post(
            f"{API_BASE_URL}/conversations/{conversation_id}/messages",
            json=payload,
            params={"user_id": user_id},
            timeout=60
        )
        return {"success": response.status_code == 201, "data": response.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}

def list_conversations(user_id: int) -> Dict[str, Any]:
    """List user's conversations."""
    try:
        response = requests.get(f"{API_BASE_URL}/conversations", params={"user_id": user_id}, timeout=10)
        return {"success": response.status_code == 200, "data": response.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_conversation(conversation_id: int, user_id: int) -> Dict[str, Any]:
    """Get full conversation history."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/conversations/{conversation_id}",
            params={"user_id": user_id},
            timeout=10
        )
        return {"success": response.status_code == 200, "data": response.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}

def delete_conversation(conversation_id: int, user_id: int) -> Dict[str, Any]:
    """Delete a conversation."""
    try:
        response = requests.delete(
            f"{API_BASE_URL}/conversations/{conversation_id}",
            params={"user_id": user_id},
            timeout=10
        )
        return {"success": response.status_code == 200, "message": "Conversation deleted successfully"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def generate_conversation_title(messages: List[Dict]) -> str:
    """Generate a meaningful title from conversation messages."""
    if not messages:
        return "Empty Conversation"
    
    # Get the first user message
    first_user_msg = None
    for msg in messages:
        if msg.get('role') == 'user':
            first_user_msg = msg.get('content', '')
            break
    
    if not first_user_msg:
        return "New Conversation"
    
    # Create title from first message (max 50 chars)
    title = first_user_msg.strip()
    if len(title) > 50:
        title = title[:47] + "..."
    
    # Remove newlines and extra spaces
    title = ' '.join(title.split())
    
    return title if title else "New Conversation"

def create_user(name: str, email: str) -> Dict[str, Any]:
    """Create a new user."""
    payload = {"name": name, "email": email}
    try:
        response = requests.post(f"{API_BASE_URL}/users", json=payload, timeout=10)
        return {"success": response.status_code == 201, "data": response.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_user(user_id: int) -> Dict[str, Any]:
    """Get user by ID."""
    try:
        response = requests.get(f"{API_BASE_URL}/users/{user_id}", timeout=10)
        return {"success": response.status_code == 200, "data": response.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}

def ensure_user_exists(user_id: int) -> bool:
    """Ensure user exists, create if not."""
    # Check if user exists
    result = get_user(user_id)
    if result["success"]:
        return True
    
    # User doesn't exist, create it
    st.warning(f"User {user_id} doesn't exist. Creating user...")
    create_result = create_user(f"User {user_id}", f"user{user_id}@example.com")
    if create_result["success"]:
        st.success(f"✅ Created user {user_id}")
        return True
    else:
        st.error(f"❌ Failed to create user: {create_result.get('error', 'Unknown error')}")
        return False

def create_dashboard_metrics(user_id: int):
    """Create interactive dashboard metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    # Get data for metrics
    docs_result = list_documents(user_id)
    convs_result = list_conversations(user_id)
    health_info = get_health_info()
    
    docs_count = len(docs_result["data"].get("documents", [])) if docs_result["success"] else 0
    convs_count = len(convs_result["data"].get("conversations", [])) if convs_result["success"] else 0
    
    with col1:
        st.metric(
            label="📄 Documents",
            value=docs_count,
            delta=f"+{docs_count} uploaded" if docs_count > 0 else "No documents yet"
        )
    
    with col2:
        st.metric(
            label="💬 Conversations", 
            value=convs_count,
            delta=f"+{convs_count} started" if convs_count > 0 else "No conversations yet"
        )
    
    with col3:
        st.metric(
            label="🔄 Backend Status",
            value="Online",
            delta="Healthy" if check_backend_health() else "Offline"
        )
    
    with col4:
        llm_status = "Ready" if health_info.get('llm_configured') else "Not Ready"
        st.metric(
            label="🤖 LLM Status",
            value=llm_status,
            delta="Model: llama-3.1-8b-instant" if health_info.get('llm_configured') else "Not configured"
        )

def create_analytics_chart(user_id: int):
    """Create analytics visualization."""
    convs_result = list_conversations(user_id)
    if not convs_result["success"] or not convs_result["data"].get("conversations"):
        st.info("📊 No conversation data available for analytics yet.")
        return
    
    conversations = convs_result["data"]["conversations"]
    
    # Create conversation timeline
    df = pd.DataFrame(conversations)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['date'] = df['created_at'].dt.date
    
    # Count conversations by date
    daily_counts = df.groupby('date').size().reset_index(name='conversations')
    
    # Create interactive chart
    fig = px.line(
        daily_counts, 
        x='date', 
        y='conversations',
        title='📈 Conversation Activity Over Time',
        markers=True
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333333'),
        title_font_size=20
    )
    fig.update_traces(
        line_color='#667eea',
        marker_color='#764ba2'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(
        page_title="🤖 Bot GPT - AI Assistant Platform",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state FIRST
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None
    if "current_mode" not in st.session_state:
        st.session_state.current_mode = "open"
    if "selected_docs" not in st.session_state:
        st.session_state.selected_docs = []
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "📊 Chat Info"
    
    # Load custom CSS
    load_custom_css()
    
    # Professional header
    st.markdown("""
    <div class="custom-header">
        <h1>🤖 Bot GPT AI Assistant</h1>
        <p>Advanced Conversational AI with RAG-powered Document Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for user settings and navigation
    with st.sidebar:
        st.markdown("### 👤 User Settings")
        user_id = st.number_input("User ID", min_value=1, value=1, help="Enter your user ID")
        
        # Show user status
        user_result = get_user(user_id)
        if user_result["success"]:
            st.success(f"✅ User {user_id} exists")
        else:
            st.warning(f"⚠️ User {user_id} not found")
            st.caption("💡 User will be created automatically when needed")
        
        st.markdown("---")
        st.markdown("### 🎛️ System Status")
        
        # Backend health check
        if check_backend_health():
            st.success("✅ Backend Online")
            health_info = get_health_info()
            st.info(f"📊 Version: {health_info.get('version', 'Unknown')}")
            
            llm_configured = health_info.get('llm_configured', False)
            if llm_configured:
                st.success("🤖 LLM Ready")
                st.caption("Model: llama-3.1-8b-instant")
            else:
                st.warning("⚠️ LLM Not Configured")
        else:
            st.error("❌ Backend Offline")
            st.code("uvicorn app.main:app --reload")
            st.stop()
        
        st.markdown("---")
        st.markdown("### 🚀 Quick Actions")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_conversation_id = None
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 💬 Recent Conversations")
        
        # Show last 3 conversations for quick access
        convs_result = list_conversations(user_id)
        if convs_result["success"] and convs_result["data"].get("conversations"):
            recent_convs = convs_result["data"]["conversations"][:3]  # Last 3
            for conv in recent_convs:
                conv_result = get_conversation(conv['id'], user_id)
                if conv_result["success"]:
                    messages = conv_result["data"]["messages"]
                    conv_title = generate_conversation_title(messages)
                    
                    # Truncate title for sidebar
                    display_title = conv_title[:25] + "..." if len(conv_title) > 25 else conv_title
                    
                    if st.button(f"🔄 {display_title}", key=f"sidebar_resume_{conv['id']}", use_container_width=True, help=f"Resume: {conv_title}"):
                        # Load conversation into current session
                        st.session_state.current_conversation_id = conv['id']
                        st.session_state.current_mode = conv['mode']
                        st.session_state.messages = messages
                        if conv['mode'] == 'rag':
                            st.session_state.selected_docs = []
                        st.success(f"✅ Resumed: {conv_title}")
                        st.rerun()
        else:
            st.caption("No recent conversations")
    
    # Main dashboard metrics
    st.markdown("### 📊 Dashboard Overview")
    create_dashboard_metrics(user_id)
    
    
    st.markdown("---")
    
    # Enhanced navigation with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Chat Assistant", 
        "📄 Document Manager", 
        "� Conversation History", 
        "🔧 Settings"
    ])
    
    with tab1:
        # Chat configuration
        st.markdown("### 🎛️ Chat Configuration")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            mode = st.selectbox("Chat Mode", ["open", "rag"], 
                               index=0 if st.session_state.current_mode == "open" else 1,
                               help="Open: General chat, RAG: Document-based chat")
            st.session_state.current_mode = mode
        
        with col2:
            # Document selection for RAG mode
            if mode == "rag":
                docs_result = list_documents(user_id)
                if docs_result["success"] and docs_result["data"].get("documents"):
                    docs = docs_result["data"]["documents"]
                    doc_options = {f"{doc['title']} (ID: {doc['id']})": doc['id'] for doc in docs}
                    selected_doc_names = st.multiselect("Choose documents:", list(doc_options.keys()))
                    st.session_state.selected_docs = [doc_options[name] for name in selected_doc_names]
                    
                    if st.session_state.selected_docs:
                        st.success(f"✅ {len(st.session_state.selected_docs)} document(s) selected for RAG")
                else:
                    st.warning("⚠️ No documents found. Upload some documents first!")
                    st.session_state.selected_docs = []
            else:
                st.session_state.selected_docs = []
                st.info("💡 Open mode: General conversation without document context")
        
        st.markdown("---")
        
        # Show conversation status if resumed
        if st.session_state.current_conversation_id:
            st.info(f"🔄 **Continuing Conversation ID: {st.session_state.current_conversation_id}** | Mode: {st.session_state.current_mode.upper()}")
            if st.button("🆕 Start New Conversation", type="secondary"):
                st.session_state.current_conversation_id = None
                st.session_state.messages = []
                st.session_state.selected_docs = []
                st.success("✅ Started new conversation!")
                st.rerun()
        
        # Enhanced chat display
        st.markdown("### 💬 Conversation")
        
        # Chat statistics
        if st.session_state.messages:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Messages", len(st.session_state.messages))
            with col2:
                user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
                st.metric("Your Messages", user_msgs)
            with col3:
                ai_msgs = len([m for m in st.session_state.messages if m["role"] == "assistant"])
                st.metric("AI Responses", ai_msgs)
            
            st.markdown("---")
        
        # Chat container with enhanced styling
        chat_container = st.container()
        with chat_container:
            if not st.session_state.messages:
                st.markdown("""
                <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin: 1rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <h3 style="color: white; margin-bottom: 1rem; font-size: 1.8rem;">👋 Welcome to Bot GPT!</h3>
                    <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem; margin-bottom: 0.5rem;">Start a conversation by typing a message below.</p>
                    <p style="color: rgba(255,255,255,0.8); font-size: 1rem;"><strong>💡 Tip:</strong> Use RAG mode to chat with your uploaded documents!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                for i, message in enumerate(st.session_state.messages):
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        
                        # Add timestamp and message number
                        st.caption(f"Message #{i+1} • {message['role'].title()}")
        
        # Enhanced chat input with mode indicator
        mode_indicator = "🤖 RAG Mode" if mode == "rag" else "💬 Open Mode"
        st.markdown(f"**Current Mode:** {mode_indicator}")
        
        if mode == "rag" and st.session_state.selected_docs:
            st.markdown(f"**Active Documents:** {len(st.session_state.selected_docs)} selected")
        
        # Chat input placeholder based on mode
        placeholder = "Ask about your documents..." if mode == "rag" else "Type your message here..."
    
    with tab2:
        st.markdown("### 📄 Document Manager")
        
        # Document upload section with enhanced UI
        st.markdown("#### 📤 Upload New Document")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            upload_method = st.radio(
                "Choose upload method:", 
                ["📁 Upload File", "✏️ Paste Text"], 
                horizontal=True
            )
        
        with col2:
            st.markdown("**Supported Formats:**")
            st.caption("📄 .txt, .md, .py, .json, .csv, .log")
        
        doc_title = st.text_input(
            "Document Title", 
            placeholder="Enter a descriptive title for your document..."
        )
        
        doc_content = ""
        
        if upload_method == "📁 Upload File":
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['txt', 'md', 'py', 'json', 'csv', 'log'],
                help="Upload a text file to add to your knowledge base"
            )
            if uploaded_file is not None:
                try:
                    doc_content = uploaded_file.read().decode('utf-8')
                    if not doc_title:
                        doc_title = uploaded_file.name
                    
                    # File preview with statistics
                    st.markdown("#### 📄 File Preview")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Characters", len(doc_content))
                    with col2:
                        st.metric("Words", len(doc_content.split()))
                    with col3:
                        st.metric("Lines", len(doc_content.split('\n')))
                    
                    with st.expander("📖 Content Preview"):
                        st.text_area(
                            "First 1000 characters:",
                            value=doc_content[:1000] + ("..." if len(doc_content) > 1000 else ""),
                            height=200,
                            disabled=True
                        )
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
                    doc_content = ""
        else:  # Paste Text
            doc_content = st.text_area(
                "Document Content", 
                height=300, 
                placeholder="Paste your document content here..."
            )
            
            if doc_content:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Characters", len(doc_content))
                with col2:
                    st.metric("Words", len(doc_content.split()))
                with col3:
                    st.metric("Lines", len(doc_content.split('\n')))
        
        upload_disabled = not (doc_title.strip() and doc_content.strip())
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("📤 Upload Document", disabled=upload_disabled, use_container_width=True):
                if doc_title.strip() and doc_content.strip():
                    # Ensure user exists before uploading document
                    if not ensure_user_exists(user_id):
                        st.error("❌ Failed to create/verify user. Please try again.")
                        st.stop()
                    
                    with st.spinner("🚀 Processing document..."):
                        progress_bar = st.progress(0)
                        progress_bar.progress(25, "Uploading...")
                        
                        result = upload_document(user_id, doc_title.strip(), doc_content.strip())
                        progress_bar.progress(75, "Creating chunks...")
                        
                        if result["success"]:
                            progress_bar.progress(100, "Complete!")
                            st.success(f"✅ Document uploaded successfully!")
                            st.info(f"📊 Document ID: {result['data']['document_id']}")
                            st.info(f"📄 Chunks created: {result['data']['num_chunks']}")
                            st.balloons()
                            time.sleep(1)
                            progress_bar.empty()
                        else:
                            progress_bar.empty()
                            error_msg = result.get('error', 'Unknown error')
                            st.error(f"❌ Upload failed: {error_msg}")
                            
                            # If it's a foreign key error, suggest using user ID 1
                            if "FOREIGN KEY constraint failed" in error_msg:
                                st.warning("💡 Try using User ID 1, or create a new user first.")
                else:
                    st.warning("⚠️ Please provide both title and content!")
        
        with col2:
            if upload_disabled:
                st.caption("💡 Fill in both title and content to enable upload")
        
        st.markdown("---")
        
        # Enhanced document list
        st.markdown("#### 📋 Your Document Library")
        
        docs_result = list_documents(user_id)
        if docs_result["success"]:
            docs_data = docs_result["data"]
            if docs_data.get("documents"):
                # Document statistics
                total_docs = len(docs_data["documents"])
                st.markdown(f"**📊 Total Documents: {total_docs}**")
                
                # Search and filter
                search_term = st.text_input("🔍 Search documents:", placeholder="Search by title...")
                
                filtered_docs = docs_data["documents"]
                if search_term:
                    filtered_docs = [doc for doc in docs_data["documents"] 
                                   if search_term.lower() in doc['title'].lower()]
                
                if filtered_docs:
                    for doc in filtered_docs:
                        with st.expander(f"📄 {doc['title']}", expanded=False):
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.markdown(f"**ID:** {doc['id']}")
                                st.markdown(f"**Created:** {doc['created_at']}")
                                st.markdown(f"**Chunks:** {doc.get('num_chunks', 'N/A')}")
                            with col2:
                                if st.button(f"🗑️ Delete", key=f"del_{doc['id']}", use_container_width=True):
                                    st.warning("Delete functionality coming soon!")
                else:
                    st.info("🔍 No documents match your search.")
            else:
                st.markdown("""
                <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <h4 style="color: white; margin-bottom: 1rem;">📚 No documents yet</h4>
                    <p style="color: rgba(255,255,255,0.9);">Upload your first document to get started with RAG-powered conversations!</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error(f"Failed to load documents: {docs_result.get('error', 'Unknown error')}")
    
    with tab3:
        st.markdown("### � Conversation History")
        
        convs_result = list_conversations(user_id)
        if convs_result["success"]:
            conversations = convs_result["data"].get("conversations", [])
            if conversations:
                st.markdown(f"**� Total Conversations: {len(conversations)}**")
                
                # Search conversations
                search_conv = st.text_input("🔍 Search conversations:", placeholder="Search by content...")
                
                filtered_convs = conversations
                if search_conv:
                    # Filter by searching in conversation titles or first messages
                    filtered_convs = []
                    for conv in conversations:
                        # Get conversation details to search in messages
                        conv_result = get_conversation(conv['id'], user_id)
                        if conv_result["success"]:
                            messages = conv_result["data"]["messages"]
                            conv_title = generate_conversation_title(messages)
                            
                            # Search in title or first few messages
                            search_text = search_conv.lower()
                            if (search_text in conv_title.lower() or 
                                any(search_text in msg.get('content', '').lower() for msg in messages[:3])):
                                filtered_convs.append(conv)
                
                if filtered_convs:
                    for conv in filtered_convs:
                        # Get conversation messages to generate proper title
                        conv_result = get_conversation(conv['id'], user_id)
                        if conv_result["success"]:
                            messages = conv_result["data"]["messages"]
                            conv_title = generate_conversation_title(messages)
                            
                            with st.expander(f"💬 {conv_title} ({conv['mode'].upper()})", expanded=False):
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.markdown(f"**ID:** {conv['id']}")
                                    st.markdown(f"**Created:** {conv['created_at']}")
                                    st.markdown(f"**Mode:** {conv['mode']}")
                                    st.markdown(f"**Messages:** {len(messages)}")
                                
                                with col2:
                                    # Action buttons
                                    col_resume, col_view, col_delete = st.columns(3)
                                    
                                    with col_resume:
                                        if st.button(f"🔄 Resume", key=f"resume_{conv['id']}", use_container_width=True, type="primary"):
                                            # Load conversation into current session
                                            st.session_state.current_conversation_id = conv['id']
                                            st.session_state.current_mode = conv['mode']
                                            st.session_state.messages = messages
                                            
                                            # If RAG mode, load associated documents
                                            if conv['mode'] == 'rag':
                                                # Get documents associated with this conversation
                                                # For now, we'll clear selected docs and let user reselect
                                                st.session_state.selected_docs = []
                                            
                                            st.success(f"✅ Resumed conversation: {conv_title}")
                                            st.info("💡 Go to the Chat Assistant tab to continue the conversation")
                                            st.rerun()
                                    
                                    with col_view:
                                        if st.button(f"👁️ View", key=f"view_{conv['id']}", use_container_width=True):
                                            st.markdown("**Messages:**")
                                            for i, msg in enumerate(messages):
                                                role_icon = "👤" if msg['role'] == 'user' else "🤖"
                                                with st.container():
                                                    st.markdown(f"**{role_icon} {msg['role'].title()}:**")
                                                    st.markdown(f"> {msg['content']}")
                                                    st.caption(f"Message {i+1}")
                                                    if i < len(messages) - 1:
                                                        st.markdown("---")
                                    
                                    with col_delete:
                                        if st.button(f"🗑️ Delete", key=f"del_{conv['id']}", use_container_width=True, type="secondary"):
                                            # Confirmation dialog
                                            if st.button(f"⚠️ Confirm Delete", key=f"confirm_del_{conv['id']}", type="secondary"):
                                                delete_result = delete_conversation(conv['id'], user_id)
                                                if delete_result["success"]:
                                                    st.success("✅ Conversation deleted!")
                                                    st.rerun()
                                                else:
                                                    st.error(f"❌ Delete failed: {delete_result.get('error', 'Unknown error')}")
                else:
                    st.info("🔍 No conversations match your search.")
            else:
                st.markdown("""
                <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <h4 style="color: white; margin-bottom: 1rem;">💬 No conversations yet</h4>
                    <p style="color: rgba(255,255,255,0.9);">Start your first conversation in the Chat Assistant tab!</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error(f"Failed to load conversations: {convs_result.get('error', 'Unknown error')}")
    
    with tab4:
        st.markdown("### � Application Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎛️ Chat Preferences")
            
            # Theme selection (placeholder)
            theme = st.selectbox("🎨 Interface Theme", ["Professional", "Dark Mode", "Light Mode"])
            
            # Auto-scroll
            auto_scroll = st.checkbox("📜 Auto-scroll chat messages", value=True)
            
            # Message limit
            msg_limit = st.slider("💬 Message history limit", 10, 100, 50)
            
            # Timeout settings
            timeout_setting = st.slider("⏱️ Request timeout (seconds)", 30, 120, 60)
        
        with col2:
            st.markdown("#### 📊 Application Info")
            
            st.markdown("**🤖 Bot GPT AI Assistant**")
            st.markdown("Advanced Conversational AI Platform")
            st.markdown("Built with Streamlit & FastAPI")
            
            st.markdown("**🔧 Technical Stack:**")
            st.code("""
• Frontend: Streamlit + Plotly
• Backend: FastAPI + SQLAlchemy  
• AI: Groq LLM (llama-3.1-8b-instant)
• Vector DB: FAISS + SentenceTransformers
• Database: SQLite
            """)
        
        st.markdown("---")
        
        # Application controls
        st.markdown("#### 🎮 Application Controls")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Refresh All Data", use_container_width=True):
                st.success("✅ Data refreshed!")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Chat History", type="secondary", use_container_width=True):
                st.session_state.messages = []
                st.session_state.current_conversation_id = None
                st.success("✅ Chat history cleared!")
                st.rerun()
        
        with col3:
            if st.button("� Reset Application", type="secondary", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("✅ Application reset!")
                st.rerun()
    
    # Enhanced chat input with faster response handling
    if prompt := st.chat_input("Type your message here..."):
        # Ensure user exists before starting conversation
        if not ensure_user_exists(user_id):
            st.stop()
        
        # Add user message to chat immediately for better UX
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response with enhanced error handling and progress indication
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🤔 Thinking...")
            
            try:
                if st.session_state.current_conversation_id is None:
                    # Start new conversation
                    title = f"Chat {len(st.session_state.messages)//2 + 1}"
                    result = start_conversation(user_id, title, st.session_state.current_mode, prompt, st.session_state.selected_docs)
                else:
                    # Continue existing conversation  
                    result = add_message(st.session_state.current_conversation_id, user_id, prompt)
                
                if result["success"]:
                    if "conversation_id" in result["data"]:
                        st.session_state.current_conversation_id = result["data"]["conversation_id"]
                    
                    messages = result["data"]["messages"]
                    assistant_message = messages[-1]["content"]  # Last message is assistant's response
                    
                    # Display the response with typing effect simulation
                    message_placeholder.markdown("✨ Generating response...")
                    time.sleep(0.5)  # Brief pause for better UX
                    message_placeholder.markdown(assistant_message)
                    
                    # Add to session state
                    st.session_state.messages.append({"role": "assistant", "content": assistant_message})
                    
                else:
                    error_msg = result.get("error", "Unknown error")
                    message_placeholder.markdown(f"❌ **Error:** {error_msg}")
                    st.error(f"Failed to get response: {error_msg}")
                    
            except Exception as e:
                message_placeholder.markdown(f"❌ **Connection Error:** {str(e)}")
                st.error(f"Connection failed: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #666; font-size: 0.9rem;">
        🤖 <strong>Bot GPT AI Assistant</strong> | Built with Streamlit & FastAPI | 
        <a href="https://github.com" target="_blank">View Source</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
