import os
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import tenacity
from openai import OpenAI as OpenAIClient, RateLimitError

import streamlit as st
import networkx as nx
from pyvis.network import Network
from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    KnowledgeGraphIndex,
    StorageContext,
    Settings,
    get_response_synthesizer,
    ServiceContext,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.base import CBEventType
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.response_synthesizers import TreeSummarize

# Load environment variables
load_dotenv()

# Configure OpenAI client with retries and rate limiting
def create_openai_client():
    return OpenAIClient(
        timeout=60.0,  # 60 second timeout
        max_retries=5,  # Retry failed requests up to 5 times
        # Default exponential backoff: ~2^n seconds between retries
    )

# Configure retry strategy for LlamaIndex
retry_decorator = tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),  # Wait between 4 and 10 seconds
    stop=tenacity.stop_after_attempt(5),  # Stop after 5 attempts
    retry=tenacity.retry_if_exception_type(RateLimitError),  # Only retry rate limit errors
)

# Configure LlamaIndex settings with retries and parallel processing
Settings.llm = OpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    max_retries=5,
    timeout=60,
    retry_decorator=retry_decorator,
)
Settings.embed_model = OpenAIEmbedding(
    api_key=os.getenv("OPENAI_API_KEY"),
    max_retries=5,
    timeout=60,
    retry_decorator=retry_decorator,
)
Settings.chunk_size = 1024
Settings.chunk_overlap = 128  # Add overlap between chunks to maintain context

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
    st.stop()

def process_uploaded_files(uploaded_files, document_dates):
    """Process uploaded files and return temporary file paths."""
    temp_dir = tempfile.mkdtemp()
    temp_files = []
    
    for uploaded_file, doc_date in zip(uploaded_files, document_dates):
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        temp_files.append({
            'path': file_path,
            'date': doc_date
        })
    
    return temp_files, temp_dir

class ProgressCallbackHandler(BaseCallbackHandler):
    """Callback handler to track progress and display sample triplets."""
    
    def __init__(self):
        # Initialize with empty sets since we want to handle all events
        self.event_starts_to_ignore = set()
        self.event_ends_to_ignore = set()
        self.progress_bar = st.progress(0)
        self.status_container = st.empty()
        self.header_placeholder = st.empty()  # For the sample pane header
        self.graph_placeholder = st.empty()  # Container to hold the graph location
        self.total_chunks = 17  # We know there are 17 chunks
        self.processed_chunks = 0
        self.sample_triplets = []
        self.status_container.text(f"Processing {self.total_chunks} chunks...")
        
    def _create_network_viz(self, triplets):
        """Create a network visualization of the triplets using NetworkX and Pyvis."""
        import networkx as nx
        from pyvis.network import Network
        import tempfile
        import os
        
        # Create a new directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges for each triplet
        for s, p, o in triplets:
            # Clean node names
            s_clean = s.replace('"', '').replace("'", "")[:50]  # Truncate long names
            o_clean = o.replace('"', '').replace("'", "")[:50]
            p_clean = p.replace('"', '').replace("'", "")[:30]
            
            # Add nodes with styling
            G.add_node(s_clean, color='lightblue', shape='box')
            G.add_node(o_clean, color='lightgreen', shape='box')
            G.add_edge(s_clean, o_clean, title=p_clean)
        
        # Convert to Pyvis network with remote CDN
        net = Network(notebook=True, directed=True, height="400px", width="100%", cdn_resources='remote')
        net.from_nx(G)
        
        # Set options for better visualization
        net.set_options("""
        var options = {
            "nodes": {
                "font": {"size": 16},
                "shape": "box",
                "shadow": true
            },
            "edges": {
                "font": {"size": 14},
                "arrows": {"to": {"enabled": true}},
                "color": {"inherit": false},
                "smooth": {"type": "curvedCW", "roundness": 0.2}
            },
            "physics": {
                "barnesHut": {"gravitationalConstant": -15000},
                "stabilization": {"iterations": 50}
            }
        }
        """)
        
        # Save to a temporary file and read back
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
            net.save_graph(f.name)
            with open(f.name, 'r', encoding='utf-8') as f:
                html_content = f.read()
            os.unlink(f.name)  # Clean up temp file
            
        return html_content
        
    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Start a trace - required abstract method."""
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """End a trace - required abstract method."""
        pass
        
    def on_event_start(self, event_type: CBEventType, payload: Optional[Dict[str, Any]] = None, event_id: Optional[str] = None, **kwargs: Any) -> None:
        """Handle the start of an event."""
        if event_type == CBEventType.EMBEDDING:
            if self.processed_chunks == 0:
                self.status_container.text(f"Extracting triplets and generating embeddings for {self.total_chunks} chunks...")
    
    def on_event_end(self, event_type: CBEventType, payload: Optional[Dict[str, Any]] = None, event_id: Optional[str] = None, **kwargs: Any) -> None:
        """Handle the end of an event."""
        if event_type == CBEventType.LLM:
            if payload and "response" in payload:
                response = str(payload["response"])
                # Look for triplets in the response
                lines = response.split('\n')
                for line in lines:
                    if '(' in line and ')' in line:
                        # Extract the triplet text between parentheses
                        triplet_text = line[line.find("(")+1:line.find(")")]
                        parts = [p.strip() for p in triplet_text.split(",")]
                        if len(parts) == 3:
                            # Add triplet and update visualization immediately
                            self.sample_triplets.append(tuple(parts))
                            if len(self.sample_triplets) > 15:  # Keep last 15 triplets
                                self.sample_triplets = self.sample_triplets[-15:]
                            
                            # Update network visualization with header
                            if len(self.sample_triplets) > 0:
                                self.header_placeholder.markdown("### Sample Triplets Extracted")
                                html_content = self._create_network_viz(self.sample_triplets[-5:])  # Show last 5 triplets
                                with self.graph_placeholder:
                                    st.components.v1.html(html_content, height=400, scrolling=True)

        elif event_type == CBEventType.EMBEDDING:
            if payload and "chunks" in payload:
                chunks = payload["chunks"]
                if isinstance(chunks, list):
                    self.processed_chunks += 1
                    progress = min(self.processed_chunks / self.total_chunks, 1.0)
                    self.progress_bar.progress(progress)
                    self.status_container.text(f"Processing {self.processed_chunks}/{self.total_chunks} chunks")
                    
                    # Clear sample pane when processing is complete
                    if self.processed_chunks == self.total_chunks:
                        self.header_placeholder.empty()
                        self.graph_placeholder.empty()
                    
def load_and_process_markdown(file_info_list):
    """Load and process markdown files into a knowledge graph."""
    try:
        # Initialize callback handler for progress tracking
        progress_handler = ProgressCallbackHandler()
        callback_manager = CallbackManager([progress_handler])

        # Configure global settings
        Settings.callback_manager = callback_manager

        # Load documents with metadata
        documents = []
        for file_info in file_info_list:
            reader = SimpleDirectoryReader(
                input_files=[file_info['path']],
            )
            docs = reader.load_data()
            # Add document date to metadata
            for doc in docs:
                doc.metadata["document_date"] = file_info['date'].isoformat()
            documents.extend(docs)
        
        # Create storage context
        storage_context = StorageContext.from_defaults()
        
        # Create the knowledge graph index
        kg_index = KnowledgeGraphIndex.from_documents(
            documents,
            storage_context=storage_context,
            max_triplets_per_chunk=None,  # Let LlamaIndex determine optimal number
            include_embeddings=True,
            show_progress=True,
            include_metadata=True,
            # Configure triplet extraction
            kg_triple_extract_template="Extract relevant relationships from the text. Only extract relationships that are explicitly stated. Do not infer relationships. Extract as many or as few as are appropriate for the content.",
        )
        
        # Get the graph and add dates to all nodes
        graph = kg_index.get_networkx_graph()
        
        # Create a mapping of text to dates
        text_to_date = {}
        for doc in documents:
            doc_date = doc.metadata.get("document_date")
            if doc_date:
                text_to_date[doc.text] = doc_date
                text_to_date[doc.text.lower()] = doc_date
        
        # Store dates in a global dictionary that we can access during visualization
        node_dates = {}
        
        # Update node attributes with dates
        nodes_with_dates = 0
        for node_id in graph.nodes():
            node_str = str(node_id).lower()
            # Try to find a matching document
            for text, date in text_to_date.items():
                if node_str in text.lower():
                    node_dates[node_str] = date
                    nodes_with_dates += 1
                    break
        
        # Update the graph in the index
        kg_index._graph = graph
        
        return kg_index, node_dates
    
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        raise

def create_pyvis_graph(G, query=None, response_nodes=None, node_dates=None):
    """Create a Pyvis network visualization."""
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    
    # Determine highlight color for query-relevant nodes
    highlighted_nodes = set()
    if response_nodes:
        # Convert response nodes to lowercase for matching
        response_text = [text.lower() for text in response_nodes]
        # Find nodes that contain any of the response text
        for node_id in G.nodes():
            node_str = str(node_id).lower()
            if any(node_str in text for text in response_text):
                highlighted_nodes.add(node_id)
    
    for node_id in G.nodes():
        node_str = str(node_id).lower()
        doc_date = node_dates.get(node_str) if node_dates else None
        
        # Default color for query highlighting
        base_color = "#ff3333" if node_id in highlighted_nodes else "#97c2fc"
        
        # Calculate opacity based on date recency
        opacity = 1.0
        if doc_date:
            try:
                doc_date = datetime.fromisoformat(doc_date).date()
                today = date.today()
                days_old = (today - doc_date).days
                opacity = max(0.3, 1.0 - min(days_old, 365) / 365)
            except (ValueError, TypeError):
                pass
        
        # Convert base color to rgba with calculated opacity
        if base_color.startswith('#'):
            r = int(base_color[1:3], 16)
            g = int(base_color[3:5], 16)
            b = int(base_color[5:7], 16)
            color = f"rgba({r},{g},{b},{opacity})"
        else:
            color = base_color.replace('rgb', 'rgba').replace(')', f',{opacity})')
        
        net.add_node(
            node_id,
            label=node_id[:20] + "..." if len(node_id) > 20 else node_id,
            color=color,
            title=f"Date: {doc_date if doc_date else 'Unknown'}"
        )
    
    # Add edges
    for edge in G.edges(data=True):
        net.add_edge(edge[0], edge[1], title=edge[2].get("edge_label", ""))
    
    return net

def extract_entities_from_query(query: str, graph: nx.Graph) -> set:
    """Extract mentioned entities from query that exist in the graph."""
    entities = set()
    query_lower = query.lower()
    
    for node in graph.nodes():
        if str(node).lower() in query_lower:
            entities.add(node)
    
    return entities

def calculate_temporal_score(doc_date):
    """Calculate a score multiplier based on document age."""
    if not doc_date:
        return 0.5  # Default score for documents without dates
    
    try:
        doc_date = datetime.fromisoformat(doc_date).date()
        today = date.today()
        days_old = (today - doc_date).days
        # Scale from 1.0 (newest) to 0.3 (oldest, 365+ days)
        # This matches our visualization scaling
        return max(0.3, 1.0 - min(days_old, 365) / 365)
    except (ValueError, TypeError):
        return 0.5

class TemporalResponseSynthesizer(TreeSummarize):
    """Custom response synthesizer that incorporates temporal weights."""
    
    def __init__(self, node_dates, **kwargs):
        super().__init__(**kwargs)
        self.node_dates = node_dates
    
    def _get_nodes_for_response(self, query_bundle, nodes):
        """Override to incorporate temporal weighting into node selection."""
        nodes_with_scores = []
        for node in nodes:
            # Get base relevance score
            base_score = node.score if hasattr(node, 'score') else 0.0
            
            # Get temporal score
            node_text = node.node.get_content().lower()
            doc_date = self.node_dates.get(node_text)
            temporal_score = calculate_temporal_score(doc_date)
            
            # Combine scores - multiply base score by temporal factor
            final_score = base_score * temporal_score
            node.score = final_score
            nodes_with_scores.append(node)
        
        # Sort by combined score
        nodes_with_scores.sort(key=lambda x: x.score, reverse=True)
        return nodes_with_scores

def main():
    st.set_page_config(page_title="GraphRAG Explorer", layout="wide")
    
    # Initialize session state
    if 'kg_index' not in st.session_state:
        st.session_state.kg_index = None
    if 'graph' not in st.session_state:
        st.session_state.graph = None
    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = None
    if 'last_processed_files' not in st.session_state:
        st.session_state.last_processed_files = set()
    if 'document_dates' not in st.session_state:
        st.session_state.document_dates = {}
    if 'ready_to_process' not in st.session_state:
        st.session_state.ready_to_process = False
    if 'file_info_list' not in st.session_state:
        st.session_state.file_info_list = []
    if 'node_dates' not in st.session_state:
        st.session_state.node_dates = {}
    
    # File upload and date selection section
    with st.expander("📁 Upload Files & Set Dates", expanded=True):
        uploaded_files = st.file_uploader("Upload Markdown Files", 
                                        type=['md', 'txt'], 
                                        accept_multiple_files=True)
        
        # Process uploaded files
        if uploaded_files:
            current_filenames = {f.name for f in uploaded_files}
            files_changed = current_filenames != st.session_state.last_processed_files
            
            if files_changed:
                st.session_state.document_dates = {}
                st.session_state.ready_to_process = False
            
            # Date selection for each file
            temp_files = []
            all_dates_selected = True
            
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.document_dates:
                    st.session_state.document_dates[uploaded_file.name] = None
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(uploaded_file.name)
                with col2:
                    date = st.date_input(f"Date for {uploaded_file.name}", 
                                       key=f"date_{uploaded_file.name}",
                                       label_visibility="collapsed")
                    st.session_state.document_dates[uploaded_file.name] = date
                
                if date:
                    # Save file and add to processing list
                    temp_path = f"/tmp/{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    temp_files.append({"path": temp_path, "date": date})
                else:
                    all_dates_selected = False
            
            st.session_state.ready_to_process = all_dates_selected
            
            # Process button
            process_button = st.button("Process Files", 
                                     disabled=not st.session_state.ready_to_process,
                                     type="primary")
            
            if process_button:
                with st.spinner("Processing files..."):
                    try:
                        st.session_state.file_info_list = temp_files
                        kg_index, node_dates = load_and_process_markdown(st.session_state.file_info_list)
                        st.session_state.kg_index = kg_index
                        st.session_state.graph = st.session_state.kg_index.get_networkx_graph()
                        
                        # Create custom response synthesizer with temporal weighting
                        response_synthesizer = TemporalResponseSynthesizer(
                            node_dates=node_dates
                        )
                        
                        # Configure query engine with temporal weighting
                        st.session_state.query_engine = st.session_state.kg_index.as_query_engine(
                            response_mode="tree_summarize",
                            response_synthesizer=response_synthesizer,
                        )
                        st.session_state.node_dates = node_dates
                        
                        # Store processed filenames
                        st.session_state.last_processed_files = current_filenames
                        st.success("Files processed successfully!")
                        
                    except Exception as e:
                        st.error(f"Error processing files: {str(e)}")
    
    # Query section
    if st.session_state.query_engine is not None:
        with st.expander("🔍 Query Knowledge Graph", expanded=True):
            query = st.text_input("Enter your query:")
            
            if query:
                with st.spinner("Processing query..."):
                    # Get response from query engine
                    response = st.session_state.query_engine.query(query)
                    
                    # Extract source nodes and clean up the response
                    source_nodes = []
                    
                    # Display response
                    st.markdown("### Answer")
                    st.write(str(response).split('\n')[0])  # Only show the actual response text
                    
                    # Create and display interactive graph
                    st.subheader("Knowledge Graph Visualization")
                    G = st.session_state.graph
                    net = create_pyvis_graph(G, query, source_nodes, st.session_state.node_dates)
                    
                    # Save and display the graph
                    net.save_graph("/tmp/graph.html")
                    with open("/tmp/graph.html", 'r', encoding='utf-8') as f:
                        html = f.read()
                        st.components.v1.html(html, height=600, scrolling=True)
            else:
                # Display initial graph
                st.subheader("Knowledge Graph Visualization")
                G = st.session_state.graph
                net = create_pyvis_graph(G, node_dates=st.session_state.node_dates)
                net.save_graph("/tmp/graph.html")
                with open("/tmp/graph.html", 'r', encoding='utf-8') as f:
                    html = f.read()
                    st.components.v1.html(html, height=600, scrolling=True)

if __name__ == "__main__":
    main()
