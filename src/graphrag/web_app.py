import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Set
from datetime import datetime, date

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
from pyvis.network import Network
import networkx as nx
from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    KnowledgeGraphIndex,
    get_response_synthesizer,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Load environment variables
load_dotenv()

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
    st.stop()

# Configure LlamaIndex settings
Settings.llm = OpenAI(
    model="gpt-4o",
    temperature=0,
    system_prompt=(
        "You are a helpful assistant that answers questions based on the knowledge graph. "
        "Always try to use the relationships and information from the graph in your answers. "
        "If you find relevant information, explain it clearly. "
        "If you don't find relevant information, say so explicitly."
    )
)
Settings.embed_model = OpenAIEmbedding()
Settings.show_progress = True  # Enable progress indicators

def process_uploaded_files(uploaded_files, document_dates):
    """Process uploaded files and return temporary file paths."""
    temp_dir = tempfile.mkdtemp()
    temp_files = []
    
    for file, doc_date in zip(uploaded_files, document_dates):
        # Save to temporary file
        temp_path = os.path.join(temp_dir, file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getvalue())
        temp_files.append({
            'path': temp_path,
            'date': doc_date
        })
    
    return temp_files, temp_dir

def load_and_process_markdown(file_info_list):
    """Load and process markdown files into a knowledge graph."""
    # Create storage context
    storage_context = StorageContext.from_defaults()
    
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
    
    # Create knowledge graph
    kg_index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=20,
        include_embeddings=True,
        show_progress=True,
        include_metadata=True,
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

def extract_source_node_texts(response):
    """Extract source node texts from response in a clean way."""
    source_nodes = []
    if hasattr(response, 'source_nodes'):
        source_nodes = [n.node.get_content() for n in response.source_nodes]
    elif hasattr(response, 'metadata') and 'source_nodes' in response.metadata:
        source_nodes = [n.node.get_content() for n in response.metadata['source_nodes']]
    return source_nodes

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

def extract_entities_from_query(query: str, graph: nx.Graph) -> Set[str]:
    """Extract mentioned entities from query that exist in the graph."""
    entities = set()
    query_lower = query.lower()
    
    for node in graph.nodes():
        if str(node).lower() in query_lower:
            entities.add(node)
    
    return entities

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
                        st.session_state.query_engine = st.session_state.kg_index.as_query_engine(
                            response_mode="tree_summarize",
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
                    source_nodes = extract_source_node_texts(response)
                    
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
                        st.components.v1.html(html, height=600)
            else:
                # Display initial graph
                st.subheader("Knowledge Graph Visualization")
                G = st.session_state.graph
                net = create_pyvis_graph(G, node_dates=st.session_state.node_dates)
                net.save_graph("/tmp/graph.html")
                with open("/tmp/graph.html", 'r', encoding='utf-8') as f:
                    html = f.read()
                    st.components.v1.html(html, height=600)

if __name__ == "__main__":
    main()
