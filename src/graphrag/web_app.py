import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Set

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

def process_uploaded_files(uploaded_files):
    """Process uploaded files and return temporary file paths."""
    temp_dir = tempfile.mkdtemp()
    temp_files = []
    
    for uploaded_file in uploaded_files:
        temp_path = Path(temp_dir) / uploaded_file.name
        temp_path.write_bytes(uploaded_file.getvalue())
        temp_files.append(temp_path)
    
    return temp_files, temp_dir

def load_and_process_markdown(input_files: List[Path]) -> KnowledgeGraphIndex:
    """Load and process markdown files into a knowledge graph."""
    documents = SimpleDirectoryReader(
        input_files=[str(f) for f in input_files]
    ).load_data()
    
    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(documents)
    
    storage_context = StorageContext.from_defaults()
    
    kg_index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=20,
        include_embeddings=True,
        show_progress=True,
    )
    
    return kg_index

def extract_entities_from_query(query: str, graph: nx.Graph) -> Set[str]:
    """Extract mentioned entities from query that exist in the graph."""
    entities = set()
    query_lower = query.lower()
    
    for node in graph.nodes():
        if str(node).lower() in query_lower:
            entities.add(node)
    
    return entities

def create_pyvis_graph(graph: nx.Graph, query: str = None, response_nodes: List[str] = None) -> Network:
    """Create an interactive Pyvis network from the NetworkX graph."""
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    
    # Add all nodes first
    for node in graph.nodes():
        net.add_node(str(node), label=str(node), title=str(node))
    
    # Add edges with relationships
    for source, target, data in graph.edges(data=True):
        relationship = data.get('relationship', 'related_to')
        net.add_edge(str(source), str(target), title=relationship)
    
    # Highlight query-related nodes if query is provided
    if query:
        query_entities = extract_entities_from_query(query, graph)
        for node_id in query_entities:
            try:
                net.get_node(str(node_id))['color'] = '#90EE90'  # Light green
            except:
                pass
    
    # Highlight response nodes if provided
    if response_nodes:
        for node_id in response_nodes:
            try:
                net.get_node(str(node_id))['color'] = '#ADD8E6'  # Light blue
            except:
                pass
    
    # Enable physics simulation for better layout
    net.toggle_physics(True)
    net.show_buttons(filter_=['physics'])
    
    return net

def main():
    st.set_page_config(page_title="GraphRAG Explorer", layout="wide")
    
    st.title("GraphRAG Knowledge Explorer")
    st.markdown("""
    This application allows you to explore and query a knowledge graph built from markdown files.
    Upload your markdown files, ask questions, and explore the relationships visually.
    """)
    
    # Initialize session state
    if 'kg_index' not in st.session_state:
        st.session_state.kg_index = None
    if 'graph' not in st.session_state:
        st.session_state.graph = None
    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = None
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = None
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Markdown Files",
        type=['md'],
        accept_multiple_files=True,
    )
    
    if uploaded_files:
        try:
            # Process files only if they've changed
            current_filenames = {f.name for f in uploaded_files}
            if (not st.session_state.kg_index or 
                'last_processed_files' not in st.session_state or 
                current_filenames != st.session_state.last_processed_files):
                
                with st.spinner("Processing markdown files..."):
                    # Clean up old temp directory if it exists
                    if st.session_state.temp_dir and Path(st.session_state.temp_dir).exists():
                        shutil.rmtree(st.session_state.temp_dir)
                    
                    # Process new files
                    temp_files, temp_dir = process_uploaded_files(uploaded_files)
                    st.session_state.temp_dir = temp_dir
                    
                    try:
                        # Create knowledge graph
                        st.info("Building knowledge graph... This may take a few moments.")
                        st.session_state.kg_index = load_and_process_markdown(temp_files)
                        st.session_state.graph = st.session_state.kg_index.get_networkx_graph()
                        st.session_state.query_engine = st.session_state.kg_index.as_query_engine(
                            response_mode="tree_summarize",
                            verbose=False,
                        )
                        
                        # Store processed filenames
                        st.session_state.last_processed_files = current_filenames
                        st.success("Knowledge graph built successfully!")
                    except Exception as e:
                        st.error(f"Error building knowledge graph: {str(e)}")
                        if "OPENAI_API_KEY" in str(e):
                            st.error("Please check your OpenAI API key in the .env file.")
                        raise
            
            # Show graph statistics
            if st.session_state.graph:
                st.subheader("Knowledge Graph Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Number of Nodes", st.session_state.graph.number_of_nodes())
                with col2:
                    st.metric("Number of Edges", st.session_state.graph.number_of_edges())
                
                # Query input
                query = st.text_input("Enter your query:")
                if query:
                    with st.spinner("Processing query..."):
                        response = st.session_state.query_engine.query(query)
                        
                        # Display response
                        st.subheader("Response")
                        st.write(str(response))
                        
                        # Get response nodes
                        response_nodes = []
                        if hasattr(response, 'source_nodes'):
                            response_nodes = [
                                node.node.node_id for node in response.source_nodes
                            ]
                        
                        # Create and display interactive graph
                        st.subheader("Knowledge Graph Visualization")
                        net = create_pyvis_graph(st.session_state.graph, query, response_nodes)
                        
                        # Save and display the graph
                        net.save_graph("/tmp/graph.html")
                        with open("/tmp/graph.html", 'r', encoding='utf-8') as f:
                            html = f.read()
                        st.components.v1.html(html, height=600)
                else:
                    # Display initial graph
                    st.subheader("Knowledge Graph Visualization")
                    net = create_pyvis_graph(st.session_state.graph)
                    net.save_graph("/tmp/graph.html")
                    with open("/tmp/graph.html", 'r', encoding='utf-8') as f:
                        html = f.read()
                    st.components.v1.html(html, height=600)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.error("If this persists, try refreshing the page and uploading the files again.")

if __name__ == "__main__":
    main()
