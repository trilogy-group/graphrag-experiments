import os
from pathlib import Path
from typing import List, Optional, Set

import typer
import networkx as nx
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich import print as rprint

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    KnowledgeGraphIndex,
    QueryBundle,
    get_response_synthesizer,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Initialize rich console and CLI app
console = Console()
app = typer.Typer()

# Load environment variables
load_dotenv()

# Configure LlamaIndex settings with more explicit prompting
Settings.llm = OpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    system_prompt=(
        "You are a helpful assistant that answers questions based on the knowledge graph. "
        "Always try to use the relationships and information from the graph in your answers. "
        "If you find relevant information, explain it clearly. "
        "If you don't find relevant information, say so explicitly."
    )
)
Settings.embed_model = OpenAIEmbedding()

# Disable progress bars from LlamaIndex
Settings.show_progress = False

def show_debug_info(documents, nodes, kg_index):
    """Display debug information about the loaded content and index."""
    console.print("\n[yellow]Debug Information:[/yellow]")
    
    # Document info
    doc_table = Table(title="Documents Loaded")
    doc_table.add_column("Document")
    doc_table.add_column("Content Preview")
    doc_table.add_column("Length")
    
    for doc in documents:
        preview = doc.text[:100] + "..." if len(doc.text) > 100 else doc.text
        doc_table.add_row(
            str(doc.metadata.get('file_name', 'Unknown')),
            preview,
            str(len(doc.text))
        )
    console.print(doc_table)
    
    # Node info
    node_table = Table(title="Nodes Created")
    node_table.add_column("Node ID")
    node_table.add_column("Content Preview")
    
    for node in nodes:
        preview = node.text[:100] + "..." if len(node.text) > 100 else node.text
        node_table.add_row(str(node.node_id), preview)
    console.print(node_table)
    
    # Graph info
    if hasattr(kg_index, 'get_networkx_graph'):
        graph = kg_index.get_networkx_graph()
        console.print(f"\nKnowledge Graph Statistics:")
        console.print(f"- Number of nodes: {graph.number_of_nodes()}")
        console.print(f"- Number of edges: {graph.number_of_edges()}")
        
        if graph.number_of_edges() > 0:
            console.print("\nSample Relationships:")
            edges_table = Table(title="Sample Graph Relationships")
            edges_table.add_column("Source")
            edges_table.add_column("Relationship")
            edges_table.add_column("Target")
            
            for i, (source, target, data) in enumerate(graph.edges(data=True)):
                if i < 5:  # Show first 5 relationships
                    edges_table.add_row(
                        str(source),
                        data.get('relationship', 'related_to'),
                        str(target)
                    )
                else:
                    break
            console.print(edges_table)

def load_and_process_markdown(input_files: List[Path]) -> KnowledgeGraphIndex:
    """Load and process markdown files into a knowledge graph."""
    console.print("[blue]Loading and processing files...[/blue]")
    
    # Load the markdown files
    documents = SimpleDirectoryReader(
        input_files=[str(f) for f in input_files]
    ).load_data()
    
    # Parse the documents into nodes
    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(documents)
    
    # Create storage context and knowledge graph
    storage_context = StorageContext.from_defaults()
    
    # Create the knowledge graph index with more aggressive relationship extraction
    kg_index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=20,
        include_embeddings=True,
        show_progress=False,  # Disable progress bars
    )
    
    # Show debug information
    show_debug_info(documents, nodes, kg_index)
    
    return kg_index

def get_example_queries(kg_index) -> List[str]:
    """Generate example queries based on the graph structure."""
    queries = []
    
    if hasattr(kg_index, 'get_networkx_graph'):
        graph = kg_index.get_networkx_graph()
        
        # Get unique entities and relationships
        entities = set()
        relationships = set()
        
        for source, target, data in graph.edges(data=True):
            entities.add(source)
            entities.add(target)
            rel = data.get('relationship', 'related_to')
            relationships.add(rel)
        
        # Generate queries based on graph structure
        for entity in list(entities)[:3]:  # Use first 3 entities
            queries.append(f"What is {entity}?")
            queries.append(f"Tell me about {entity}'s relationships.")
            
        # Add relationship-based queries
        for rel in list(relationships)[:2]:  # Use first 2 relationships
            queries.append(f"What entities are {rel}?")
            
        # Add general queries
        queries.extend([
            "What are the main topics discussed in the document?",
            "What are the key relationships between entities?",
        ])
    
    return queries[:5]  # Return at most 5 queries

def extract_entities_from_query(query: str, graph: nx.Graph) -> Set[str]:
    """Extract mentioned entities from query that exist in the graph."""
    entities = set()
    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower()
    
    # Check each node in the graph
    for node in graph.nodes():
        # Convert node to lowercase for case-insensitive matching
        if str(node).lower() in query_lower:
            entities.add(node)
    
    return entities

def visualize_relevant_subgraph(graph: nx.Graph, query: str, response_nodes: List[str] = None) -> None:
    """Visualize a subgraph relevant to the query using NetworkX."""
    try:
        # Check if we're in a terminal that supports display
        import matplotlib
        if not os.getenv('DISPLAY') and matplotlib.get_backend() == 'agg':
            matplotlib.use('TkAgg')  # Try to use TkAgg backend
        import matplotlib.pyplot as plt
        
        # Get entities mentioned in the query
        query_entities = extract_entities_from_query(query, graph)
        
        # Create a set of nodes to include in visualization
        nodes_to_include = set()
        
        # Add query entities and their neighbors (ego graph)
        for entity in query_entities:
            nodes_to_include.add(entity)
            nodes_to_include.update(graph.neighbors(entity))
        
        # Add response nodes if provided
        if response_nodes:
            nodes_to_include.update(response_nodes)
            # Add immediate neighbors of response nodes
            for node in response_nodes:
                if node in graph:
                    nodes_to_include.update(graph.neighbors(node))
        
        # Create subgraph
        subgraph = graph.subgraph(nodes_to_include)
        
        if len(subgraph.nodes()) == 0:
            console.print("[yellow]No relevant nodes found to visualize[/yellow]")
            return
        
        # Clear any existing plots
        plt.clf()
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(subgraph, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(subgraph, pos, 
                             node_color='lightblue',
                             node_size=2000,
                             alpha=0.7)
        
        # Highlight query entities
        if query_entities:
            nx.draw_networkx_nodes(subgraph, pos,
                                 nodelist=list(query_entities),
                                 node_color='lightgreen',
                                 node_size=2000,
                                 alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(subgraph, pos, alpha=0.5)
        
        # Draw labels
        nx.draw_networkx_labels(subgraph, pos)
        
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(subgraph, 'relationship')
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels)
        
        plt.title(f"Knowledge Graph Subgraph for Query: {query}")
        plt.axis('off')
        
        # Try to show the plot
        try:
            plt.show()
        except Exception as e:
            console.print("[yellow]Could not display plot in current environment.[/yellow]")
            # Save to file instead
            plt.savefig('knowledge_graph_visualization.png')
            console.print("[green]Visualization saved to 'knowledge_graph_visualization.png'[/green]")
        
    except ImportError:
        console.print("[yellow]matplotlib is required for visualization. Install it with:[/yellow]")
        console.print("pip install matplotlib")
    except Exception as e:
        console.print(f"[red]Error visualizing graph:[/red] {str(e)}")

@app.command()
def main(
    files: List[Path] = typer.Argument(
        ...,
        help="Markdown files to process",
        exists=True,
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive", "-i",
        help="Enable interactive query mode"
    ),
    debug: bool = typer.Option(
        False,
        "--debug", "-d",
        help="Show debug information during querying"
    ),
    visualize: bool = typer.Option(
        False,
        "--visualize", "-v",
        help="Visualize relevant subgraphs for queries"
    )
):
    """
    Process markdown files into a knowledge graph and query it.
    """
    # Print welcome message
    console.print(Panel.fit(
        "[bold blue]GraphRAG Knowledge Graph Demo[/bold blue]\n"
        "Process markdown files and query the resulting knowledge graph"
    ))
    
    # Process markdown files
    kg_index = load_and_process_markdown(files)
    graph = kg_index.get_networkx_graph()
    
    console.print("\n[green]✓[/green] Knowledge graph created successfully!")
    
    # Create query engine directly from the index
    query_engine = kg_index.as_query_engine(
        response_mode="tree_summarize",
        verbose=False,  # Reduce output verbosity
    )
    
    if interactive:
        # Interactive query mode
        console.print("\n[yellow]Enter your queries (press Ctrl+C to exit):[/yellow]")
        console.print("\nSuggested query formats:")
        console.print("- What is [entity]?")
        console.print("- Tell me about [entity]'s relationships")
        console.print("- What entities are [relationship]?")
        console.print("- Who is related to [entity]?")
        
        try:
            while True:
                query = Prompt.ask("\n[bold blue]Query")
                with console.status("[bold green]Processing query..."):
                    try:
                        response = query_engine.query(query)
                        if debug:
                            console.print("[yellow]Debug: Retrieved nodes:[/yellow]")
                            for node in response.source_nodes:
                                console.print(f"- {node.node.text[:200]}...")
                        
                        console.print(Panel(str(response), title="Response"))
                        
                        if visualize:
                            # Get node IDs from response
                            response_nodes = [
                                node.node.node_id for node in response.source_nodes
                            ] if hasattr(response, 'source_nodes') else None
                            
                            # Visualize the subgraph
                            visualize_relevant_subgraph(graph, query, response_nodes)
                            
                    except Exception as e:
                        console.print(f"[red]Error processing query:[/red] {str(e)}")
        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting interactive mode[/yellow]")
    else:
        # Generate example queries based on the graph
        example_queries = get_example_queries(kg_index)
        
        console.print("\n[yellow]Running example queries based on the graph structure:[/yellow]")
        for query in example_queries:
            console.print(f"\n[bold blue]Query:[/bold blue] {query}")
            with console.status("[bold green]Processing query..."):
                try:
                    response = query_engine.query(query)
                    if debug:
                        console.print("[yellow]Debug: Retrieved nodes:[/yellow]")
                        for node in response.source_nodes:
                            console.print(f"- {node.node.text[:200]}...")
                    
                    console.print(Panel(str(response), title="Response"))
                    
                    if visualize:
                        # Get node IDs from response
                        response_nodes = [
                            node.node.node_id for node in response.source_nodes
                        ] if hasattr(response, 'source_nodes') else None
                        
                        # Visualize the subgraph
                        visualize_relevant_subgraph(graph, query, response_nodes)
                        
                except Exception as e:
                    console.print(f"[red]Error processing query:[/red] {str(e)}")

if __name__ == "__main__":
    app()
