# GraphRAG Architecture

## Overview

GraphRAG combines knowledge graph construction with retrieval-augmented generation (RAG) to create an intelligent system for querying and exploring document relationships. The system processes markdown documents to build a knowledge graph, which can then be queried using natural language.

## System Components

### 1. Document Processing & Entity Extraction
- Documents are loaded using `SimpleDirectoryReader`
- `SimpleNodeParser` splits documents into manageable chunks
- LLM performs:
  - Entity identification within chunks
  - Relationship extraction between entities
  - Triplet generation (subject, predicate, object)

### 2. Knowledge Graph Construction
- `KnowledgeGraphIndex` creates a NetworkX graph from extracted triplets
- Nodes represent entities from the documents
- Edges represent relationships with predicates as relationship types
- Original context is preserved in node metadata
- Node embeddings assist in relationship identification during construction

### 3. Embedding Layer
- OpenAI's embedding model generates vector representations for nodes
- Embeddings are used during graph construction to:
  - Help identify potential relationships
  - Support entity disambiguation
  - Enhance relationship extraction

### 4. Query Processing
- Analyzes input questions for entity mentions
- Uses graph-based retrieval:
  - Identifies mentioned entities
  - Traverses graph to find connected nodes
  - Scores relevance based on graph structure

### 5. Response Generation
- Collects relevant nodes and relationships from graph traversal
- Formats subgraph information into LLM prompt
- Generates natural language response using:
  - Original question
  - Retrieved subgraph
  - Associated document context

## Architecture Diagram

```mermaid
graph TD
    subgraph Input
        MD[Markdown Files]
    end

    subgraph "Document Processing"
        DR[Document Reader] --> NP[Node Parser]
        NP --> EE[Entity Extraction<br/>LLM]
        EE --> TR[Triplet Generation]
    end

    subgraph "Knowledge Graph Construction"
        KG[Knowledge Graph Index]
        EM[Embedding Model<br/>OpenAI]
        TR --> KG
        EM --> KG
    end

    subgraph "Query Processing"
        QA[Query Analysis] --> ER[Entity Recognition]
        ER --> KG
        KG --> GT[Graph Traversal]
    end

    subgraph "Response Generation"
        GT --> SG[Subgraph Collection]
        KG --> SG
        SG --> PF[Prompt Formatting]
        PF --> LLM[LLM Response]
    end

    MD --> DR

    style KG fill:#f9f,stroke:#333,stroke-width:2px
    style EM fill:#bbf,stroke:#333,stroke-width:2px
    style EE fill:#bfb,stroke:#333,stroke-width:2px
    style LLM fill:#bfb,stroke:#333,stroke-width:2px
```

## Data Flow

1. **Input Stage**
   - Markdown files are loaded into the system
   - Documents are split into manageable chunks

2. **Processing Stage**
   - Entities and relationships are extracted
   - Node embeddings are generated
   - Knowledge graph is constructed using both explicit relationships and embedding-assisted relationship identification

3. **Query Stage**
   - User submits natural language query
   - Query is analyzed for entities
   - Graph is traversed to find relevant subgraph

4. **Response Stage**
   - Subgraph is formatted into prompt
   - LLM generates natural language response
   - Results are displayed with visualization

## Performance Considerations

- Embedding generation is computationally intensive but only needed during graph construction
- Initial graph construction requires multiple API calls for:
  - Entity extraction
  - Relationship identification
  - Embedding generation
- Query processing uses efficient graph traversal algorithms
- Response generation is optimized for context relevance

## Future Enhancements

The current implementation of the system incorporates a basic temporal knowledge graph (TKG) by associating temporal metadata, such as document ingestion dates, with nodes to track when information was added to the graph. While this provides foundational temporal context for data provenance and versioning, it does not yet model time as an inherent part of the domain knowledge itself.

To evolve into a true TKG, future enhancements would involve embedding temporal semantics directly into the graph's structure, such as timestamping edges with validity intervals (e.g., "CEO of Company X from 2015–2023"), enabling dynamic node attributes that reflect changes over time (e.g., fluctuating stock prices), and representing time-bound events as first-class entities (e.g., "Product Launch on 2024-05-01"). These improvements would allow the system to support complex temporal reasoning, such as querying historical states of relationships, analyzing trends, or inferring causality across time, transforming the graph into a robust tool for understanding evolving real-world dynamics.