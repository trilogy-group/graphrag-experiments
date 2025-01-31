# Temporal Knowledge Graph (TKG)

The current implementation of the system incorporates a basic temporal knowledge graph (TKG) by associating temporal metadata, such as document ingestion dates, with nodes to track when information was added to the graph. While this provides foundational temporal context for data provenance and versioning, it does not yet model time as an inherent part of the domain knowledge itself.

To evolve into a true TKG, future enhancements would involve embedding temporal semantics directly into the graph's structure, such as timestamping edges with validity intervals (e.g., "CEO of Company X from 2015–2023"), enabling dynamic node attributes that reflect changes over time (e.g., fluctuating stock prices), and representing time-bound events as first-class entities (e.g., "Product Launch on 2024-05-01"). These improvements would allow the system to support complex temporal reasoning, such as querying historical states of relationships, analyzing trends, or inferring causality across time, transforming the graph into a robust tool for understanding evolving real-world dynamics.

## Temporal Node Weighting in GraphRAG

### Implementation

The temporal weighting of nodes is implemented in the [create_pyvis_graph](cci:1://file:///Users/magos/dev/trilogy/GraphRAG/graphrag-experiments/src/graphrag/web_app.py:132:0-185:14) function in [web_app.py](cci:7://file:///Users/magos/dev/trilogy/GraphRAG/graphrag-experiments/src/graphrag/web_app.py:0:0-0:0). The weighting affects both the visual opacity of nodes based on their age and query relevance scoring.

```python
# Calculate opacity based on date recency
if doc_date:
    try:
        doc_date = datetime.fromisoformat(doc_date).date()
        today = date.today()
        days_old = (today - doc_date).days
        # Scale opacity from 0.3 to 1.0 based on age
        # Nodes older than 365 days get minimum opacity of 0.3
        opacity = max(0.3, 1.0 - min(days_old, 365) / 365)
    except (ValueError, TypeError):
        opacity = 1.0  # Default to full opacity if date parsing fails
```

### Key Components

1. **Date Storage**: Document dates are stored in the `node_dates` dictionary, mapping node text to ISO format date strings:
   ```python
   node_dates[node_str] = doc_date  # doc_date is in ISO format: YYYY-MM-DD
   ```

2. **Age Calculation**: Node age is calculated as days between the document date and current date:
   - Recent documents (0 days old) → opacity = 1.0
   - Year-old documents (365 days) → opacity = 0.3
   - Linear interpolation for documents between 0-365 days

3. **Visual Representation**: The opacity is applied to both default and highlighted nodes:
   ```python
   color = f"rgba({r},{g},{b},{opacity})"  # Applied to both blue (default) and red (highlighted) nodes
   ```

### Current Limitations

1. The temporal weighting affects both visualization and query relevance:
   - **Query Relevance**: Document age influences node selection through the `TemporalResponseSynthesizer`, which scales relevance scores based on document age
   - **Visual Representation**: Node opacity reflects the same temporal weighting (newer=more opaque)
   - **Scoring Formula**: Both query relevance and visualization use the same temporal scaling:
     ```python
     temporal_score = max(0.3, 1.0 - min(days_old, 365) / 365)
     ```

2. Dates are stored at the node level rather than the edge level, limiting the ability to represent temporal relationships between nodes.

### Implementation Details

#### Query Relevance Scoring

The temporal weighting of query results is implemented in the `TemporalResponseSynthesizer` class, which extends LlamaIndex's `TreeSummarize`:

```python
class TemporalResponseSynthesizer(TreeSummarize):
    def _get_nodes_for_response(self, query_bundle, nodes):
        nodes_with_scores = []
        for node in nodes:
            # Combine base relevance with temporal score
            base_score = node.score if hasattr(node, 'score') else 0.0
            temporal_score = calculate_temporal_score(doc_date)
            final_score = base_score * temporal_score
            node.score = final_score
            nodes_with_scores.append(node)
        return sorted(nodes_with_scores, key=lambda x: x.score, reverse=True)
```

This ensures that:
1. Recent documents (0 days old) maintain their full relevance score
2. Year-old documents (365+ days) have their relevance reduced by 70%
3. Documents between 0-365 days have linearly interpolated score reduction

#### Visual Representation

The same temporal weighting formula is applied to node opacity in the visualization:
```python
opacity = max(0.3, 1.0 - min(days_old, 365) / 365)
```

This creates a consistent representation where both visual prominence and query relevance are affected by document age in the same way.

### Location in Codebase

The implementation spans several key locations:
- [web_app.py](cci:7://file:///Users/magos/dev/trilogy/GraphRAG/graphrag-experiments/src/graphrag/web_app.py:0:0-0:0): Main implementation
  - `TemporalResponseSynthesizer`: Custom response synthesizer that incorporates temporal weights
  - `calculate_temporal_score()`: Shared temporal scoring function
  - `load_and_process_markdown()`: Date metadata attachment
  - `create_pyvis_graph()`: Visualization with temporal opacity
- [architecture.md](cci:7://file:///Users/magos/dev/trilogy/GraphRAG/graphrag-experiments/architecture.md:0:0-0:0): Documentation of TKG capabilities and roadmap