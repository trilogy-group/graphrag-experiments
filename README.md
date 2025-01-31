# GraphRAG Experiments

An experimental project for building and querying knowledge graphs from markdown files using LlamaIndex.

## Features

- Build knowledge graphs from markdown files
- Interactive visualization using Streamlit and Pyvis
- Query the knowledge graph using natural language
- Highlight relevant nodes and relationships in the visualization

## Setup

1. Create a Python environment (3.9 or later)
### Install uv if you haven't already
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
### Create and activate a new virtual environment
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
# .venv\Scripts\activate  # On Windows
```

2. Install dependencies using uv:
   ```bash
   uv pip install -e .
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_key_here
   ```

## Usage

### Web Interface

Run the web interface:
```bash
./run_webapp.sh
```

This will start a Streamlit app where you can:
1. Upload markdown files
2. View the knowledge graph visualization
3. Query the graph using natural language
4. See highlighted nodes and relationships based on your query

### CLI Interface

Run the CLI tool:
```bash
graphrag process path/to/markdown/files
```

## Development

The project uses:
- LlamaIndex for knowledge graph creation and querying
- Streamlit for the web interface
- Pyvis for graph visualization
- NetworkX for graph operations

## License

MIT License
