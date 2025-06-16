# Arabic Dialects Q&A Generation

This project is set up with Poetry for dependency management.

## Setup

1. Make sure you have Poetry installed. If not, install it following the instructions at https://python-poetry.org/docs/#installation

2. Install dependencies:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

## Dependencies

The project uses Python 3.12 and includes various dependencies for natural language processing, machine learning, and Arabic text processing. Key dependencies include:

- LangChain and related packages for LLM integration
- Google Cloud AI Platform
- FastText for text processing
- PyArabic for Arabic text handling
- Various utility packages (tqdm, numpy, etc.)

For a complete list of dependencies, see `pyproject.toml`.