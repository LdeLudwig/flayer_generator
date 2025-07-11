# Flyer Generator

A multimodal AI-powered flyer generator that creates professional supermarket flyers using text and image inputs. The system uses RAG (Retrieval-Augmented Generation) with CLIP embeddings for similarity search and DALL-E for image generation.

## Features

- **Multimodal Input**: Accept both text descriptions and reference images
- **RAG-based Generation**: Uses similarity search to find relevant examples
- **AI Image Generation**: Powered by DALL-E for high-quality flyer creation
- **Vector Database**: Qdrant for efficient similarity search
- **Web Interface**: Streamlit frontend for easy interaction
- **REST API**: FastAPI backend for programmatic access

## Tech Stack

- **Backend**: FastAPI, Python 3.10+
- **Frontend**: Streamlit
- **AI/ML**: OpenAI (DALL-E, GPT-4), Transformers (CLIP), LangChain
- **Database**: Qdrant (vector database)
- **Dependency Management**: uv
- **Containerization**: Docker, Docker Compose

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Docker and Docker Compose (optional)
- OpenAI API key

### Local Development

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd flyer_generator
   ```

2. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

5. **Run the application**:
   ```bash
   # Backend (API)
   uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
   
   # Frontend (Streamlit) - in another terminal
   uv run streamlit run backend/streamlit_app.py --server.port 8501
   ```

### Docker Development

1. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

2. **Access the application**:
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## Migration from Poetry to uv

This project has been migrated from Poetry to uv for faster dependency management. See [UV_MIGRATION.md](UV_MIGRATION.md) for details.

### Quick Migration

Run the migration script:
```bash
./migrate_to_uv.sh
```

## Project Structure

```
flyer_generator/
├── backend/
│   ├── api/              # FastAPI routes
│   ├── db/               # Database connections
│   ├── models/           # Pydantic models
│   ├── services/         # Business logic
│   ├── scripts/          # Utility scripts
│   ├── main.py           # FastAPI app
│   └── streamlit_app.py  # Streamlit frontend
├── tests/                # Test files
├── flyers_data/          # Training data
├── docker-compose.yml    # Docker configuration
├── Dockerfile           # Docker image definition
├── pyproject.toml       # Project dependencies
└── README.md           # This file
```

## API Endpoints

- `GET /` - Health check
- `POST /train_embedding` - Train embeddings from flyer data
- `POST /get_rag_response` - Get similar flyers for a query
- `POST /api/generate_dalle` - Generate flyer using DALL-E
- `GET /get_collection_info` - Get vector database collection info

## Development

### Adding Dependencies

```bash
# Production dependency
uv add package-name

# Development dependency
uv add --dev package-name
```

### Running Tests

```bash
# Test uv setup
uv run python test_uv_setup.py

# Test embedding service
uv run python test_embedding_fix.py
```

### Code Quality

```bash
# Format code
uv run black backend/

# Sort imports
uv run isort backend/

# Lint code
uv run flake8 backend/
```

## Troubleshooting

### Common Issues

1. **Missing models**: Download CLIP models when you have internet:
   ```bash
   uv run python backend/scripts/download_models.py
   ```

2. **Permission errors with .venv**: Remove with sudo:
   ```bash
   sudo rm -rf .venv
   ```

3. **Docker build issues**: Clear Docker cache:
   ```bash
   docker system prune -a
   ```

### Environment Variables

- `OPENAI_API_KEY`: Required for DALL-E and GPT-4
- `TRANSFORMERS_OFFLINE`: Set to "1" for offline mode
- `CLIP_MODEL_NAME`: Custom CLIP model (default: openai/clip-vit-base-patch32)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

[Add your license here]

## Support

For issues and questions:
- Check [UV_MIGRATION.md](UV_MIGRATION.md) for uv-related issues
- Check [EMBEDDING_FIXES.md](EMBEDDING_FIXES.md) for model-related issues
- Open an issue on GitHub
