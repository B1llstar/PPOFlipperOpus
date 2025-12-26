# Using UV with PPO Flipper Opus

This project now uses [uv](https://github.com/astral-sh/uv) - a fast Python package installer and resolver.

## Installation

### Install UV

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Basic Commands

### Install Dependencies

```bash
# Install all project dependencies
uv sync

# Install with optional dependencies
uv sync --extra analysis --extra export

# Install dev dependencies
uv sync --group dev
```

### Run Python Scripts

```bash
# Run a script with the project's virtual environment
uv run python data/data_collector.py --email your@email.com

# Or activate the virtual environment first
source .venv/bin/activate  # Unix
.venv\Scripts\activate     # Windows
python data/data_collector.py --email your@email.com
```

### Add New Dependencies

```bash
# Add a new dependency
uv add package-name

# Add a dev dependency
uv add --dev package-name

# Add an optional dependency to a specific group
uv add --optional analysis pandas
```

### Update Dependencies

```bash
# Update all dependencies
uv lock --upgrade

# Update specific package
uv lock --upgrade-package torch
```

## Project Structure

The project has two `pyproject.toml` files:

1. **Root `pyproject.toml`**: Main project dependencies (PPO training, data collection, etc.)
2. **`dashboard/backend/pyproject.toml`**: Dashboard backend dependencies (FastAPI, websockets)

To work with the dashboard backend:

```bash
cd dashboard/backend
uv sync
uv run python server.py
```

## Why UV?

- **10-100x faster** than pip for dependency resolution and installation
- **Reproducible builds** with automatic lock file generation
- **Compatible** with existing pip/requirements.txt workflows
- **Built in Rust** for performance and reliability
- **All-in-one** tool: manages Python versions, virtual environments, and packages

## Migrating from Pip

If you prefer pip, the original `requirements.txt` files are still available:

```bash
pip install -r requirements.txt
```

However, we recommend using uv for better performance and dependency management.

## Common Tasks

### Training a Model
```bash
uv run python training/train_ppo.py
```

### Collecting Data
```bash
uv run python data/data_collector.py --email your@email.com --daemon
```

### Running Real Trading
```bash
uv run python inference/run_real_trading_enhanced.py
```

### Running Tests
```bash
uv run pytest tests/
```

## Troubleshooting

### "command not found: uv"
Make sure uv is installed and in your PATH. Restart your terminal after installation.

### "No Python version found"
UV will use the Python version specified in `.python-version` (3.11). If you don't have it:
```bash
uv python install 3.11
```

### Virtual Environment Issues
UV automatically creates and manages a `.venv` directory. To recreate it:
```bash
rm -rf .venv  # or rmdir /s .venv on Windows
uv sync
```

## Learn More

- [UV Documentation](https://docs.astral.sh/uv/)
- [UV GitHub Repository](https://github.com/astral-sh/uv)
