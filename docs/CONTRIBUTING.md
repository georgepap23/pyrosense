# Contributing to PyroSense

Thank you for your interest in contributing!

## Development Setup

1. Clone and install:
   ```bash
   git clone https://github.com/georgepap23/pyrosense.git
   cd pyrosense
   pip install -e ".[dev]"
   ```

2. Install pre-commit hooks (optional):
   ```bash
   pre-commit install
   ```

## Code Style

We use Ruff for linting and formatting:

```bash
ruff check src/          # Check for issues
ruff check src/ --fix    # Fix auto-fixable issues
ruff format src/         # Format code
```

### Guidelines

- Use type hints for function signatures
- Write docstrings for public functions and classes
- Keep lines under 100 characters

## Pull Requests

1. Fork and create a branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make changes following code style guidelines

3. Run linting:
   ```bash
   ruff check src/
   ```

4. Commit with a clear message:
   ```bash
   git commit -m "feat: add your feature description"
   ```

5. Push and create a Pull Request

## Commit Messages

Use conventional commit format:

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation changes
- `refactor:` code changes that neither fix bugs nor add features

## Questions?

Open an issue for bugs or feature requests.
