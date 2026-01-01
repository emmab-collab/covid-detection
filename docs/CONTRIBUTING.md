# Contributing Guide

Thank you for your interest in contributing to the COVID-19 Detection project!

## Table of Contents
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/covid-detection.git
   cd covid-detection
   ```

3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Development Dependencies

```bash
pip install -r requirements.txt
pip install -e .  # Install package in editable mode
```

### 3. Install Testing Tools

```bash
pip install pytest pytest-cov black flake8 mypy
```

## Project Structure

```
covid-detection/
├── src/              # Source code
│   ├── data/         # Data processing
│   ├── features/     # Feature engineering
│   ├── models/       # Model training & evaluation
│   └── visualization/ # Plotting functions
├── scripts/          # Executable scripts
├── notebooks/        # Jupyter notebooks
├── tests/            # Unit tests
└── docs/             # Documentation
```

### Adding New Features

#### 1. Data Processing Functions
Add to `src/data/preprocessing.py`:

```python
def new_preprocessing_function(df: pd.DataFrame) -> pd.DataFrame:
    """
    Brief description.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    pd.DataFrame
        Processed dataframe
    """
    # Implementation
    return df
```

#### 2. Feature Engineering
Add to `src/features/engineering.py`:

```python
def new_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new feature.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    pd.DataFrame
        Dataframe with new feature
    """
    # Implementation
    return df
```

#### 3. Models
Add to `src/models/train.py`:

```python
def build_new_model():
    """Build and return new model."""
    model = SomeClassifier()
    return model
```

## Coding Standards

### Python Style Guide

We follow **PEP 8** style guidelines:

- **Indentation:** 4 spaces (no tabs)
- **Line length:** Maximum 88 characters (Black formatter)
- **Naming conventions:**
  - Functions: `lowercase_with_underscores`
  - Classes: `CapitalizedWords`
  - Constants: `UPPERCASE_WITH_UNDERSCORES`

### Format Code with Black

```bash
black src/ scripts/ tests/
```

### Check Code Quality

```bash
flake8 src/ scripts/ tests/
```

### Type Hints

Use type hints for function parameters and returns:

```python
def process_data(df: pd.DataFrame, n_features: int = 10) -> Tuple[pd.DataFrame, List[str]]:
    """Process data and return features."""
    # Implementation
    return processed_df, feature_names
```

## Documentation

### Docstrings

Use **NumPy-style docstrings**:

```python
def example_function(param1, param2):
    """
    Short description of function.

    Longer description if needed. Can span multiple lines
    and provide more context about what the function does.

    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type
        Description of param2

    Returns
    -------
    type
        Description of return value

    Examples
    --------
    >>> example_function(1, 2)
    3

    Notes
    -----
    Additional notes about implementation or usage.
    """
    return param1 + param2
```

### Update Documentation

When adding new features:
1. Add docstrings to functions
2. Update `docs/API.md` with new API reference
3. Update `README.md` if adding user-facing features

## Testing

### Writing Tests

Create test files in `tests/` directory:

```python
# tests/test_new_feature.py
import pytest
from src.features.engineering import new_feature

class TestNewFeature:
    """Test new feature function."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        result = new_feature(df)
        assert 'new_col' in result.columns

    def test_edge_case(self):
        """Test edge case."""
        df = pd.DataFrame()
        result = new_feature(df)
        assert len(result) == 0
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_preprocessing.py

# Run with coverage
pytest --cov=src tests/

# Run with verbose output
pytest -v tests/
```

### Test Coverage

Aim for **>80% code coverage**:

```bash
pytest --cov=src --cov-report=html tests/
# Open htmlcov/index.html to view coverage report
```

## Submitting Changes

### 1. Commit Your Changes

Use clear, descriptive commit messages:

```bash
git add .
git commit -m "Add feature: new preprocessing function for X

- Implement function to handle Y
- Add tests for edge cases
- Update documentation"
```

**Commit message format:**
- First line: Brief summary (50 chars max)
- Blank line
- Detailed description (if needed)

### 2. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 3. Create Pull Request

1. Go to the original repository on GitHub
2. Click "New Pull Request"
3. Select your branch
4. Fill in the PR template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] All tests pass
- [ ] Added new tests for changes
- [ ] Code coverage maintained/improved

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented if yes)
```

### 4. Code Review

- Respond to reviewer comments
- Make requested changes
- Push updates to the same branch

## Reporting Bugs

### Bug Report Template

```markdown
**Description**
Clear description of the bug

**To Reproduce**
1. Step 1
2. Step 2
3. See error

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Windows 10]
- Python version: [e.g., 3.9]
- Package version: [e.g., 0.1.0]

**Additional Context**
Any other relevant information
```

## Feature Requests

### Feature Request Template

```markdown
**Problem**
What problem does this solve?

**Proposed Solution**
How should it work?

**Alternatives Considered**
Other approaches you've thought about

**Additional Context**
Any other relevant information
```

## Questions?

- Check existing [documentation](../README.md)
- Review [API documentation](API.md)
- Open an issue for discussion

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Accept responsibility for mistakes

Thank you for contributing! 🎉
