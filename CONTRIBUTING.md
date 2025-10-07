# Contributing to Exoplanet Detection

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Development Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- Git

### Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/SpaceApps.git
cd SpaceApps
```

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
```

### ML Setup
```bash
cd ml
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
```

### Frontend Setup
```bash
cd frontend
npm install
```

## Development Workflow

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Adding tests

### 2. Make Changes

Follow our coding standards:
- **Python**: PEP 8, max line length 100
- **TypeScript**: ESLint configuration
- Write tests for new features
- Update documentation as needed

### 3. Run Tests Locally

#### Backend Tests
```bash
cd backend
pytest tests/ -v --cov=app
```

#### ML Tests
```bash
cd ml
pytest tests/ -v --cov=src
```

#### Frontend Tests
```bash
cd frontend
npm run lint
npm run build
```

### 4. Format Code

#### Python (Backend & ML)
```bash
# Auto-format
black backend/ ml/

# Check imports
isort backend/ ml/

# Lint
flake8 backend/ ml/src/
```

#### TypeScript (Frontend)
```bash
cd frontend
npm run lint
```

### 5. Commit Changes

Use conventional commit messages:
```
feat(scope): add new feature
fix(scope): fix bug
docs(scope): update documentation
test(scope): add tests
refactor(scope): refactor code
```

Examples:
```bash
git commit -m "feat(ml): add SHAP explanations for predictions"
git commit -m "fix(backend): handle missing CSV columns"
git commit -m "docs(readme): update installation instructions"
```

### 6. Push and Create PR
```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Style Guidelines

### Python

**Line Length**: 100 characters

**Imports**:
```python
# Standard library
import os
import sys

# Third-party
import numpy as np
import pandas as pd

# Local
from app.models import Model
```

**Docstrings**:
```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description of function.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something is wrong
    """
    pass
```

**Type Hints**: Always use type hints
```python
def process_data(data: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    pass
```

### TypeScript

**Component Structure**:
```typescript
import { useState } from 'react';
import styles from './Component.module.css';

interface ComponentProps {
  prop1: string;
  prop2?: number;
}

export default function Component({ prop1, prop2 = 0 }: ComponentProps) {
  const [state, setState] = useState<string>('');
  
  return (
    <div className={styles.container}>
      {/* content */}
    </div>
  );
}
```

## Testing Guidelines

### Writing Tests

#### Python (pytest)
```python
import pytest
import numpy as np

@pytest.fixture
def sample_data():
    return np.random.randn(100, 10)

class TestFeature:
    def test_basic_functionality(self, sample_data):
        """Test basic functionality."""
        result = process(sample_data)
        assert result is not None
    
    def test_edge_case(self):
        """Test edge case."""
        with pytest.raises(ValueError):
            process(None)
```

#### Coverage Requirements
- New features: >80% coverage
- Bug fixes: Add test reproducing bug
- Critical paths: >90% coverage

### Running Tests

**All tests**:
```bash
# Backend
cd backend && pytest tests/ -v

# ML
cd ml && pytest tests/ -v

# Frontend
cd frontend && npm test
```

**With coverage**:
```bash
pytest tests/ -v --cov=app --cov-report=html
```

**Specific test**:
```bash
pytest tests/test_specific.py::TestClass::test_method -v
```

## Pull Request Process

### Before Submitting

- [ ] All tests pass locally
- [ ] Code is formatted (black, isort)
- [ ] No linting errors (flake8)
- [ ] Tests added for new features
- [ ] Documentation updated
- [ ] Commit messages follow convention

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How has this been tested?

## Checklist
- [ ] Tests pass
- [ ] Code formatted
- [ ] Documentation updated
- [ ] No merge conflicts
```

### Review Process

1. Automated CI checks must pass
2. At least one reviewer approval required
3. Address all review comments
4. Keep PR scope focused

## CI/CD Pipeline

Our CI runs on every push and PR:

1. **Backend Tests** - Linting, formatting, pytest
2. **ML Tests** - Linting, formatting, pytest
3. **Frontend Build** - ESLint, build
4. **Integration Tests** - Full stack health check

See `.github/workflows/ci.yml` for details.

## Common Issues

### Import Errors
```bash
# Ensure you're in the right directory
cd backend  # or cd ml
# Activate virtual environment
source venv/bin/activate
# Reinstall dependencies
pip install -r requirements-dev.txt
```

### Flake8 Errors
```bash
# Auto-fix most issues
black .
isort .

# Check remaining issues
flake8 .
```

### Test Failures
```bash
# Run with verbose output
pytest tests/ -v -s

# Run single test
pytest tests/test_file.py::test_function -v

# Debug with pdb
pytest tests/ --pdb
```

### Frontend Build Errors
```bash
# Clear cache
rm -rf node_modules .next
npm install
npm run build
```

## Documentation

### Update Documentation When:
- Adding new features
- Changing APIs
- Modifying configuration
- Adding dependencies

### Documentation Files
- `README.md` - Project overview
- `CONTRIBUTING.md` - This file
- `backend/README.md` - Backend documentation
- `ml/DATA_PROCESSING_GUIDE.md` - ML pipeline docs
- Code docstrings and comments

## Getting Help

- **Issues**: Search existing issues or create new one
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check README and guides
- **Tests**: Look at existing tests for examples

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

## Recognition

Contributors will be recognized in:
- GitHub contributors page
- Release notes
- README acknowledgments

Thank you for contributing! 🚀

