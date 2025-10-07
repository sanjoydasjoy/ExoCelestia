# GitHub Actions CI/CD Workflows

## Overview

This directory contains GitHub Actions workflows for continuous integration and deployment.

## Workflows

### `ci.yml` - Continuous Integration

**Triggers:**
- Push to `main` branch
- Pull requests to `main` branch

**Jobs:**

#### 1. Backend Tests
- Sets up Python 3.11
- Caches pip dependencies
- Installs backend requirements
- Runs flake8 linting
- Checks code formatting with black
- Runs pytest with coverage
- Uploads coverage to Codecov

**Commands:**
```bash
flake8 backend --count --select=E9,F63,F7,F82 --show-source --statistics
black --check backend --line-length 100
pytest tests/ -v --cov=app --cov-report=xml
```

#### 2. ML Tests
- Sets up Python 3.11
- Caches pip dependencies
- Installs ML requirements
- Runs flake8 linting on `ml/src`
- Checks code formatting with black
- Runs pytest with coverage
- Uploads coverage to Codecov

**Commands:**
```bash
flake8 ml/src --count --select=E9,F63,F7,F82 --show-source --statistics
black --check ml/src ml/tests --line-length 100
pytest tests/ -v --cov=src --cov-report=xml
```

#### 3. Frontend Build
- Sets up Node.js 18
- Caches npm dependencies
- Runs `npm ci` for clean install
- Runs ESLint via `npm run lint`
- Builds production bundle via `npm run build`
- Uploads build artifacts

**Commands:**
```bash
npm ci
npm run lint
npm run build
```

#### 4. Integration Tests
- Runs after all other jobs complete
- Sets up both Python and Node.js
- Starts backend server
- Tests health endpoint
- Ensures full stack integration

## Caching Strategy

### Python Dependencies
```yaml
cache: 'pip'
cache-dependency-path: |
  backend/requirements.txt
  ml/requirements.txt
```

### Node.js Dependencies
```yaml
cache: 'npm'
cache-dependency-path: frontend/package-lock.json
```

## Configuration Files

### `.flake8`
Configures flake8 linting rules:
- Max line length: 100
- Max complexity: 10
- Ignores common false positives

### `pyproject.toml`
Configures black formatting and pytest:
- Black line length: 100
- Python target: 3.11
- Coverage settings
- Test markers

## Running Locally

### Backend Tests
```bash
cd backend
pip install -r requirements.txt
pip install flake8 black pytest pytest-cov

# Lint
flake8 .
black --check .

# Test
pytest tests/ -v --cov=app
```

### ML Tests
```bash
cd ml
pip install -r requirements.txt
pip install flake8 black pytest pytest-cov

# Lint
flake8 src/
black --check src/ tests/

# Test
pytest tests/ -v --cov=src
```

### Frontend Build
```bash
cd frontend
npm ci

# Lint
npm run lint

# Build
npm run build
```

## Status Badges

Add to your README.md:

```markdown
![CI](https://github.com/YOUR_USERNAME/YOUR_REPO/workflows/CI/badge.svg)
```

## Troubleshooting

### Flake8 Errors
- Check `.flake8` for ignored rules
- Run locally: `flake8 backend/` or `flake8 ml/src/`
- Fix with: `black backend/` or `black ml/`

### Black Formatting
- Auto-format: `black backend/` or `black ml/`
- Check without modifying: `black --check backend/`

### Test Failures
- Run locally first: `pytest tests/ -v`
- Check test logs in GitHub Actions
- Ensure all dependencies installed

### Build Failures
- Check Node.js version (18 required)
- Delete `node_modules` and `package-lock.json`, then `npm install`
- Check for TypeScript errors

### Cache Issues
- Caches may become stale
- Clear cache in GitHub Actions settings
- Force new cache by updating requirements.txt

## Coverage Reports

Coverage reports are uploaded to Codecov (if configured):
- Backend coverage: `backend-coverage` flag
- ML coverage: `ml-coverage` flag

Setup Codecov:
1. Sign up at https://codecov.io
2. Add repository
3. Add `CODECOV_TOKEN` to GitHub Secrets (optional for public repos)

## Extending the CI

### Add New Job
```yaml
new-job:
  name: New Job
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Do something
      run: echo "Hello"
```

### Add Environment Variables
```yaml
env:
  MY_VAR: value
```

### Add Secrets
1. Go to repository Settings → Secrets
2. Add new secret
3. Use in workflow:
```yaml
env:
  SECRET_KEY: ${{ secrets.MY_SECRET }}
```

### Matrix Testing (Multiple Python Versions)
```yaml
strategy:
  matrix:
    python-version: ['3.9', '3.10', '3.11']
steps:
  - uses: actions/setup-python@v4
    with:
      python-version: ${{ matrix.python-version }}
```

## Best Practices

1. **Keep workflows fast** - Use caching, run jobs in parallel
2. **Fail fast** - Critical errors (syntax) checked first
3. **Informative logs** - Use `--verbose`, `--show-source`
4. **Version pinning** - Pin action versions (`@v4` not `@latest`)
5. **Secrets security** - Never log secrets, use GitHub Secrets
6. **Artifact retention** - Limit to 7 days to save storage
7. **Required checks** - Mark critical jobs as required in branch protection

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Python Setup Action](https://github.com/actions/setup-python)
- [Node Setup Action](https://github.com/actions/setup-node)
- [Codecov Action](https://github.com/codecov/codecov-action)

