# Quick Start - CI/CD

## ⚡ Quick Commands

### Before Committing

```bash
# Backend
cd backend
black . && flake8 . && pytest tests/ -v

# ML
cd ml
black src/ tests/ && flake8 src/ && pytest tests/ -v

# Frontend
cd frontend
npm run lint && npm run build
```

### Auto-fix Issues

```bash
# Format Python code
black backend/ ml/

# Fix imports
isort backend/ ml/

# Fix TypeScript (auto-fix where possible)
cd frontend && npm run lint -- --fix
```

## 📋 Pre-Push Checklist

- [ ] All tests pass locally
- [ ] Code formatted with `black`
- [ ] No `flake8` errors
- [ ] Commit message follows convention: `feat(scope): description`
- [ ] Updated documentation if needed

## 🚀 CI Pipeline

Your push triggers 4 jobs:

1. **Backend Tests** (~2min) - Lint, format check, pytest
2. **ML Tests** (~2min) - Lint, format check, pytest  
3. **Frontend Build** (~1min) - ESLint, build verification
4. **Integration** (~1min) - Full stack health check

**Total**: ~5-7 minutes

## 🔍 Checking CI Status

1. Go to **Actions** tab on GitHub
2. Click on your commit/PR
3. View individual job logs
4. Green ✅ = pass, Red ❌ = fail

## 🛠️ Common Fixes

### Flake8 Errors
```bash
black .  # Auto-fixes most style issues
```

### Test Failures
```bash
pytest tests/ -v -s  # Verbose with print output
pytest tests/test_file.py::test_name -v  # Run specific test
```

### Build Errors (Frontend)
```bash
rm -rf node_modules .next
npm install
npm run build
```

## 📦 Installation

### First Time Setup

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt

# ML
cd ml
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Frontend
cd frontend
npm install
```

## 🎯 Conventional Commits

```
feat(scope): add new feature
fix(scope): fix bug  
docs(scope): update docs
test(scope): add tests
refactor(scope): refactor code
```

Examples:
```
feat(ml): add SHAP explanations
fix(backend): handle missing columns
docs(readme): update installation steps
```

## 💡 Pro Tips

1. **Run CI locally before pushing** - Catches issues early
2. **Use `--check` flags** - Verify without modifying files
3. **Cache dependencies** - Already done in CI!
4. **Small commits** - Easier to review and debug
5. **Test-driven** - Write tests first, then code

## 📚 Resources

- [Contributing Guide](../../CONTRIBUTING.md)
- [CI Workflow](./workflows/ci.yml)
- [CI Documentation](./workflows/README.md)

## 🆘 Getting Help

**CI failing?**
1. Check the specific job logs
2. Run the same commands locally
3. Review the error messages
4. Check recent changes

**Still stuck?**
- Create an issue with CI logs
- Ask in GitHub Discussions
- Review CONTRIBUTING.md

---

**Remember**: CI is your friend! It catches bugs before they reach production. 🎉

