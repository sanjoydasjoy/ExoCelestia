#!/bin/bash
# Run backend tests

echo "Running unit tests..."
pytest tests/ -v --cov=app --cov-report=term-missing

echo ""
echo "Running linter..."
# Uncomment if you want to add linting
# flake8 app/ tests/

echo ""
echo "Done!"

