# GraphRAG-Viz Tests

This directory contains tests for the GraphRAG-Viz Glass Box pipeline.

## Running Tests

### Run all tests:
```bash
pytest tests/
```

### Run with verbose output:
```bash
pytest tests/ -v
```

### Run specific test file:
```bash
pytest tests/test_pipeline.py
```

### Run specific test:
```bash
pytest tests/test_pipeline.py::TestDocumentChunker::test_basic_chunking
```

## Test Coverage

Current test coverage includes:

- **Document Chunking**: Tests for text segmentation and overlap
- **Graph Building**: Tests for knowledge graph construction
- **Community Detection**: Tests for graph clustering
- **Configuration**: Tests for configuration handling

## Adding New Tests

When adding new tests:

1. Create test files with the `test_` prefix
2. Name test classes with the `Test` prefix
3. Name test methods with the `test_` prefix
4. Follow the AAA pattern: Arrange, Act, Assert
5. Include docstrings explaining what is being tested

## Mocking External APIs

Tests that require OpenAI API calls should be mocked in CI/CD environments:

```python
from unittest.mock import patch

@patch('openai.OpenAI')
def test_extraction(mock_openai):
    # Your test here
    pass
```

## Test Philosophy

Following the Glass Box approach:

- Tests should verify transparency features
- Check that provenance is maintained
- Verify logging and statistics
- Ensure interpretability is preserved

## CI/CD Integration

These tests can be integrated into CI/CD pipelines. Remember to:

- Set `OPENAI_API_KEY` as a secret (or mock API calls)
- Install dependencies before running tests
- Generate coverage reports
