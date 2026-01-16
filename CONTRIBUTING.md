# Contributing to GraphRAG-Viz

Thank you for your interest in contributing to GraphRAG-Viz! This document provides guidelines for contributing to this Glass Box implementation of the GraphRAG pipeline.

## Glass Box Philosophy

This project is built on the principle of **complete transparency and interpretability**. All contributions should maintain this philosophy:

- ✅ All processing must be traceable
- ✅ Maintain provenance information at every step
- ✅ Provide clear logging and statistics
- ✅ Keep dependencies minimal
- ✅ Document all design decisions

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/GraphRAG-Viz.git
   cd GraphRAG-Viz
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```
4. Set up your environment:
   ```bash
   cp .env.example .env
   # Add your OpenAI API key
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write clear docstrings for all public functions and classes
- Keep functions focused and modular

### Documentation

- Update the README.md if you add new features
- Add docstrings to all new modules, classes, and functions
- Include inline comments for complex logic
- Update the tutorial.ipynb if relevant

### Testing

- Write tests for new functionality
- Ensure existing tests pass
- Test with different document types and sizes
- Verify transparency features (logging, provenance)

### Transparency Requirements

When adding new features, ensure:

1. **Provenance**: Track where data comes from
2. **Logging**: Log important steps and decisions
3. **Metadata**: Include metadata about processing
4. **Statistics**: Provide statistics for transparency
5. **Interpretability**: Make results explainable

## Types of Contributions

### Bug Fixes

- Create an issue describing the bug
- Submit a PR with the fix and test
- Reference the issue in your PR

### New Features

Before implementing a new feature:
1. Open an issue to discuss the feature
2. Ensure it aligns with the Glass Box philosophy
3. Consider backward compatibility
4. Document the feature thoroughly

### Documentation

- Fix typos or unclear documentation
- Add examples and tutorials
- Improve API documentation
- Create guides for specific use cases

### Performance Improvements

- Profile before optimizing
- Maintain transparency while improving performance
- Document performance gains
- Ensure correctness is preserved

## Pull Request Process

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the guidelines above

3. Test your changes:
   ```bash
   python -m pytest tests/
   python example.py
   ```

4. Commit your changes with clear messages:
   ```bash
   git commit -m "Add feature: brief description"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a Pull Request with:
   - Clear description of changes
   - Reference to related issues
   - Screenshots (if UI changes)
   - Test results

## Code Review Process

All PRs will be reviewed for:

- Code quality and style
- Adherence to Glass Box philosophy
- Test coverage
- Documentation completeness
- Performance impact
- Backward compatibility

## Community Guidelines

- Be respectful and inclusive
- Help others learn
- Share knowledge openly
- Focus on constructive feedback
- Celebrate contributions

## Questions?

- Open an issue for questions
- Join discussions in existing issues
- Check the README for basic questions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make GraphRAG-Viz better! 🎉
