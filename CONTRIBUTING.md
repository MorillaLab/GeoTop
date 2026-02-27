# Contributing to GeoTop

Thank you for your interest! Contributions that extend GeoTop's geometric or topological feature sets, improve computational efficiency, or validate it on new image domains are very welcome.

## 🐛 Reporting Bugs

Open a [GitHub Issue](https://github.com/MorillaLab/GeoTop/issues) with:
- The notebook or script where the error occurs
- Your environment (OS, Python version, giotto-tda version, OpenCV version)
- The full error traceback
- Image dimensions and format if relevant

## 💡 Suggesting Features

Open an issue tagged `enhancement`. Good examples:
- New geometric descriptors (fractal dimension, curvature profiles)
- Alternative filtrations (Čech, alpha, cubical)
- Faster LKC computation for large images
- Support for 3D volumetric data
- New downstream classifiers (XGBoost, neural network head)

## 🔧 Submitting Code

1. Fork the repository and create a branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install flake8 pytest
   ```
3. Add your changes to `Code/`. Include:
   - Docstrings with mathematical definitions for any new geometric/topological descriptor
   - A corresponding unit test in `tests/`
4. Run tests before submitting:
   ```bash
   pytest tests/ -v
   ```
5. Lint:
   ```bash
   flake8 Code/ --max-line-length=127
   ```
6. Clear notebook outputs before committing.
7. Open a pull request against `main`.

## 📐 Mathematical Conventions

- Feature names should reference the mathematical object they represent (e.g., `betti_0`, `persistence_entropy_h1`, `euler_characteristic`)
- Docstrings should cite the relevant paper or textbook formula
- For new LKC variants, reference Adler & Taylor (2007) or equivalent

## 📜 License

By contributing code, you agree your work will be released under GPL-3.0.  
Figures and documentation fall under CC BY-NC-ND 4.0.
