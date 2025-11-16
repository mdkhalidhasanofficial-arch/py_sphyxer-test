# py_sphyxer Documentation Setup

This folder contains all necessary configuration files to generate documentation for the `py_sphyxer` project.

## How to build the documentation

1. Make sure Python and pip are installed.
2. Install Sphinx and the ReadTheDocs theme:
   ```bash
   pip install sphinx sphinx_rtd_theme
   ```
3. Navigate to the `docs` folder:
   ```bash
   cd docs
   ```
4. Build the HTML documentation:
   ```bash
   make html
   ```
5. View the generated site by opening:
   ```
   _build/html/index.html
   ```

## GitHub Pages Setup (optional)
- Commit the entire `docs/_build/html` folder to your repository.
- In your GitHub repository settings, go to **Pages** and set the source to `docs/_build/html`.
- Your documentation will be viewable online.

---
**Project:** py_sphyxer  
**Author:** Ptolemy  
**Theme:** sphinx_rtd_theme
