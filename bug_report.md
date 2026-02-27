---
name: Bug report
about: Report a bug in GeoTop
title: '[BUG] '
labels: bug
assignees: ''
---

## Describe the bug
A clear description of what the bug is.

## Which component fails?
- [ ] Topological pipeline (TDA / persistence diagrams)
- [ ] Geometric pipeline (LKC computation)
- [ ] Feature fusion / selection
- [ ] Random Forest classification
- [ ] GeoTop.ipynb notebook
- [ ] Other: ___

## Minimal reproducible example
```python
from Code.geotop import GeoTop
import numpy as np

img = np.random.rand(224, 224)
model = GeoTop()
# Error occurs here:
feats = model.topological_features(img)
```

## Error traceback
```
Paste full traceback here
```

## Image details
- Dimensions: [e.g. 224×224]
- Format: [RGB / grayscale]
- Domain: [skin lesion / protein embedding / other]

## Environment
- OS: [e.g. Ubuntu 22.04]
- Python version: [e.g. 3.9.12]
- giotto-tda version:
- gudhi version:
- OpenCV version:

## Additional context
Any other context about the problem.
