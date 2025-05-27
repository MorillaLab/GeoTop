# GEOTOP 
[![arXiv](https://img.shields.io/badge/arXiv-2311.16157-red)](https://arxiv.org/abs/2311.16157)
[![License](https://img.shields.io/badge/License-CC_BY_NC_ND_4.0-green)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/MorillaLab/TopoTransformers/)

Lipschitz-Killing Curvatures and Topological Data Analysis for Machine Learning Image Classification

In this study, we explore the application of Topological Data Analysis (TDA) and Lipschitz-Killing Curvatures (LKCs) as powerful tools for feature extraction and classification in the context of biomedical multiomics problems. TDA allows us to capture topological features and patterns within complex datasets, while LKCs provide essential geometric insights. We investigate the potential of combining both methods to improve classification accuracy. Using a dataset of biomedical images, we demonstrate that TDA and LKCs can effectively extract topological and geometrical features, respectively. The combination of these features results in enhanced classification performance compared to using each method individually. This approach offers promising results and has the potential to advance our understanding of complex biological processes in various biomedical applications. Our findings highlight the value of integrating topological and geometrical information in biomedical data analysis. As we continue to delve into the intricacies of multiomics problems, the fusion of these insights holds great promise for unraveling the underlying biological complexities.

![augmented_images](https://github.com/MorillaLab/MLITLKC/blob/main/Images/augmented_images.png)

<div align="center">
  <h2>Machine Learning Workflow</h2>

  <img src="https://github.com/MorillaLab/MLITLKC/blob/main/Images/ML_workflow_GeoTop.png?raw=true" alt="workflow_GeoTop" width="50%"/>
  
</div>

<p>The framework processes biomedical images through parallel topological and geometric feature extraction pipelines that converge for classification.</p>

<h3>Input & Preprocessing</h3>
<p>Raw images (RGB or grayscale) undergo normalization and tumor-centric alignment, ensuring consistent feature extraction. The preprocessed images branch into two computational streams:</p>

<h3>Topological Pipeline (Left)</h3>
<ul>
  <li><strong>Grayscale Conversion</strong>: Color images are converted to intensity maps</li>
  <li><strong>Superlevel Filtration</strong>: Constructs a nested sequence of binary images from highest to lowest intensities</li>
  <li><strong>Persistence Diagram Generation</strong>: Tracks birth/death of topological features (connected components/H₀, loops/H₁) across thresholds</li>
  <li><strong>Feature Extraction</strong>: Computes 64 descriptors including Betti numbers, persistence entropy, and diagram amplitudes</li>
</ul>

<h3>Geometric Pipeline (Right)</h3>
<ul>
  <li><strong>Multi-threshold Binarization</strong>: Creates 200 threshold-specific binary images</li>
  <li><strong>Component Analysis</strong>: Identifies and tracks connected components across thresholds</li>
  <li><strong>LKC Computation</strong>: Calculates three geometric features per component:
    <ul>
      <li><em>Area</em>: White pixel count (occupation density)</li>
      <li><em>Perimeter</em>: Boundary complexity via Hermine-Agnes algorithm</li>
      <li><em>Euler Characteristic</em>: #Components - #Holes (topological invariant)</li>
    </ul>
  </li>
  <li><strong>Feature Extraction</strong>: Derives 120 descriptors including threshold profiles, derivatives, and summary statistics</li>
</ul>

<h3>Feature Fusion & Classification</h3>
<ul>
  <li><strong>Concatenation</strong>: Combines 64 topological + 120 geometric features</li>
  <li><strong>Feature Selection</strong>: Retains top 100 features via mutual information scoring</li>
  <li><strong>Ensemble Classification</strong>: Random forest (500 trees) trained on fused features achieves:
    <ul>
      <li>87% accuracy (vs. 84% single-modality)</li>
      <li>15-18% reduction in false positives/negatives</li>
    </ul>
  </li>
</ul>

<h3>Key Innovations</h3>
<ul>
  <li><strong>Synergistic Processing</strong>: Maintains topological invariance while capturing geometric nuances</li>
  <li><strong>Computational Efficiency</strong>: Parallel pipelines process 224×224px images in &lt;0.5s</li>
  <li><strong>Clinical Interpretability</strong>: Features map to diagnostic criteria (e.g., perimeter→margin irregularity)</li>
</ul>

<p>The workflow's dual-path architecture addresses the topological equivalence problem while preserving the computational advantages of both methods, as demonstrated in our skin lesion and plant peptide case studies (Figures 2-5).</p>

<!-- ============================================== -->
<div align="left">
  <h1 id="citation">🎈 Citation</h1>
  <hr style="height: 3px; background: linear-gradient(90deg, #EF8E8D, #5755A3); border: none; border-radius: 3px;">
</div>

If you find GeoTop helpful, please cite us.

```bibtex
@misc{abaach2023geotopadvancingimageclassification,
      title={GeoTop: Advancing Image Classification with Geometric-Topological Analysis}, 
      author={Mariem Abaach and Ian Morilla},
      year={2023},
      eprint={2311.16157},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2311.16157}, 
}
```
