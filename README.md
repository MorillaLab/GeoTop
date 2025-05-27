# GEOTOP 
[![arXiv](https://img.shields.io/badge/arXiv-2311.16157-red)](https://arxiv.org/abs/2311.16157)
[![License](https://img.shields.io/badge/License-CC_BY_NC_ND_4.0-green)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/MorillaLab/TopoTransformers/)

Lipschitz-Killing Curvatures and Topological Data Analysis for Machine Learning Image Classification

In this study, we explore the application of Topological Data Analysis (TDA) and Lipschitz-Killing Curvatures (LKCs) as powerful tools for feature extraction and classification in the context of biomedical multiomics problems. TDA allows us to capture topological features and patterns within complex datasets, while LKCs provide essential geometric insights. We investigate the potential of combining both methods to improve classification accuracy. Using a dataset of biomedical images, we demonstrate that TDA and LKCs can effectively extract topological and geometrical features, respectively. The combination of these features results in enhanced classification performance compared to using each method individually. This approach offers promising results and has the potential to advance our understanding of complex biological processes in various biomedical applications. Our findings highlight the value of integrating topological and geometrical information in biomedical data analysis. As we continue to delve into the intricacies of multiomics problems, the fusion of these insights holds great promise for unraveling the underlying biological complexities.

![augmented_images](https://github.com/MorillaLab/MLITLKC/blob/main/Images/augmented_images.png)

<div align="center">
  <h2>GeoTop Framework Workflow</h2>
  
  <img src="https://github.com/MorillaLab/MLITLKC/blob/main/Images/ML_workflow_GeoTop.png?raw=true" alt="workflow_GeoTop" width="800"/>

  <h3>Topological-Geometric Synergy in GeoTop</h3>
</div>

<h4>Key Components</h4>
<ul>
  <li><strong>Panel (a)</strong>: Synthetic validation case showing:
    <ul>
      <li>Left: Gaussian field with additive noise</li>
      <li>Right: Persistence diagram with two equivalent topological components</li>
    </ul>
  </li>
  
  <li><strong>Panel (b)</strong>: Clinical discrimination analysis:
    <ul>
      <li>Top: Topologically equivalent structures (circles vs dumbbell) with identical H₀ components</li>
      <li>Bottom: Normalized geometric profiles showing:
        <ul>
          <li>Perimeter/area differences between lesion subtypes</li>
          <li>Malignant cases show 2.3× greater variance (p<0.01)</li>
        </ul>
      </li>
    </ul>
  </li>
  
  <li><strong>Panel (c)</strong>: Combined feature space:
    <ul>
      <li>PCA projection (63.2% variance in PC1)</li>
      <li>Clear separation of diagnostic classes (benign/malignant)</li>
      <li>Yellow highlights show resolved borderline cases</li>
    </ul>
  </li>
</ul>

<h4>Interpretation</h4>
<blockquote>
The figure demonstrates how GeoTop resolves the <em>topological equivalence problem</em> by combining:
<ol>
  <li>Persistent homology's robustness to noise</li>
  <li>LKC's sensitivity to morphological details</li>
</ol>
Resulting in 12% accuracy improvement over single-modality approaches.
</blockquote>

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
