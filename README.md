# scRNA-DRComparison
In this study, we systematically evaluated 26 representative single-cell dimensionality reduction (DR) methods applied across a broad spectrum of single-cell sequencing scenarios, encompassing 50 real-world datasets and 50 synthetic datasets. Based on our proposed unified mathematical framework, we categorized existing methods into four major classes and systematically evaluated their performance and robustness across multiple dimensions. This study not only establishes the most comprehensive benchmarking system to date but also theoretically elucidates the underlying mechanisms behind performance variations among different methods. Consequently, it provides actionable and universally applicable guidelines for method selection in practical applications.

## Datasets
All 50 authentic scRNA-seq datasets used in this study are publicly available and can be downloaded from the Gene Expression Omnibus (GEO), Sequence Read Archive (SRA), or ArrayExpress databases. For specific sources, please refer to the Supplementary Table S2.
The synthetic dataset was generated using Splatter, see the Supplementary Table S3 for detailed parameters.

## Results
The results of the dimension reduction performed on the 100 datasets mentioned above using the 26 methods employed in this study, along with the corresponding evaluation metrics, can be downloaded from [Figshare](https://doi.org/10.6084/m9.figshare.32064900).
The real and simulate folders contain the results of dimensionality reduction; the downsampling folder contains the runtime and peak memory usage of each method on downsampled data; the metric folder stores the results of various evaluation metrics, with the score folder containing the overall scores for each method, corresponding to Figure 3 in the paper.
