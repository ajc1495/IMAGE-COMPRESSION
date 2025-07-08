# K-means Clustering with Image Compression
## Algorithm Foundation
K-means clustering operates on the principle of partitioning data into K distinct clusters by minimizing within-cluster variance. The algorithm employs Lloyd's method, which consists of iterative updates through alternating assignment and update steps. The core objective function minimizes the sum of squared distances between data points and their assigned cluster centroids.

The algorithm uses Euclidean distance as the primary similarity metric, calculating the straight-line distance between data points and centroids. For RGB image applications, this translates to measuring color similarity in three-dimensional space. The mathematical foundation ensures that points within the same cluster exhibit maximum similarity while maintaining maximum dissimilarity between different clusters.
## Core Implementation
###  K-means Algorithm Theory
The K-means algorithm is an iterative procedure that automatically clusters similar data points together. Given a training set $$\\{x^{(1)}, x^{(2)}, ..., x^{(m)}\\}$$  the algorithm groups data into cohesive clusters through the following process:

### Core Functions to Implement

### Exercise 1: Finding Closest Centroids (find_closest_centroids)

### Mathematical Foundation:

For every example 
$x^{(i)}$
 , assign it to the closest centroid:
$$c^{(i)} := j \quad \mathrm{that \; minimizes} \quad ||x^{(i)} - \mu_j||^2$$

where:

* $c^{(i)}$
  is the index of the closest centroid to 
$x^{(i)}$
 

* $\mu_j$
  is the position of the j'th centroid

### Implementation Requirements:

* Input: Data matrix X and centroid locations centroids

* Output: One-dimensional array idx containing centroid assignments

* Key operations: Calculate Euclidean distance using np.linalg.norm() and find minimum using np.argmin()

### Exercise 2: Computing Centroid Means (compute_centroids)

### Mathematical Foundation:

For every centroid $\mu_k$, recompute its position as:

$$\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} x^{(i)}$$

where $C_k$ is the set of examples assigned to centroid $k$.

### Implementation Requirements:

* Input: Data matrix X, assignments idx, and number of clusters K

* Output: Updated centroid positions

* Key operations: Use np.mean() with axis=0 to compute means across assigned points

## Project Sections

### Section 1: Implementing K-means
1.1 Finding closest centroids - Core assignment logic

1.2 Computing centroid means - Centroid update mechanism

### Section 2: K-means on Sample Dataset
Apply the implemented functions to a sample 2D dataset

Visualize clustering results and algorithm convergence

### Section 3: Random Initialization
Explore the importance of proper centroid initialization

Understand why multiple random initializations are recommended

Learn about selecting solutions based on cost function minimization

### Section 4: Image Compression with K-means
4.1 Dataset - Working with image data (bird_small.png)

4.2 K-Means on image pixels - Treat RGB values as feature vectors

4.3 Compress the image - Replace colors with cluster centroids

## Centroid Assignment Process
The assignment step determines cluster membership by calculating distances from each data point to all centroids. Each point is assigned to the cluster with the nearest centroid, effectively partitioning the dataset into K disjoint sets. This process employs vectorized operations in NumPy implementations to achieve computational efficiency.

## Centroid Update Mechanism
The update step recalculates centroid positions as the mean of all points assigned to each cluster. For continuous variables, the centroid value represents the arithmetic mean of member values. This iterative refinement continues until convergence criteria are met.

## Implementation Architecture
### Dependencies and Libraries
The implementation relies on several key Python libraries:

* NumPy: Provides fundamental array operations and mathematical functions for efficient computation

* Matplotlib: Enables data visualization and cluster plotting capabilities

* Scikit-learn: Offers optimized K-means implementation with additional features

### Core Algorithm Structure
The implementation follows a modular design with distinct components for initialization, iteration, and convergence detection. The algorithm maintains centroids as a collection of K points in the feature space, with each iteration updating both assignments and centroid positions.

### Initialization Methods
Multiple initialization strategies are available:

* Random initialization: Selects K data points randomly as initial centroids

* K-means++: Employs probabilistic selection to spread initial centroids

* Deterministic methods: Use specific criteria like maximum distance or density-based selection

### Iteration Framework
The iterative process alternates between assignment and update phases until convergence. Each iteration maintains cluster assignments and updates centroids based on current memberships. The algorithm tracks changes in centroid positions to determine convergence.

## Convergence and Optimization
### Convergence Criteria
The algorithm employs multiple convergence criteria to ensure optimal clustering:

* Centroid stability: Monitors minimal movement of cluster centers between iterations

* Assignment consistency: Checks for unchanged cluster assignments across iterations

* Maximum iterations: Provides computational limits to prevent infinite loops

### Possible Optimization Techniques
Several optimization strategies enhance performance:

* Vectorized operations: Utilize NumPy broadcasting for efficient distance calculations

* Incremental updates: Modify centroids based on individual point movements rather than complete recalculation

* Cluster pruning: Eliminate distant clusters from consideration during assignment

## Image Compression Application
### RGB Color Space Processing
Image compression represents a practical application where K-means clusters similar pixel colors. Each pixel's RGB values form a three-dimensional feature vector, with the algorithm grouping similar colors into K clusters. The compression process replaces original pixel values with their corresponding cluster centroids.

### Compression Methodology
The compression workflow involves:

* Pixel vectorization: Converting image pixels from 2D arrays to 1D vectors

* Clustering application: Grouping similar colors into K representative clusters

* Color replacement: Substituting original colors with cluster centroids

* Size reduction: Achieving compression through reduced color palette

## Implementation Considerations
## Data Preprocessing
Effective K-means implementation requires careful data preparation:

* Feature scaling: Normalizing data to prevent bias from varying scales

* Dimensionality management: Handling high-dimensional data through techniques like PCA

* Outlier treatment: Addressing extreme values that may skew clustering results

### Parameter Selection
Critical parameters influence algorithm performance:

* Number of clusters (K): Determined through methods like elbow analysis or silhouette scoring

* Initialization strategy: Affects convergence speed and final cluster quality, hence chosen from a permuted training set randomly

* Convergence tolerance: Balances accuracy with computational efficiency

## Practical Applications
Computer Vision Applications
K-means finds extensive use in:

* Image segmentation: Partitioning images into meaningful regions

* Color quantization: Reducing color palettes for compression

* Object recognition: Preprocessing for feature extraction

### Data Analysis Applications
The algorithm serves various analytical purposes:

* Customer segmentation: Grouping customers by behavior patterns

* Market research: Identifying consumer segments

* Anomaly detection: Detecting outliers in datasets

## Visualization and Interpretation
### Cluster Visualization
Effective visualization employs:

* Scatter plots: Displaying data points colored by cluster assignment

* Centroid marking: Highlighting cluster centers with distinct markers

* Boundary visualization: Showing cluster separation regions

## Conclusion
K-means clustering represents a fundamental unsupervised learning technique with broad applicability across domains. The implementation combines mathematical rigor with computational efficiency, making it suitable for diverse applications from image compression to customer segmentation. Success depends on appropriate parameter selection, data preprocessing, and understanding of the algorithm's assumptions and limitations. The modular implementation architecture facilitates customization and optimization for specific use cases while maintaining algorithmic integrity.
