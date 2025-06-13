# Clustering System

A clean, modular clustering system for automated grading of programming assignments. This system groups similar submissions together using various embedding techniques and clustering algorithms.

## Architecture Overview

```
src/cluster/
├── embedders/          # Different embedding strategies
├── processors/         # Data processing and submission handling  
├── clustering/         # Clustering algorithms and management
├── scripts/           # Training and evaluation scripts
└── utils/             # Shared utilities
```

## Key Components

### 1. Embedders (`embedders/`)
- **Base Embedder**: Abstract interface for all embedders
- **Java Embedder**: Semantic embeddings for Java code using StarCoder2/Ollama
- **Repomix Embedder**: Embeddings of processed codebases using repomix
- **Factory**: Creates embedders based on configuration

### 2. Processors (`processors/`)
- **Submission Processor**: Handles extraction and processing of student submissions
- Supports the data structure: `task_folder/studentname_....zip`

### 3. Clustering (`clustering/`)
- **Cluster Manager**: Manages clustering training, prediction, and evaluation
- Supports multiple clustering algorithms (K-Means, DBSCAN, Hierarchical)
- Algorithm-agnostic design using embedder interface

### 4. Scripts (`scripts/`)
- **Training Script**: Train clustering models on submissions
- **Evaluation Script**: Evaluate clustering performance and quality
- **Cache Clearing Script**: Clear cached embeddings to force regeneration

## Data Structure

The system expects submissions organized as:
```
src/faiss/data/
├── task1_SOLID/
│   ├── studentname1_123_456_submission.zip
│   ├── studentname2_789_012_submission.zip
│   └── ...
├── task2_PreparingRefactoring/
│   └── ...
└── task3_NOAMCalculator/
    └── ...
```

## Quick Start

### 1. Train a Clustering Model

```bash
# Train clustering on Java code embeddings
python src/cluster/scripts/train_clustering.py \
    --task-folder src/faiss/data/task1_SOLID \
    --embedder-type java \
    --model-name starCoder2:7b \
    --n-clusters 5 \
    --output-dir models/task1

# Train clustering on repomix embeddings  
python src/cluster/scripts/train_clustering.py \
    --task-folder src/faiss/data/task1_SOLID \
    --embedder-type repomix \
    --model-name starCoder2:3b \
    --n-clusters 4 \
    --output-dir models/task1_repomix
```

### 2. Evaluate Clustering

```bash
# Evaluate with auto-generated organized output directory
python src/cluster/scripts/evaluate_clustering.py models/task1_java_kmeans

# Evaluate with reference clusters for accuracy analysis
python src/cluster/scripts/evaluate_clustering.py models/task1_java_kmeans \
    --reference-clusters src/cluster/metrics/task1_SOLID_reference_clusters.csv

# Evaluate on different task folder
python src/cluster/scripts/evaluate_clustering.py models/task1_java_kmeans \
    --task-folder src/faiss/data/task2_PreparingRefactoring

# Custom output directory
python src/cluster/scripts/evaluate_clustering.py models/task1_java_kmeans \
    --output-dir custom_evaluation_results
```

**Output Directory Structure:**
- Auto-generated: `cluster_evaluation_results/task1_SOLID_java_kmeans_starCoder2_3b_evaluation/`
- Contains: `evaluation_report.json`, `detailed_cluster_report.json`, `misassigned_students.csv`

### 3. Clear Cached Embeddings

```