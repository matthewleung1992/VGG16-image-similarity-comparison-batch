# VGG16 Multi-Layer Image Similarity Analysis

Batch processing system for computing image similarity using VGG16 convolutional neural network features. Developed for neuroscience research at RIKEN Center for Brain Science to analyze visual stimulus patterns.

## Overview

This tool extracts deep learning features from multiple layers of a pre-trained VGG16 model and computes Euclidean distances between image pairs. The system uses parallel processing to efficiently handle large image datasets.

## Key Features

- **Multi-layer feature extraction**: Analyzes 11 layers of VGG16 architecture simultaneously
- **Parallel processing**: Utilizes multiprocessing with configurable worker count for performance optimization
- **Resume capability**: Automatically resumes from previous results if interrupted
- **Memory efficient**: Queue-based processing prevents memory overflow with large datasets
- **Research-grade output**: Results saved as CSV for downstream statistical analysis

## Technical Architecture

### Neural Network Layers Analyzed
```python
'block1_conv1'  # Early edge detection
'block1_pool'   # Low-level feature pooling
'block2_pool'   # Mid-level patterns
'block3_conv3'  # Complex feature detection
'block3_pool'   
'block4_conv1'  # High-level features
'block4_pool'   
'block5_pool'   # Semantic representations
'flatten'       
'fc1'           # Fully connected layer 1
'fc2'           # Fully connected layer 2
```

### Processing Pipeline

1. **Image Loading**: Automatically discovers all `.jpg` files in directory tree
2. **Mission Planning**: Generates pairwise comparison list (excludes same-directory pairs)
3. **Feature Extraction**: VGG16 pre-trained on ImageNet extracts normalized features
4. **Distance Calculation**: Computes Euclidean distance for each layer's feature space
5. **Result Storage**: Appends results to CSV with checkpoint capability

## Requirements
```bash
tensorflow>=2.x
numpy
pandas
Pillow
```

Install dependencies:
```bash
pip install tensorflow numpy pandas pillow
```

## Usage

### Basic Usage

Place script in the directory containing your image folders:
```bash
python vgg16_similarity.py
```

The script will:
- Scan current directory for all `.jpg` images
- Generate pairwise comparisons (excluding within-folder pairs)
- Process images using 15 parallel workers
- Save results to `results.csv`

### Configuration

**Adjust number of workers** (line 165):
```python
num_workers = 15  # Modify based on CPU cores
```

**Modify layers of interest** (lines 38-49):
```python
layernames = [
    'block1_conv1',
    # Add or remove layers as needed
]
```

**Change image limit per folder** (line 128):
```python
if i >= 3:  # Limit images per directory
    break
```

### Resume Functionality

If interrupted, the script automatically resumes from `results.csv`:
- Previously computed pairs are skipped
- Only remaining comparisons are processed
- No duplicate calculations

## Output Format

Results are saved as CSV with the following structure:

| Image 1 | Image 2 | block1_conv1 | block1_pool | ... | fc2 |
|---------|---------|--------------|-------------|-----|-----|
| img1.jpg | img2.jpg | 0.542 | 0.631 | ... | 0.891 |

Each column (except first two) represents the Euclidean distance in that layer's feature space.

## Performance

**Processing Speed:**
- ~2-3 seconds per image pair (depends on hardware)
- 15 workers can process ~300 pairs/minute
- Checkpoint saves every 50 results (configurable)

**Tested Configuration:**
- 1,000+ images across multiple directories
- ~500,000 pairwise comparisons
- Runtime: ~24-48 hours for full batch

## Technical Notes

### Multiprocessing Architecture

- Uses `multiprocessing.Queue` for inter-process communication
- Each worker maintains separate TensorFlow session (prevents GPU conflicts)
- Input thread distributes tasks round-robin across workers
- Main thread collects results and writes to CSV

### Feature Normalization

Features are L2-normalized before distance calculation:
```python
feature_normalized = feature / np.linalg.norm(feature)
```

This ensures:
- Scale-invariant comparisons
- Distances represent angular similarity in feature space
- Consistent metrics across layers

### Memory Management

- Queue max size: 128 items per worker (prevents memory overflow)
- Images resized to 224×224 (VGG16 input requirement)
- Processed in batches, not loaded all at once

## Use Cases

Originally developed for:
- Visual stimulus similarity analysis in neuroscience experiments
- Comparing neural response patterns to similar images
- Identifying perceptually similar stimuli across categories

Applicable to:
- Image dataset deduplication
- Visual similarity search systems
- Content-based image retrieval
- Dataset quality analysis

## Limitations

- **GPU recommended**: CPU-only processing is significantly slower
- **Same-directory exclusion**: Assumes images in same folder don't need comparison (Our folders were sorted into items like "avocado/avocado1.jpg" etc.)
- **Format restriction**: Currently only processes `.jpg` files
- **No distance threshold**: Computes all pairs regardless of similarity


## License

This code was developed for research purposes at RIKEN Center for Brain Science.

---

## Troubleshooting

**Issue: "TensorFlow not found"**
```bash
pip install tensorflow --break-system-packages  # For system Python
```

**Issue: Multiprocessing deadlock on Windows**
Uncomment line 30:
```python
set_start_method("spawn")
```

**Issue: Out of memory errors**
Reduce number of workers:
```python
num_workers = 5  # Lower number
```

**Issue: Slow processing**
- Ensure TensorFlow is using GPU
- Check `nvidia-smi` for GPU utilization
- Reduce image batch size if needed
