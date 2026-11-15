# Dense Image-Based 2.5D Mapping

### MiDaS + MSAC + DBSCAN + Canny Edge Detection

A computer vision project that transforms a single image into a dense 2.5D representation using monocular depth estimation, spatial coordinate projection, dominant-plane extraction, object clustering, and edge visualization.

## Overview

This project combines multiple computer vision techniques to generate a depth-based spatial representation of an image.

The system uses MiDaS for dense depth estimation, MSAC for dominant-plane extraction, DBSCAN for non-ground point clustering, and Canny edge detection for object outline visualization.

The final output is a colorized 2.5D map with ground regions, clustered objects, and bold red outlines.

## Features

- Dense depth estimation using MiDaS Small
- Original-resolution depth map generation
- Dense X, Y, Z coordinate projection
- MSAC dominant-plane extraction
- Ground and non-ground segmentation
- DBSCAN spatial clustering
- Canny edge detection
- Bold red object outlines
- Turbo color depth visualization
- Dense coordinate array generation
- Ground mask and cluster map generation
- Processing latency measurement
- Estimated FPS calculation
- Automatic output generation

## System Architecture

```text
             INPUT IMAGE
                  |
                  v
        IMAGE PREPROCESSING
                  |
         +--------+--------+
         |                 |
         v                 v
    CANNY EDGE          MiDaS SMALL
    DETECTION           DEPTH ESTIMATION
         |                 |
         v                 v
    BOLD OUTLINE       DENSE DEPTH MAP
                           |
                           v
                  X, Y, Z PROJECTION
                           |
                           v
                  DENSE POINT CLOUD
                           |
                           v
                  MSAC PLANE EXTRACTION
                           |
                           v
                GROUND / NON-GROUND
                      SEGMENTATION
                           |
                           v
                    DBSCAN CLUSTERING
                           |
                           v
                  2.5D MAP GENERATION
                           |
                           v
                    FINAL OUTPUT
```

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Image processing and edge detection |
| PyTorch | Deep learning inference |
| MiDaS | Monocular depth estimation |
| NumPy | Dense coordinate computation |
| Matplotlib | Result visualization |
| Scikit-learn | DBSCAN clustering |

## Project Structure

```text
Dense-2.5D-Mapping/
│
├── data/
│   └── road.jpg
│
├── src/
│   └── main.py
│
├── output/
│   ├── 2.5D_MSAC_DBSCAN_outline.png
│   ├── dense_depth.png
│   ├── bold_outline.png
│   ├── msac_ground_mask.png
│   ├── dbscan_clusters.png
│   ├── X_coordinates.npy
│   ├── Y_coordinates.npy
│   ├── Z_coordinates.npy
│   ├── DBSCAN_labels.npy
│   └── MSAC_ground_mask.npy
│
├── docs/
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Dense-2.5D-Mapping.git
```

```bash
cd Dense-2.5D-Mapping
```

### 2. Create Virtual Environment

Windows:

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

## Requirements

```text
opencv-python
torch
numpy
matplotlib
scikit-learn
```

## Execution

1. Place your input image inside the `data` directory.
2. Configure the input image path in `main.py`.
3. Execute the program.

```bash
python src/main.py
```

The generated results will be saved automatically inside the `output` directory.

## Output Results

| Output | Description |
|---|---|
| 2.5D Mapping | Final depth visualization with MSAC, DBSCAN, and outlines |
| Dense Depth Map | Normalized depth representation |
| Ground Mask | Dominant-plane segmentation |
| Cluster Map | DBSCAN cluster visualization |
| Bold Outline | Canny edge visualization |
| X Coordinates | Horizontal spatial coordinates |
| Y Coordinates | Vertical spatial coordinates |
| Z Coordinates | Normalized depth coordinates |

## Algorithm Details

### 1. MiDaS Depth Estimation
Generates dense relative depth information from a single RGB image.

### 2. XYZ Coordinate Projection
Projects image pixels into approximate spatial coordinates using simplified camera parameters.

### 3. MSAC Plane Extraction
Estimates a dominant plane and separates ground candidates from non-ground regions.

### 4. DBSCAN Clustering
Groups spatially related non-ground points into clusters.

### 5. Canny Edge Detection
Extracts image edges and enhances them using morphological dilation.

### 6. 2.5D Visualization
Combines depth information, ground segmentation, cluster highlighting, and bold red outlines into a final visualization.

## Performance Analysis

The program calculates:

- Outline processing time
- Depth inference time
- XYZ projection time
- MSAC processing time
- DBSCAN processing time
- 2.5D map generation time
- Total estimated processing latency
- Estimated FPS

## Limitations

- MiDaS generates relative depth rather than calibrated metric depth.
- XYZ coordinates are approximate and depend on camera assumptions.
- MSAC plane extraction does not guarantee semantic ground detection.
- DBSCAN computational cost can increase significantly with dense point clouds.
- Estimated FPS is based on selected processing stages and may differ from end-to-end FPS.
- First-time MiDaS model download requires internet access.

## Future Development

- Camera calibration and metric depth integration
- Efficient point-cloud processing
- Interactive 3D visualization
- Real-time camera processing
- Advanced ground segmentation
- Optimized clustering algorithms
- GUI-based image processing interface

## Author

**Dhanvanth A**

Electronics and Communication Engineering  
Computer Vision | Image Processing | AI & Embedded Systems

## License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for details.

---

**Dense Image-Based 2.5D Mapping using MiDaS, MSAC, DBSCAN, and Canny Edge Detection.**
