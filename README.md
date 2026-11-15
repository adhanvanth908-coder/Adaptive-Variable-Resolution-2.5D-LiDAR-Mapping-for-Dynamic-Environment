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
