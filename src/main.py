import os
import time

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN


# ============================================================
# SETTINGS
# ============================================================

INPUT_IMAGE = "road.jpg"

OUTPUT_FOLDER = "output"

# ------------------------------------------------------------
# Outline
# ------------------------------------------------------------

CANNY_LOW = 50
CANNY_HIGH = 150

# Bold outline thickness
OUTLINE_THICKNESS = 4

# ------------------------------------------------------------
# Camera approximation
# ------------------------------------------------------------

FOCAL_LENGTH_SCALE = 1.0

# ------------------------------------------------------------
# MSAC
# ------------------------------------------------------------

MSAC_ITERATIONS = 400

MSAC_THRESHOLD = 0.05

# ------------------------------------------------------------
# DBSCAN
# ------------------------------------------------------------

DBSCAN_EPS = 0.035

DBSCAN_MIN_SAMPLES = 12


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# PROGRAM START
# ============================================================

program_start = time.perf_counter()

print("\n================================================")
print("       DENSE IMAGE BASED 2.5D MAPPING")
print("       MSAC + DBSCAN + BOLD OUTLINE")
print("================================================")


# ============================================================
# 1. LOAD IMAGE
# ============================================================

print("\n[1] Loading image...")

image = cv2.imread(
    INPUT_IMAGE
)

if image is None:

    print("\nERROR: Image not found!")

    print(
        "Expected:"
    )

    print(
        os.path.abspath(INPUT_IMAGE)
    )

    raise SystemExit


image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


height, width = image.shape[:2]


print(
    f"Original image size: "
    f"{width} x {height}"
)

print(
    "IMPORTANT: No image resizing will be performed."
)


# ============================================================
# 2. OUTLINE EXTRACTION
# ============================================================

print("\n[2] Detecting object outlines...")

outline_start = time.perf_counter()


gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# Small noise reduction

blur = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)


# Canny edges

edges = cv2.Canny(
    blur,
    CANNY_LOW,
    CANNY_HIGH
)


# ------------------------------------------------------------
# Make outline bold
# ------------------------------------------------------------

outline_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (
        OUTLINE_THICKNESS,
        OUTLINE_THICKNESS
    )
)


bold_edges = cv2.dilate(
    edges,
    outline_kernel,
    iterations=1
)


# ------------------------------------------------------------
# Close small gaps
# ------------------------------------------------------------

close_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3, 3)
)


bold_edges = cv2.morphologyEx(
    bold_edges,
    cv2.MORPH_CLOSE,
    close_kernel
)


outline_mask = (
    bold_edges > 0
)


outline_time = (
    time.perf_counter()
    -
    outline_start
) * 1000


print(
    f"Outline processing: "
    f"{outline_time:.2f} ms"
)


# ============================================================
# 3. LOAD MiDaS
# ============================================================

print("\n[3] Loading MiDaS depth model...")

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)


midas = torch.hub.load(
    "intel-isl/MiDaS",
    "MiDaS_small"
)


midas.to(
    device
)

midas.eval()


# ============================================================
# 4. MiDaS TRANSFORM
# ============================================================

midas_transforms = torch.hub.load(
    "intel-isl/MiDaS",
    "transforms"
)


transform = (
    midas_transforms.small_transform
)


# ============================================================
# 5. DEPTH ESTIMATION
# ============================================================

print("\n[4] Generating dense depth...")

depth_start = time.perf_counter()


input_batch = transform(
    image_rgb
)


input_batch = input_batch.to(
    device
)


with torch.no_grad():

    prediction = midas(
        input_batch
    )


    prediction = torch.nn.functional.interpolate(

        prediction.unsqueeze(1),

        size=(
            height,
            width
        ),

        mode="bicubic",

        align_corners=False

    ).squeeze()


depth = prediction.cpu().numpy()


# ============================================================
# DEPTH NORMALIZATION
# ============================================================

depth_min = depth.min()

depth_max = depth.max()


depth_normalized = (

    depth
    -
    depth_min

) / (

    depth_max
    -
    depth_min
    +
    1e-8

)


depth_time = (
    time.perf_counter()
    -
    depth_start
) * 1000


print(
    f"Depth processing: "
    f"{depth_time:.2f} ms"
)


# ============================================================
# 6. DENSE X,Y,Z PROJECTION
# ============================================================

print("\n[5] Creating original-resolution X,Y,Z...")

projection_start = time.perf_counter()


# ------------------------------------------------------------
# IMPORTANT:
# No resizing here.
#
# U and V have exactly the original image dimensions.
# ------------------------------------------------------------

u = np.arange(
    width,
    dtype=np.float32
)

v = np.arange(
    height,
    dtype=np.float32
)


U, V = np.meshgrid(
    u,
    v
)


# ------------------------------------------------------------
# Approximate camera parameters
# ------------------------------------------------------------

fx = (
    width
    *
    FOCAL_LENGTH_SCALE
)


fy = fx


cx = (
    width
    /
    2.0
)


cy = (
    height
    /
    2.0
)


# ------------------------------------------------------------
# Z
# ------------------------------------------------------------

Z = depth_normalized.astype(
    np.float32
)


# ------------------------------------------------------------
# X
# ------------------------------------------------------------

X = (

    (U - cx)
    *
    Z
    /
    fx

)


# ------------------------------------------------------------
# Y
# ------------------------------------------------------------

Y = (

    (V - cy)
    *
    Z
    /
    fy

)


projection_time = (
    time.perf_counter()
    -
    projection_start
) * 1000


print(
    f"X,Y,Z projection: "
    f"{projection_time:.2f} ms"
)

print(
    f"XYZ dimensions: "
    f"{X.shape}"
)


# ============================================================
# 7. CREATE POINT CLOUD
# ============================================================

print("\n[6] Creating dense spatial points...")

points = np.column_stack(

    (

        X.reshape(-1),

        Y.reshape(-1),

        Z.reshape(-1)

    )

).astype(
    np.float32
)


print(
    "Total spatial points:",
    len(points)
)


# ============================================================
# 8. MSAC DOMINANT PLANE
# ============================================================

print("\n[7] Running MSAC plane extraction...")

msac_start = time.perf_counter()


def fit_plane(
    p1,
    p2,
    p3
):

    v1 = p2 - p1

    v2 = p3 - p1

    normal = np.cross(
        v1,
        v2
    )

    norm = np.linalg.norm(
        normal
    )

    if norm < 1e-8:

        return None


    normal = (
        normal
        /
        norm
    )


    d = -np.dot(
        normal,
        p1
    )


    return np.array(

        [
            normal[0],
            normal[1],
            normal[2],
            d
        ],

        dtype=np.float32

    )


# ------------------------------------------------------------
# MSAC uses a subset for model search.
# This keeps CPU latency reasonable.
# ------------------------------------------------------------

sample_step = max(
    1,
    len(points) // 15000
)


sample_indices = np.arange(
    0,
    len(points),
    sample_step
)


sample_points = points[
    sample_indices
]


rng = np.random.default_rng(
    42
)


best_plane = None

best_cost = float(
    "inf"
)


sample_count = len(
    sample_points
)


# ------------------------------------------------------------
# MSAC iterations
# ------------------------------------------------------------

for _ in range(
    MSAC_ITERATIONS
):

    if sample_count < 3:

        break


    indices = rng.choice(

        sample_count,

        3,

        replace=False

    )


    p1 = sample_points[
        indices[0]
    ]

    p2 = sample_points[
        indices[1]
    ]

    p3 = sample_points[
        indices[2]
    ]


    plane = fit_plane(
        p1,
        p2,
        p3
    )


    if plane is None:

        continue


    a, b, c, d = plane


    distances = np.abs(

        a * sample_points[:, 0]
        +
        b * sample_points[:, 1]
        +
        c * sample_points[:, 2]
        +
        d

    )


    # --------------------------------------------------------
    # MSAC cost
    # --------------------------------------------------------

    truncated_error = np.minimum(

        distances ** 2,

        MSAC_THRESHOLD ** 2

    )


    cost = np.sum(
        truncated_error
    )


    if cost < best_cost:

        best_cost = cost

        best_plane = plane


# ============================================================
# FIND GROUND / DOMINANT PLANE
# ============================================================

if best_plane is not None:

    a, b, c, d = best_plane


    all_distances = np.abs(

        a * points[:, 0]
        +
        b * points[:, 1]
        +
        c * points[:, 2]
        +
        d

    )


    ground_mask_flat = (

        all_distances
        <
        MSAC_THRESHOLD

    )

else:

    ground_mask_flat = np.zeros(

        len(points),

        dtype=bool

    )


ground_mask = ground_mask_flat.reshape(

    height,
    width

)


msac_time = (
    time.perf_counter()
    -
    msac_start
) * 1000


ground_percentage = (

    np.sum(
        ground_mask
    )
    /
    ground_mask.size
    *
    100

)


print(
    f"MSAC processing: "
    f"{msac_time:.2f} ms"
)

print(
    f"Dominant plane points: "
    f"{ground_percentage:.2f}%"
)


# ============================================================
# 9. DBSCAN CLUSTERING
# ============================================================

print("\n[8] Running DBSCAN...")

dbscan_start = time.perf_counter()


# ------------------------------------------------------------
# Only process non-ground points
# ------------------------------------------------------------

non_ground_indices = np.where(
    ~ground_mask_flat
)[0]


non_ground_points = points[
    non_ground_indices
]


print(
    "Non-ground points:",
    len(non_ground_points)
)


# ------------------------------------------------------------
# DBSCAN
# ------------------------------------------------------------

if len(non_ground_points) >= DBSCAN_MIN_SAMPLES:

    dbscan = DBSCAN(

        eps=DBSCAN_EPS,

        min_samples=DBSCAN_MIN_SAMPLES,

        n_jobs=-1

    )


    labels = dbscan.fit_predict(
        non_ground_points
    )

else:

    labels = np.full(

        len(non_ground_points),

        -1,

        dtype=int

    )


dbscan_time = (
    time.perf_counter()
    -
    dbscan_start
) * 1000


# ============================================================
# CLUSTER INFORMATION
# ============================================================

valid_labels = labels[
    labels >= 0
]


if len(valid_labels) > 0:

    cluster_ids = np.unique(
        valid_labels
    )

else:

    cluster_ids = []


cluster_count = len(
    cluster_ids
)


print(
    f"DBSCAN processing: "
    f"{dbscan_time:.2f} ms"
)

print(
    f"Clusters detected: "
    f"{cluster_count}"
)


# ============================================================
# CREATE FULL-SIZE CLUSTER MAP
# ============================================================

cluster_map_flat = np.full(

    len(points),

    -1,

    dtype=np.int32

)


cluster_map_flat[
    non_ground_indices
] = labels


cluster_map = cluster_map_flat.reshape(

    height,
    width

)


# ============================================================
# 10. CREATE 2.5D DEPTH MAP
# ============================================================

print("\n[9] Creating 2.5D map...")

map_start = time.perf_counter()


map_25d = (
    depth_normalized.copy()
)


# ------------------------------------------------------------
# Colorize depth
# ------------------------------------------------------------

map_8bit = (

    map_25d
    *
    255

).astype(
    np.uint8
)


color_map = cv2.applyColorMap(

    map_8bit,

    cv2.COLORMAP_TURBO

)


# ============================================================
# 11. MSAC GROUND VISUALIZATION
# ============================================================

# Make detected ground slightly darker

color_map[
    ground_mask
] = (
    70,
    70,
    70
)


# ============================================================
# 12. DBSCAN CLUSTER HIGHLIGHT
# ============================================================

# Give clustered non-ground areas a brighter appearance

cluster_mask = (
    cluster_map >= 0
)


# Blend clustered areas with original depth

cluster_pixels = color_map[
    cluster_mask
].astype(
    np.float32
)


cluster_pixels = (
    cluster_pixels
    *
    1.15
)


cluster_pixels = np.clip(

    cluster_pixels,

    0,

    255

)


color_map[
    cluster_mask
] = cluster_pixels.astype(
    np.uint8
)


# ============================================================
# 13. BOLD RED OUTLINE
# ============================================================

# ------------------------------------------------------------
# The outline is applied LAST so that MSAC/DBSCAN do not
# overwrite it.
# ------------------------------------------------------------

color_map[
    outline_mask
] = (
    0,
    0,
    255
)


# ============================================================
# 14. SAVE FINAL 2.5D IMAGE
# ============================================================

output_file = os.path.join(

    OUTPUT_FOLDER,

    "2.5D_MSAC_DBSCAN_outline.png"

)


cv2.imwrite(

    output_file,

    color_map

)


# ============================================================
# 15. SAVE DEPTH MAP
# ============================================================

depth_output = (

    depth_normalized
    *
    255

).astype(
    np.uint8
)


cv2.imwrite(

    os.path.join(

        OUTPUT_FOLDER,

        "dense_depth.png"

    ),

    depth_output

)


# ============================================================
# 16. SAVE BOLD OUTLINE
# ============================================================

cv2.imwrite(

    os.path.join(

        OUTPUT_FOLDER,

        "bold_outline.png"

    ),

    bold_edges

)


# ============================================================
# 17. SAVE GROUND MASK
# ============================================================

ground_output = (

    ground_mask.astype(
        np.uint8
    )
    *
    255

)


cv2.imwrite(

    os.path.join(

        OUTPUT_FOLDER,

        "msac_ground_mask.png"

    ),

    ground_output

)


# ============================================================
# 18. SAVE DBSCAN CLUSTER MAP
# ============================================================

dbscan_output = np.zeros(

    (
        height,
        width
    ),

    dtype=np.uint8

)


if cluster_count > 0:

    # Convert every cluster to visible grayscale

    for cluster_id in cluster_ids:

        mask = (
            cluster_map
            ==
            cluster_id
        )

        value = (
            (
                int(cluster_id)
                *
                50
            )
            %
            255
        )

        if value == 0:

            value = 255


        dbscan_output[
            mask
        ] = value


cv2.imwrite(

    os.path.join(

        OUTPUT_FOLDER,

        "dbscan_clusters.png"

    ),

    dbscan_output

)


# ============================================================
# 19. SAVE X Y Z
# ============================================================

np.save(

    os.path.join(

        OUTPUT_FOLDER,

        "X_coordinates.npy"

    ),

    X

)


np.save(

    os.path.join(

        OUTPUT_FOLDER,

        "Y_coordinates.npy"

    ),

    Y

)


np.save(

    os.path.join(

        OUTPUT_FOLDER,

        "Z_coordinates.npy"

    ),

    Z

)


np.save(

    os.path.join(

        OUTPUT_FOLDER,

        "DBSCAN_labels.npy"

    ),

    cluster_map

)


np.save(

    os.path.join(

        OUTPUT_FOLDER,

        "MSAC_ground_mask.npy"

    ),

    ground_mask

)


# ============================================================
# 20. 2.5D MAP TIME
# ============================================================

map_time = (
    time.perf_counter()
    -
    map_start
) * 1000


# ============================================================
# 21. DISPLAY
# ============================================================

result_rgb = cv2.cvtColor(

    color_map,

    cv2.COLOR_BGR2RGB

)


plt.figure(
    figsize=(14, 8)
)


plt.imshow(

    result_rgb,

    interpolation="nearest"

)


plt.title(

    "Dense 2.5D Mapping | "
    "MSAC + DBSCAN + Bold Outline"

)


plt.xlabel(
    "Original Image X"
)


plt.ylabel(
    "Original Image Y"
)


plt.tight_layout()


plt.show()


# ============================================================
# 22. TOTAL LATENCY
# ============================================================

total_latency = (

    outline_time
    +
    depth_time
    +
    projection_time
    +
    msac_time
    +
    dbscan_time
    +
    map_time

)


fps = (

    1000
    /
    total_latency

)


total_program_time = (

    time.perf_counter()
    -
    program_start

) * 1000


# ============================================================
# 23. FINAL RESULTS
# ============================================================

print("\n================================================")
print("                  FINAL RESULTS")
print("================================================")

print(
    f"Original resolution : "
    f"{width} x {height}"
)

print(
    "Output resolution   : "
    f"{color_map.shape[1]} x "
    f"{color_map.shape[0]}"
)

print(
    "\nNo resizing performed."
)

print("\n---------------- LATENCY ----------------")

print(
    f"Outline             : "
    f"{outline_time:.2f} ms"
)

print(
    f"Depth               : "
    f"{depth_time:.2f} ms"
)

print(
    f"XYZ projection      : "
    f"{projection_time:.2f} ms"
)

print(
    f"MSAC                : "
    f"{msac_time:.2f} ms"
)

print(
    f"DBSCAN              : "
    f"{dbscan_time:.2f} ms"
)

print(
    f"2.5D map            : "
    f"{map_time:.2f} ms"
)

print(
    "----------------------------------------"
)

print(
    f"TOTAL LATENCY       : "
    f"{total_latency:.2f} ms"
)

print(
    f"ESTIMATED FPS       : "
    f"{fps:.2f}"
)

print(
    f"\nMSAC ground         : "
    f"{ground_percentage:.2f}%"
)

print(
    f"DBSCAN clusters     : "
    f"{cluster_count}"
)

print(
    f"\nTotal program time  : "
    f"{total_program_time:.2f} ms"
)

print("\n---------------- OUTPUT ----------------")

print(
    "2.5D map:"
)

print(
    os.path.abspath(
        output_file
    )
)

print("\n================================================")
print("                     DONE")
print("================================================")
