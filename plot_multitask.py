import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import itertools
# import mayavi.mlab as mlab


def downsample_sum(arr):
    return np.add.reduceat(arr, np.arange(0, len(arr), 10))

def min_max_normalize(arr):
    """Normalize a list/array to the range [0, 1]."""
    arr = np.array(arr, dtype=float)
    rng = arr.max() - arr.min()
    if rng == 0:
        return np.zeros_like(arr)
    return (arr - arr.min()) / rng

# def load_json_files(folder):
#     """
#     Loads each JSON file in the folder.
#     Returns a dictionary mapping filename (without path) to its normalized token_use_counts numpy array.
#     """
#     data = {}
#     json_files = glob.glob(os.path.join(folder, "*.json"))
#     for filepath in json_files:
#         with open(filepath, "r") as fp:
#             jdata = json.load(fp)
#             norm_counts = min_max_normalize(jdata["token_use_counts"])
#             filename = os.path.basename(filepath)
#             data[filename] = norm_counts
#     return data

def load_json_files(subfolder):
    """
    Loads each JSON file in the subfolder, extracts token_use_counts,
    and normalizes using predefined dataset sizes.
    """
    data = {}
    json_files = glob.glob(os.path.join(subfolder, "*.json"))
    for filepath in json_files:
        with open(filepath, "r") as fp:
            jdata = json.load(fp)
            dataset_name = os.path.basename(filepath).split("_0.json")[0]
            size = DATASET_SIZES.get(dataset_name, 1)  # Avoid division by zero
            norm_counts = np.array(jdata["token_use_counts"]) / size  # Normalize using dataset size
            filename = os.path.basename(filepath)
            data[filename] = norm_counts  # Store normalized values
    return data


# Mapping from dataset names to task names
TASK_MAPPING = {
    "voxceleb_sv": "SV",
    "er": "ER",
    "libriasr": "ASR",
    "librisqa": "QA",
    "clotho_audio_cap": "ACAP",
    "cv_trans": "En2Zh"
}

# Dataset sizes for normalization
DATASET_SIZES = {
    "voxceleb_sv": 37611,
    "libriasr": 2620,
    "librisqa": 2620,
    "er": 554,
    "clotho_audio_cap": 1045,
    "cv_trans": 15530
}

TASK_ORDER = ["ASR", "ER", "SV", "QA", "En2Zh", "ACAP"]


def extract_task_name(filename):
    """Extract dataset name from filename and return the corresponding task name."""
    dataset_name = filename.split("_0.json")[0]
    return TASK_MAPPING.get(dataset_name, dataset_name)  # Default to dataset_name if no mapping


def process_subfolder(subfolder):
    """
    For a given subfolder, loads its JSON files and creates a single figure containing a grid
    of subplots—one per unique pair of JSON files. In each subplot, we plot a grouped bar chart
    (i.e. a histogram) showing the entire pair of token_use_counts lists. Each list is normalized
    to [0, 1], and for each pair the first list is sorted and the second list is reordered accordingly.
    
    In the subplot, the blue bars correspond to the first JSON file and the red bars to the second.
    The x-axis represents the token indices.
    """
    print(f"Processing subfolder: {subfolder}")
    json_data = load_json_files(subfolder)
    if len(json_data) < 2:
        print(f"Not enough JSON files in {subfolder} to compare pairs. Skipping...")
        return

    # Get unique pairs of filenames.
    pairs = list(itertools.combinations(json_data.keys(), 2))
    n_pairs = len(pairs)
    
    # Determine grid dimensions for subplots (roughly square)
    import math
    ncols = math.ceil(math.sqrt(n_pairs))
    nrows = math.ceil(n_pairs / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), squeeze=False)
    axes_flat = axes.flatten()
    
    for idx, (file1, file2) in enumerate(pairs):
        ax = axes_flat[idx]
        list1 = json_data[file1]
        list2 = json_data[file2]
        
        # Sort list1 and reorder list2 accordingly.
        sort_idx = np.argsort(list1)
        list1_sorted = list1[sort_idx]
        list2_sorted = list2[sort_idx]
        
        n_tokens = len(list1_sorted)
        x = np.arange(n_tokens)
        width = 0.1
        
        # Plot the two sets of bars side-by-side.
        # ax.bar(x - width, list1_sorted, width=width, color='blue', label=file1)
        # ax.bar(x + width, list2_sorted, width=width, color='red', label=file2)
        width = 0.3  # Increase spacing
        ax.bar(x - width/2, list1_sorted, width=width, color='blue', alpha=0.6, label=file1, edgecolor='blue')
        ax.bar(x + width/2, list2_sorted, width=width, color='red', alpha=0.6, label=file2, edgecolor='red')

        
        ax.set_title(f"{file1} vs {file2}", fontsize=10)
        ax.set_xlabel("Token Index", fontsize=8)
        ax.set_ylabel("Normalized Count", fontsize=8)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.legend(fontsize=8)
    
    # Hide any unused subplots.
    for j in range(len(pairs), len(axes_flat)):
        axes_flat[j].axis('off')
    
    fig.suptitle(f"Histogram Comparisons in Subfolder: {os.path.basename(subfolder)}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join(subfolder, "token_uses.png")
    plt.savefig(output_path)
    plt.close(fig)
    
import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def process_subfolder3D(subfolder):
    """
    Loads JSON files in a subfolder and creates a 3D bar plot where:
    - X-axis represents token indices.
    - Y-axis represents different JSON files.
    - Z-axis represents the normalized token values.
    
    One JSON file is sorted, and others are reordered accordingly.
    """
    print(f"Processing subfolder: {subfolder}")
    json_data = load_json_files(subfolder)
    filenames = list(json_data.keys())
    num_files = len(filenames)

    if num_files < 2:
        print(f"Not enough JSON files in {subfolder} to compare. Skipping...")
        return

    # Extract task names and enforce the given order
    task_name_map = {f: extract_task_name(f) for f in filenames}
    filenames = sorted(filenames, key=lambda f: TASK_ORDER.index(task_name_map[f]))

    # Choose a reference JSON file for sorting (e.g., the first one)
    ref_filename = filenames[0]
    ref_list = json_data[ref_filename]

    # Sort reference list and reorder all other lists accordingly
    sort_idx = np.argsort(ref_list)[::-1]
    sorted_ref_list = ref_list[sort_idx]

    reordered_data = {ref_filename: sorted_ref_list}
    for filename in filenames[1:]:
        reordered_data[filename] = json_data[filename][sort_idx]

    # Convert to 3D data for plotting
    n_tokens = len(sorted_ref_list)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=-30)

    width = 0.5  # Width of the bars
    colors = plt.cm.viridis(np.linspace(0.2, 1, num_files)[::-1])  # Color mapping for each file

    y_positions = np.arange(num_files)  # Numeric positions for tasks on the Y-axis
    task_names = [extract_task_name(f) for f in filenames]  # Extracted task names

    for i, filename in enumerate(filenames):
        x = np.arange(n_tokens)  # Token indices
        y = np.full(n_tokens, y_positions[i])  # Y-axis (task index)
        z = np.zeros(n_tokens)  # Start at z=0
        dz = reordered_data[filename]  # Heights of bars

        ax.bar3d(x, y, z, 2*width, width, dz, color=colors[i], alpha=0.7, label=task_names[i])

    # Labels & Formatting
    # ax.set_xlabel("Token Index")
    # ax.set_ylabel("Task")
    # ax.set_zlabel("Normalized Token Count")

    # Set Y-axis labels to show task names instead of indices
    ax.set_yticks(np.arange(num_files+1))
    ax.set_yticklabels([""] + task_names, fontsize=10, rotation=180+135)  # Rotated to align with x-axis

    # Reduce white space
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)

    # Save the figure instead of showing
    # output_path = os.path.join(subfolder, "comparison_3D_histogram.png")
    # plt.savefig(output_path, dpi=300, bbox_inches='tight')
    output_path = os.path.join(subfolder, "comparison_3D_histogram.pdf")
    plt.savefig(output_path, format="pdf", dpi=600, bbox_inches='tight')
    plt.close(fig)

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def process_subfolder_heatmap(subfolder):
    """
    Loads JSON files in a subfolder and creates a heatmap where:
    - X-axis represents token indices.
    - Y-axis represents different tasks (datasets).
    - Color intensity represents normalized token usage.
    
    One JSON file is sorted, and others are reordered accordingly.
    """
    print(f"Processing subfolder: {subfolder}")
    json_data = load_json_files(subfolder)
    filenames = list(json_data.keys())
    num_files = len(filenames)

    if num_files < 2:
        print(f"Not enough JSON files in {subfolder} to compare. Skipping...")
        return

    # Extract task names and enforce the given order
    task_name_map = {f: extract_task_name(f) for f in filenames}
    filenames = sorted(filenames, key=lambda f: TASK_ORDER.index(task_name_map[f]))

    # Choose a reference JSON file for sorting (e.g., the first one)
    ref_filename = filenames[0]
    ref_list = json_data[ref_filename]

    # Sort reference list and reorder all other lists accordingly
    sort_idx = np.argsort(ref_list)[::-1]
    sorted_ref_list = ref_list[sort_idx]

    reordered_data = {ref_filename: sorted_ref_list}
    for filename in filenames[1:]:
        reordered_data[filename] = json_data[filename][sort_idx]

    # Convert data into a matrix for heatmap
    heatmap_data = np.array([reordered_data[f] for f in filenames])

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        heatmap_data,
        cmap="viridis",
        xticklabels=20,  # Reduce number of x-axis labels
        yticklabels=[extract_task_name(f) for f in filenames],
        cbar_kws={'label': 'Normalized Token Count'}
    )

    ax.set_xlabel("Token Index (Sorted)")
    ax.set_ylabel("Task")
    
    # Save the figure
    # output_path = os.path.join(subfolder, "comparison_heatmap.png")
    # plt.savefig(output_path, dpi=300, bbox_inches='tight')
    output_path = os.path.join(subfolder, "comparison_heatmap.pdf")
    plt.savefig(output_path, format="pdf", dpi=600, bbox_inches='tight')
    plt.close(fig)


def process_parent_folder(parent_folder):
    """
    Walks through each immediate subfolder in the given parent folder and processes
    each subfolder separately.
    """
    for entry in os.scandir(parent_folder):
        if entry.is_dir():
            process_subfolder3D(entry.path)

# Specify your parent folder here.
parent_folder = "evaluations/multi_task/"  # Change this to your actual parent folder path.
process_parent_folder(parent_folder)
