import numpy as np
import matplotlib.pyplot as plt
import colorsys
import torch 


def get_centroid(mask):
    """
    Function to calculate the centroid of a binary mask. 

    Parameters
    ----------
    mask : tensor of bools
        A tensor of bools representing a mask with shape (width, height)

    Returns
    -------
    tuple of ints
        Tuple with two elements representing the x and y position of 
        the centroid, respectively 
    """

    # Get nonzero indices
    y_indices, x_indices = torch.where(mask)  

    # Exit if mask is empty 
    if len(y_indices) == 0:  
        return None

    # Calculate the centroid in the x and y directions using the mean
    centroid_x = int(x_indices.float().mean())  
    centroid_y = int(y_indices.float().mean()) 

    return (centroid_x, centroid_y)

def get_spaced_colors(n, start_hue=120):
    """
    Generate n well-separated RGB colors starting from `start_hue` 
    using the golden ratio.

    Parameters
    ----------
    n : int
        The number of RGB colors to generate 
    start_hue : int
        The starting hue value e.g. 120 corresponds to green 

    Returns
    -------
    list of tuples of ints 
        A list with each element representing a tuple of RGB 
        values between 0 and 255
    """
    colors = []
    golden_ratio_conjugate = 0.61803398875  # Ensures good spacing
    hue = start_hue / 360  # Convert degrees to range [0,1]
    
    for _ in range(n):
        hue = (hue + golden_ratio_conjugate) % 1  # Shift hue by golden ratio
        rgb = colorsys.hsv_to_rgb(hue, 1, 1)  # Convert HSV to RGB
        colors.append(tuple(int(c * 255) for c in rgb))  # Scale to 0-255
        
    return colors

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def add_text(ax, text, position, fontsize=12, color='yellow'):
    """Adds a text box to the given axis."""
    x, y = position
    # Transform position from pixel coordinates to axis coordinates
    x_axis, y_axis = ax.transData.transform((x, y))
    ax.figure.text(
        x_axis / ax.figure.bbox.width,
        y_axis / ax.figure.bbox.height,
        text,
        fontsize=fontsize,
        color=color,
        bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'),
        ha='center',  # Center-align horizontally
        va='bottom'   # Align the text above the mask
    )