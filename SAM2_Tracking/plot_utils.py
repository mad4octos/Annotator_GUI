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