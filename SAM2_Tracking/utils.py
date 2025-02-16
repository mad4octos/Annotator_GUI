import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tkinter import *
from tkinter import ttk
import tkinter as tk
from tkinter import filedialog
from tkVideoPlayer import TkinterVideo
import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
import pickle
import glob
import memory_profiler


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
    
''' 
def process_video_segment(file_path, video_writer, frame_names, video_dir, fps, out_fps, SAM2_start):
    with open(file_path, "rb") as f:
        video_segments = pickle.load(f)
    
    print(f"{file_path} successfully loaded!")
    
    for out_frame_idx in range(0, len(frame_names)):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(f"frame {out_frame_idx}/{out_frame_idx * (fps/out_fps) + SAM2_start}")

        image = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
        ax.imshow(image)

        if out_frame_idx in video_segments:
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask, ax, obj_id=out_obj_id)
                
                mask_coords = np.column_stack(np.where(out_mask))
                if mask_coords.size > 0:
                    min_y = mask_coords[:, 0].min()
                    min_x = mask_coords[mask_coords[:, 0] == min_y][:, 1].mean()
                    add_text(ax, f"ID: {out_obj_id}", position=(min_x, min_y - 10))
        
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        #frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        plt.close(fig)
'''