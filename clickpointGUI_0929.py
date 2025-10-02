import numpy as np
import cv2
from tkinter import *
from tkinter import ttk, filedialog, simpledialog
from tkinter import messagebox
import customtkinter as ctk
from collections import defaultdict
from PIL import Image, ImageTk
import csv
import os
import pandas as pd
from collections import OrderedDict

# Initialize Global variables
current_frame_index = [0]
xLocation = [0]
yLocation = [0]
ClickType = [1]  # Default to positive click (1)
ObjID = [0]
paused = [False]
playing_task = None
annotations = []
video_speed = 1.0  # Playback speed multiplier
fps = 30  # Default FPS, will update dynamically based on video
out_fps = 3 # Default temporal resolution for SAM2. 
special_frame_start = 0  # Default starting frame for SAM2
special_frame_interval = 10  # Default, will calculate dynamically
ObjType = ["Parrotfish"]  # Default fish family
orig_vid_width, orig_vid_height = 0, 0
aspect_ratio = 1.78
cap = None
total_frames = 0
video_size_x, video_size_y = 600, 400 # Define video player sizes. 
cap_last_index = None       # index of the frame currently cached in cap_last_frame
cap_last_frame = None       # numpy array (BGR) of the cached frame
frame_cache = OrderedDict()
frame_cache_size = 50
canvas_image_id = None

# Create the main window
root = ctk.CTk()
root.title("Video Annotation GUI")
root.geometry("1200x700") # May need to change this line to fit different computer screens


##### Load and Play Video Functions #######
# Play and Pause Function
def pause():
    if root.focus_get() and isinstance(root.focus_get(), (ctk.CTkEntry, Entry)):
        return
    paused[0] = not paused[0]
    button_play_pause.configure(text="Pause ||" if not paused[0] else "Play ▶")
    if not paused[0]:
        play_video()

# Get Cache for Images
def get_cache(index):
    global frame_cache
    if index in frame_cache:
        frame=frame_cache.pop(index)
        frame_cache[index] = frame
        return frame 
    return None

# Populate cache 
def put_cache(index, frame):
    global frame_cache
    frame_cache[index] = frame
    frame_cache.move_to_end(index)
    if len(frame_cache) > frame_cache_size:
        frame_cache.popitem(last=False)

# Load Video Function
def load_video():
    global orig_vid_height, orig_vid_width, aspect_ratio, special_frame_interval
    global cap, fps, total_frames, cap_last_frame, cap_last_index, current_frame_index
    global video_size_x, video_size_y

    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
    if not file_path:
        return
    if 'cap' in globals() and cap is not None:
        try:
            cap.release()
        except Exception:
            pass

    cap = cv2.VideoCapture(file_path)

    # Video metadata
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Update FPS dynamically
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = total_frames - 1
    orig_vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    aspect_ratio = orig_vid_width/orig_vid_height
    video_size_y = int(video_size_x/aspect_ratio)

    special_frame_interval = max(1, round(fps)/out_fps)  # Calculate interval for SAM2 extracted frames.
    # Reset cap cache
    cap_last_frame = None
    cap_last_index = None
    
    # Reset current index and slider
    slider_frame.configure(to=max(0, total_frames))
    current_frame_index[0] = 0
    update_frame_display()
    play_video()

# Read cached frames efficiently
def read_frame(index):
    """
    Efficient frame reader:
      - reuses cached last frame if index == cap_last_index
      - uses cap.read() if index == cap_last_index + 1 (fast sequential)
      - otherwise uses cap.set(...) then cap.read() (random access, slower)
    Returns BGR np.array or None.
    """
    global cap , cap_last_frame, cap_last_index

    if cap is None or total_frames==0:
        return None
    
    if index < 0 or index >= total_frames:
        return None
    
    # Check cache
    cached = get_cache(index)
    if cached is not None:
        return cached
    
    if (cap_last_index is not None) and (index == cap_last_index + 1):
        ret, frame = cap.read()
        if not ret:
            return None
        cap_last_index = index
        cap_last_frame = frame
        put_cache(index,frame)
        return frame
    
    # Random access: set position and read
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = cap.read()
    if not ret: 
        return None
    cap_last_index = index
    cap_last_frame = frame
    put_cache(index, frame)
    return frame

# Resize frame and convert to photo
def frame_to_photo(frame_bgr, resize = True):
    if frame_bgr is None:
        return None
    if resize:
        resized = cv2.resize(frame_bgr, (video_size_x, video_size_y))
    else:
        resized = frame_bgr
    frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(frame_rgb))

def display_frame_index(index, reuse_if_same=True):
    global current_frame_index, canvas_image_id
    frame = read_frame(index)
    if frame is None: 
        return False
    photo = frame_to_photo(frame, resize=True)
    if photo is None:
        return False
   # Keep reference alive
    canvas_video.photo = photo
    
    if canvas_image_id is None:
        canvas_image_id = canvas_video.create_image(0,0, anchor="nw", image=photo)
    else:
        canvas_video.itemconfig(canvas_image_id, image=photo)

    # Keep crosshairs on top
    canvas_video.tag_raise(crosshair_h)
    canvas_video.tag_raise(crosshair_v)
    # Update scroll region to new zoom size
    canvas_video.config(scrollregion=(0,0,video_size_x,video_size_y))
    return True

def update_frame_display():
    """Resize current frame to video_size_x/y and display it in the label."""
    idx = current_frame_index[0]
    if idx < 0 or idx >= total_frames:
        return
    display_frame_index(idx)

# Play Video
def play_video():
    global playing_task, current_frame_index

    # Stop if paused or at end
    if paused[0] or current_frame_index[0] >= total_frames:
        playing_task = None
        return

    # Display current frame
    ok = display_frame_index(current_frame_index[0])
    if not ok:
        playing_task = None
        return
    
    slider_frame.set(current_frame_index[0])
    update_time_display()

    current_frame_index[0] += 1
    delay_ms = int(1000 / (fps * video_speed)) if fps and video_speed else 3
    playing_task = root.after(delay_ms, play_video)
    
# Function for zooming widtet
def on_slider_change(value):
    global video_size_x, video_size_y
    video_size_x = int(float(value)) 
    video_size_y = int(video_size_x / aspect_ratio)
    update_frame_display()
    # Update scroll region after resizing
    canvas_video.config(scrollregion=(0,0, video_size_x, video_size_y))

##### Annotation Functions #######

# Canvas Click Event
def canvas_click_events(event):

    x_in_image = canvas_video.canvasx(event.x)
    y_in_image = canvas_video.canvasy(event.y)

    # Ignore clicks outside of image
    if x_in_image < 0 or x_in_image >= video_size_x or y_in_image < 0 or y_in_image>= video_size_y:
        print("Click outside displayed image area. Try again.")
        return
    
    orig_x = x_in_image * (orig_vid_width / video_size_x) 
    orig_y = y_in_image * (orig_vid_height / video_size_y)

    xLocation[0] = orig_x
    yLocation[0] = orig_y
    # Draw temporary circle on display frame
    frame = read_frame(current_frame_index[0])
    if frame is None:
        return
    resized=cv2.resize(frame, (video_size_x, video_size_y))
    frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Draw in display space, not in original space
    cv2.circle(frame_rgb, (int(x_in_image), int(y_in_image)), 2, (255, 0, 0), -1)

    photo = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
    canvas_video.photo = photo
    canvas_video.itemconfig(canvas_image_id, image=photo)

    # Refresh the original frame after 2 seconds
    root.after(2000, update_frame_display)

# Add Annotation Function
def add_annotation():
    ObjID[0] = fish_name.get()
    annotation = {
        "Frame": current_frame_index[0],
        "ClickType": ClickType[0],
        "ObjID": ObjID[0],
        "ObjType": ObjType[0],
        "Location": np.array([round(xLocation[0], 3), round(yLocation[0], 3)])
    }
    annotations.append(annotation)
    update_annotation_table()

    print(f"Annotation added for {current_frame_index[0]}.")

#Add Entry Hotkey
def add_entry(event=None):
    """Adds a new annotation with ClickType=3 and (0,0) location."""
    global annotations, ObjType, ObjID, current_frame_index, root
    if root.focus_get() and isinstance(root.focus_get(), (ctk.CTkEntry, Entry)):
        return
    ObjID[0] = fish_name.get()
    annotation = {
        "Frame": current_frame_index[0],
        "ClickType": 3,
        "ObjID": ObjID[0],
        "ObjType": ObjType[0],
        "Location": np.array([0.0, 0.0])
    }
    annotations.append(annotation)
    print(f"Annotation added: {annotation}")
    update_annotation_table()

#Add Exit Hotkey
def add_exit(event=None):
    """Adds a new annotation with ClickType=4 and (0,0) location."""
    global annotations, ObjType, ObjID, current_frame_index, root
    if root.focus_get() and isinstance(root.focus_get(), (ctk.CTkEntry, Entry)):
        return
    ObjID[0] = fish_name.get()
    annotation = {
        "Frame": current_frame_index[0],
        "ClickType": 4,
        "ObjID": ObjID[0],
        "ObjType": ObjType[0],
        "Location": np.array([0.0, 0.0])
    }
    annotations.append(annotation)
    print(f"Annotation added: {annotation}")
    update_annotation_table()

# Update Annotations Table
def update_annotation_table():
    for row in treeview.get_children():
        treeview.delete(row)
    for i, annotation in enumerate(annotations):
        treeview.insert(
            "",
            "end",
            iid=i,
            values=(
                i,
                annotation["Frame"],
                annotation["ClickType"],
                annotation["ObjID"],
                annotation["ObjType"],
                annotation["Location"][:2],
            ),
        )
        treeview.see(i)

# Delete Selected Annotations
def delete_selected():
    selected_items = treeview.selection()
    for item in sorted(selected_items, reverse=True):
        annotations.pop(int(item))
    update_annotation_table()

# Delete All Annotations
def delete_all():
    response = messagebox.askquestion(
        "Confirm Delete All",
        "Are you sure you want to delete all annotations?",
        icon="warning"
    )
    if response == "yes":
        annotations.clear()
        update_annotation_table()

# Edit Selected Annotation
def edit_selected():
    selected_items=treeview.selection()
    if not selected_items:
        messagebox.showwarning("No selection","Please select an annotation to edit")
        return
    idx = int(selected_items[0])
    annotation=annotations[idx]

    # Popup window
    edit_win = Toplevel(root)
    edit_win.title(f"Edit Annotation {idx}")
    # Fields to Edit
    Label(edit_win, text="Frame:").grid(row=0, column=0)
    frame_entry = Entry(edit_win)
    frame_entry.insert(0, annotation["Frame"])
    frame_entry.grid(row=0, column=1)

    Label(edit_win, text="ClickType:").grid(row=1, column=0)
    click_entry = Entry(edit_win)
    click_entry.insert(0, annotation["ClickType"])
    click_entry.grid(row=1, column=1)

    Label(edit_win, text="ObjID:").grid(row=2, column=0)
    objid_entry = Entry(edit_win)
    objid_entry.insert(0, annotation["ObjID"])
    objid_entry.grid(row=2, column=1)

    Label(edit_win, text="ObjType:").grid(row=3, column=0)
    objtype_options = ["Parrotfish", "Surgeonfish", "Damselfish", "Ball", "Garden_eel", "Other"]
    objtype_combo = ttk.Combobox(edit_win, values = objtype_options, state="readonly")
    objtype_combo.set(annotation["ObjType"])
    objtype_combo.grid(row=3, column=1)

    Label(edit_win, text="Location (x,y):").grid(row=4, column=0)
    loc_entry = Entry(edit_win)
    loc_entry.insert(0, f"{annotation['Location'][0]},{annotation['Location'][1]}")
    loc_entry.grid(row=4, column=1)
    
    def save_changes():
        try:
            annotation["Frame"] = int(frame_entry.get())
            annotation["ClickType"] = int(click_entry.get())
            annotation["ObjID"] = objid_entry.get()
            annotation["ObjType"] = objtype_combo.get()
            x, y = map(float, loc_entry.get().split(","))
            annotation["Location"] = np.array([round(x, 3), round(y, 3)])
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
            return
        update_annotation_table()
        edit_win.destroy()
    Button(edit_win, text="Save", command=save_changes).grid(row=5, column=0, columnspan=2, pady=5)

# Bind edit event to double-click event
def on_double_click(event):
    # make sure user actually clicked a row, not blank space
    selected = treeview.identify_row(event.y)
    if selected:
        treeview.selection_set(selected)  # ensure the row is selected
        edit_selected()  # reuse your existing function

# Toggle Click Type
def toggle_click_type():
    if ClickType[0] == 1:
        ClickType[0] = 0
        button_toggle_click.configure(text="Negative Click")
    elif ClickType[0] == 0:
        ClickType[0] = 2
        button_toggle_click.configure(text="Bite")
    else:
        ClickType[0] = 1
        button_toggle_click.configure(text="Positive Click")

# Toggle Fish Family
def toggle_obj_type():
    if ObjType[0] == "Parrotfish":
        ObjType[0] = "Surgeonfish"
        button_toggle_obj_type.configure(text="Surgeonfish")
    elif ObjType[0] == "Surgeonfish":
        ObjType[0] = "Damselfish"
        button_toggle_obj_type.configure(text="Damselfish")
    elif ObjType[0] == "Damselfish":
        ObjType[0] = "Ball"
        button_toggle_obj_type.configure(text="Ball")
    elif ObjType[0] == "Ball":
        ObjType[0]="Garden_eel"
        button_toggle_obj_type.configure(text="Garden_eel")
    elif ObjType[0] == "Garden_eel":
        ObjType[0] = "Other"
        button_toggle_obj_type.configure(text="Other")
    else:
        ObjType[0] = "Parrotfish"
        button_toggle_obj_type.configure(text="Parrotfish")


# Import Previous Annotations Function
def import_annotations():

    # Open a file dialog to select a .npy file
    file_path = filedialog.askopenfilename(
        title = "Import Previous Annotations",
        filetypes=(("Numpy Files", "*.npy"), ("CSV Files", "*.csv"), ("All Files", "*.*"))
    )
#Check if the user selected a file or canceled the dialog
    if not file_path:
        return

    try:
        file_extension = os.path.splitext(file_path)[1].lower()

        #Case 1: Load .npy file
        if file_extension == ".npy":
            imported_annotations=np.load(file_path, allow_pickle=True)

            #Check if the loaded file has the expected format
            if not isinstance(imported_annotations, np.ndarray):
                raise ValueError("The selected file does not contain compatible annotation data")
        
        #Append annotations to existing list
            for annotation in imported_annotations:
                if isinstance(annotation, dict) and all(key in annotation for key in ["Frame", "ClickType", "ObjID", "ObjType", "Location"]):
                    annotations.append(annotation)
                else:
                    raise ValueError("One or more annotations in the file have an invalid format.")
        
        #Case 2: Load .csv file
        elif file_extension == ".csv":
            imported_annotations = pd.read_csv(file_path)

            #check if the necessary columns are in the .csv
            required_columns = ["Frame", "ClickType", "ObjID", "ObjType", "Location"]
            if not all(col in imported_annotations.columns for col in required_columns):
                raise ValueError(f"The CSV file must contain the following columns: {', '.join(required_columns)}.")
            for _, row in imported_annotations.iterrows():
                location_str = row["Location"]
                try:
                    location = eval(location_str) if isinstance(location_str, str) else location_str
                except Exception:
                    location = row["Location"]
                annotation = {
                    "Frame": int(row["Frame"]),
                    "ClickType": row["ClickType"],
                    "ObjID": row["ObjID"],
                    "ObjType": row["ObjType"],
                    "Location": np.array(location)
                }
                annotations.append(annotation)
                        #Update the annotation table with new data

        else:
            raise ValueError("The selected file is neither a valid .npy nor .csv file.")

        #Update the annotation table with new data
        update_annotation_table()
        # Optionally, show a message to the user that the import was successful
        messagebox.showinfo("Import Successful", f"Successfully imported {len(imported_annotations)} annotations.")

    except Exception as e:
        # Handle any errors (e.g., file not found, invalid format, etc.)
        messagebox.showerror("Error", f"An error occurred while importing annotations: {str(e)}")

def check_annotations():
    """
    Checks that for every ObjID in annotations, the number of entries and exits are equal.
    If a mismatch is found, a warning is shown and the user can choose to continue or go back.
    Returns True if the user decides to continue, False if the user chooses to go back.
    """
    # Dictionary to count ClickTypes 3 (entry) and 4 (exit) for each ObjID
    counts = defaultdict(lambda:{3: 0, 4: 0})
    for annotation in annotations:
        click = annotation.get("ClickType")
        if click in [3,4]:
            ObjID=annotation.get("ObjID")
            counts[ObjID][click] += 1

    # Collect any mismatches
    mismatches = []
    for ObjID, count in counts.items():
        if count[3] != count[4]:
            mismatches.append(
                f"ObjID '{ObjID}': Entries (ClickType 3) = {count[3]}, Exits (ClickType 4) = {count[4]}"
            )
    # If there are mismatches, prompt the user
    if mismatches:
        message = (
            "Mismatched entry or exit points detected for the following fish labels:\n\n"
            + "\n".join(mismatches)
            + "\n\nDo you want to continue anyways?"
        )
        # Ask yes/no returns True for Yes (continue) and False for No (go back)
        if not messagebox.askyesno("Annotation Mismatch", message):
            # User chose to go back
            return False
        
    return True


# Save Annotations Function
def save_annotations():
    # First, check for any entry/exit mismatches
    if not check_annotations():
        print("Check annotations: user chose to go back.")
        return # User chose to go back, do not proceed with saving

    print("No mismatches; proceeding to save annotations.")

    file_name = file_name_var.get().strip() or "annotations"  # Default name if none provided
    cwd = os.getcwd()
    save_dir = filedialog.askdirectory(title="Select Folder to Save Annotations", 
                                       initialdir=cwd)
    if not save_dir:
        print("Save cancelled: no directory specified")
        return
    
    # Split annotations by Click Type
    general_annotations = [
        a for a in annotations if a["ClickType"] in [0, 1, 3, 4]
    ]
    bite_annotations = [
        a for a in annotations if a["ClickType"] == 2
    ]

    msg = f"File(s) saved in {save_dir}. "

    # Case 1: Save locations as .npy for SAM2
    if save_locations_var.get():
        np.save(os.path.join(save_dir, f"{file_name}_annotations.npy"), general_annotations)
        msg += f"Location annotations saved as '{file_name}_annotation.npy'. "
    
    # Case 2: Save bites as .csv
    if save_bites_var.get():
        csv_path = os.path.join(save_dir, f"{file_name}_bites.csv")
        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = ["Frame", "ClickType", "ObjID", "ObjType", "Location"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for annotation in bite_annotations:
                writer.writerow({
                    "Frame": annotation["Frame"],
                    "ClickType": annotation["ClickType"],
                    "ObjID": annotation["ObjID"],
                    "ObjType": annotation["ObjType"],
                    "Location": annotation["Location"].tolist()
                })
        msg += f"Bite annotations saved as '{file_name}_bites.csv'. "

    # Case 3: Save ALL annotations in a single csv
    if save_all_var.get():
        all_csv_path = os.path.join(save_dir, f"{file_name}_all.csv")
        with open(all_csv_path, "w", newline="") as csvfile:
            fieldnames = ["Frame", "ClickType", "ObjID", "ObjType", "Location", "x", "y"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for annotation in annotations:
                writer.writerow({
                    "Frame": annotation["Frame"],
                    "ClickType": annotation["ClickType"],
                    "ObjID": annotation["ObjID"],
                    "ObjType": annotation["ObjType"],
                    "Location": annotation["Location"].tolist(),
                    "x": annotation["Location"][0], # first number
                    "y": annotation["Location"][1] # second number
                })
        msg += f"All annotations saves as '{file_name}_all.csv"

    messagebox.showinfo("Saved", msg)

# Update Frame from Slider
def update_frame_from_slider(event):
    global playing_task
    if playing_task is not None:
        root.after_cancel(playing_task)
    current_frame_index[0] = int(slider_frame.get())
    display_frame_index(current_frame_index[0])
    update_time_display()
    update_frame_display()

# Update Time Display
def update_time_display():
    frame_num = current_frame_index[0]
    time_in_seconds = frame_num / fps
    minutes = int(time_in_seconds // 60)
    seconds = int(time_in_seconds % 60)
    time_display_var.set(f"Time: {minutes:02}:{seconds:02} | Frame: {frame_num} | Speed: {video_speed:.1f}x")

    if frame_num >= special_frame_start and (frame_num - special_frame_start) % special_frame_interval == 0:
        special_frame_var.set("SAM2 Frame: Annotate Fish Position")
        label_special_frame.configure(font=("Arial", 14, "bold"), fg="red")
    else:
        special_frame_var.set("----")
        label_special_frame.configure(font=("Arial", 12), fg="black")

# Advance Frame
def advance_frame(delta):
    global playing_task, current_frame_index
    if root.focus_get() and isinstance(root.focus_get(), (ctk.CTkEntry, Entry)):
        return
    if playing_task is not None:
        try:
            root.after_cancel(playing_task)
        except Exception:
            pass
        playing_task = None
        paused[0] = True
    new_idx = current_frame_index[0] + delta
    new_idx = max(0, min(total_frames - 1, new_idx))
    current_frame_index[0] = new_idx
    
    ok = display_frame_index(new_idx)
    if ok:
        slider_frame.set(new_idx)
        update_time_display()


# Navigate to Next Special Frame
def next_special_frame():
    global playing_task, current_frame_index
    if root.focus_get() and isinstance(root.focus_get(), (ctk.CTkEntry, Entry)):
        return
    if playing_task is not None:
        try:
            root.after_cancel(playing_task)
        except Exception:
            pass
        playing_task = None
        paused[0] = True
    
    while current_frame_index[0] < total_frames:
        current_frame_index[0] += 1
        ok = display_frame_index(current_frame_index[0])

        if (current_frame_index[0] - special_frame_start) % special_frame_interval == 0:
            break
    ok = display_frame_index(current_frame_index[0])
    if ok:
        slider_frame.set(current_frame_index[0])
        update_time_display()


#Navigate to Previous Special Frame
def prev_special_frame():
    if root.focus_get() and isinstance(root.focus_get(), (ctk.CTkEntry, Entry)):
        return
    global playing_task, current_frame_index
    if playing_task is not None:
        try:
            root.after_cancel(playing_task)
        except Exception:
            pass
        playing_task = None
        paused[0] = True

    while current_frame_index[0] > 0:
        current_frame_index[0] -= 1
        if (current_frame_index[0] - special_frame_start) % special_frame_interval == 0:
            break
    ok = display_frame_index(current_frame_index[0])
    if ok:
        slider_frame.set(current_frame_index[0])
        update_time_display()

    
# Adjust Playback Speed
def adjust_speed(delta):
    global video_speed
    video_speed = max(0.1, video_speed + delta)
    update_time_display()

def reset_speed():
    global video_speed
    video_speed = 1.0
    update_time_display()

# Change cursor to a cross-hairs when it enters the video player
def on_mouse_enter(event):
    # Change cursor to crosshair when mouse enters the video area
    event.widget.config(cursor="none")

def on_mouse_leave(event):
    # Revert to default cursor when mouse leaves
    event.widget.config(cursor="")
def on_mouse_wheel(event):
    global video_size_x, video_size_y
    zoom_step = 50 # Adjust for sensitivity
    # Windows/Linux
    if hasattr(event, "delta"):
        if event.delta>0:
            video_size_x += zoom_step # zoom in
        else:
            video_size_x = max(100, video_size_x - zoom_step)
    # Macos
    elif hasattr(event, "num"):
        if event.num==4:
            video_size_x += zoom_step
        elif event.num == 5:
            video_size_x = max(100, video_size_x - zoom_step)
    # maintain aspect ratio
    video_size_y = int(video_size_x/aspect_ratio)

    # Update display and scroll region
    update_frame_display()
    canvas_video.config(scrollregion=(0,0,video_size_x, video_size_y))

    # Update slider to reflect current zoom
    slider_zoom.set(video_size_x)

PAN_STEP = 1  # pixels per key press

def pan_canvas(event):
    if root.focus_get() and isinstance(root.focus_get(), (ctk.CTkEntry, Entry)):
        return
    if event.keysym == "w":
        canvas_video.yview_scroll(-PAN_STEP, "units")
    elif event.keysym == "s":
        canvas_video.yview_scroll(PAN_STEP, "units")
    elif event.keysym == "a":
        canvas_video.xview_scroll(-PAN_STEP, "units")
    elif event.keysym == "d":
        canvas_video.xview_scroll(PAN_STEP, "units")


def update_crosshair(event):
    x = canvas_video.canvasx(event.x)
    y = canvas_video.canvasy(event.y)

    size = 10

    # Horizontal line across the video
    canvas_video.coords(crosshair_h, x-size, y, x+size, y)
    # Vertical line across the video
    canvas_video.coords(crosshair_v, x, y-size, x, y+size)

####### UI Layout  ##########

### Bind hotkeys ###
root.bind("<Return>", lambda event: add_annotation())
root.bind("<Left>", lambda event: prev_special_frame())
root.bind("<Right>", lambda event: next_special_frame())

# Advance one frame at time with 
root.bind(".", lambda event: advance_frame(1))
root.bind(",", lambda event: advance_frame(-1))
#Bind the spacebar to play/pause
root.bind("<space>", lambda event: pause())
# Bind the 'o' key to add an exit annotation
root.bind("o", lambda event: add_exit())

# Bind the 'i' key to add an annotation
root.bind("i", lambda event: add_entry())

# Top level container layout
frame_main=ctk.CTkFrame(root)
frame_main.pack(side=TOP, fill=BOTH, expand=True)
### Controls on the Left Panel #### 

frame_controls = ctk.CTkFrame(frame_main)
frame_controls.pack(side=LEFT, fill=Y, padx=5, pady=5)

button_browse = ctk.CTkButton(frame_controls, text="Browse Video", command=load_video, height = 20)
button_browse.pack(pady=10)

Label(frame_controls, text="SAM2 Start Frame:").pack(pady=5)
special_frame_start_var = IntVar(value=0)
special_frame_entry = ttk.Entry(frame_controls, textvariable=special_frame_start_var)
special_frame_entry.pack(pady=5)

def update_special_frame_start():
    global special_frame_start
    special_frame_start = special_frame_start_var.get()

button_set_special_frame = ctk.CTkButton(frame_controls, text="Set SAM2 Frame", command=update_special_frame_start, height = 20)
button_set_special_frame.pack(pady=10)

button_toggle_click = ctk.CTkButton(frame_controls, text="Positive Click", command=toggle_click_type, height = 20)
button_toggle_click.pack(pady=5)

button_toggle_obj_type = ctk.CTkButton(
    frame_controls,
    text="Parrotfish",
    command=toggle_obj_type, height = 20
)
button_toggle_obj_type.pack(pady=5)

Label(frame_controls, text="Fish Name:").pack(pady=5)
fish_name = StringVar()
entry_fish_name = ttk.Entry(frame_controls, textvariable=fish_name)
entry_fish_name.pack(pady=5)


button_add_annotation = ctk.CTkButton(frame_controls, text="Add Annotation ('Return')", command=add_annotation, height = 20)
button_add_annotation.pack(pady=5)

button_add_entry = ctk.CTkButton(frame_controls, text="Add Entry ('i')", command=add_entry, height = 20)
button_add_entry.pack(pady=5)

button_add_exit = ctk.CTkButton(frame_controls, text="Add Exit ('o')", command=add_exit, height = 20)
button_add_exit.pack(pady=5)

button_edit_selected = ctk.CTkButton(frame_controls, text="Edit Selected", command=edit_selected, height = 20)
button_edit_selected.pack(pady=10)

button_delete_selected = ctk.CTkButton(frame_controls, text="Delete Selected", command=delete_selected, height = 20)
button_delete_selected.pack(pady=10)

button_delete_all = ctk.CTkButton(frame_controls, text="Delete All", command=delete_all, height = 20)
button_delete_all.pack(pady=5)

button_import = ctk.CTkButton(frame_controls, text="Import Previous Annotations", command=import_annotations, height = 20)
button_import.pack(pady=5)

Label(frame_controls, text="Saving File Name:").pack(pady=5)
file_name_var = StringVar()
entry_file_name = ttk.Entry(frame_controls, textvariable=file_name_var)
entry_file_name.pack(pady=5)

# Checkboxes for saving options
save_bites_var = BooleanVar(value=False)
save_locations_var = BooleanVar(value=True)
save_all_var = BooleanVar(value=False)

checkbox_bites = ctk.CTkCheckBox(frame_controls, text="Save Bites File", variable=save_bites_var)
checkbox_bites.pack()

checkbox_locations = ctk.CTkCheckBox(frame_controls, text="Save Locations File", variable=save_locations_var)
checkbox_locations.pack()

checkbox_all = ctk.CTkCheckBox(frame_controls, text="Save Eels", variable=save_all_var)
checkbox_all.pack()

button_save_annotations = ctk.CTkButton(frame_controls, text="Save Annotations", command=save_annotations, height = 20)
button_save_annotations.pack(pady=5)


#### Top Central Playback controls ####
frame_center = ctk.CTkFrame(frame_main)
frame_center.pack(side=LEFT, fill=BOTH, expand=True, padx=5,pady=5)
# Playback controls at top of center frame
frame_central_controls = ctk.CTkFrame(frame_center)
frame_central_controls.pack(side=TOP, fill=X, padx=10, pady=10)

button_play_pause = ctk.CTkButton(frame_central_controls, text="Pause ||", command=pause, width = 40)
button_play_pause.pack(pady=5, side=LEFT)

button_prev_special_frame = ctk.CTkButton(frame_central_controls, text="Prev SAM2 Frame", command=prev_special_frame)
button_prev_special_frame.pack(side=LEFT, padx=5, pady=5)

button_prev_frame = ctk.CTkButton(frame_central_controls, text="<< Prev Frame", command=lambda: advance_frame(-1), width = 40)
button_prev_frame.pack(pady=5, side=LEFT)

button_next_frame = ctk.CTkButton(frame_central_controls, text="Next Frame >>", command=lambda: advance_frame(1), width = 40)
button_next_frame.pack(pady=5, side=LEFT)


button_next_special_frame = ctk.CTkButton(frame_central_controls, text="Next SAM2 Frame", command=next_special_frame)
button_next_special_frame.pack(side=LEFT, padx=5, pady=5)

button_decrease_speed = ctk.CTkButton(frame_central_controls, text="- Speed", command=lambda: adjust_speed(-0.1), width = 40)
button_decrease_speed.pack(pady=5, side=LEFT)

button_increase_speed = ctk.CTkButton(frame_central_controls, text="+ Speed", command=lambda: adjust_speed(0.1), width = 40)
button_increase_speed.pack(pady=5, side=LEFT)

button_reset_speed = ctk.CTkButton(frame_central_controls, text="Reset Speed", command=reset_speed, width = 40)
button_reset_speed.pack(pady=5, side=LEFT)


# Central Video Player: 
frame_video = ctk.CTkFrame(frame_center)
frame_video.pack(side=TOP, fill=BOTH, expand=True)

frame_bottom_controls = ctk.CTkFrame(frame_center)
frame_bottom_controls.pack(side=BOTTOM, fill=X)

# Add scrollbars to pan
scrollbar_y = ctk.CTkScrollbar(frame_video, orientation="vertical")
scrollbar_y.pack(side="right", fill = "y")

scrollbar_x = ctk.CTkScrollbar(frame_video, orientation="horizontal")
scrollbar_x.pack(side="bottom", fill = "x")

# Create video player
canvas_video = Canvas(frame_video, bg="black", highlightthickness=0)  
canvas_video.pack(side=LEFT, fill=BOTH, expand=True)

# Link scrollbars to canvas 
canvas_video.configure(yscrollcommand = scrollbar_y.set, xscrollcommand=scrollbar_x.set)
scrollbar_y.configure(command=canvas_video.yview)
scrollbar_x.configure(command=canvas_video.xview)

# Zooming scrollbar
slider_zoom = ctk.CTkSlider(frame_video, from_=300, to=3000,number_of_steps=100, command=on_slider_change, orientation="vertical")
slider_zoom.set(video_size_x)
slider_zoom.pack(side="bottom", fill = "y", padx=5,pady=5)


# Bind the click event
canvas_video.bind('<Button-1>', canvas_click_events)
# Bind cursor change events
canvas_video.bind("<Enter>", on_mouse_enter)
canvas_video.bind("<Leave>", on_mouse_leave)

# Bind mouse wheel
# For Windows and Linux
canvas_video.bind("<MouseWheel>", on_mouse_wheel)
# For MacOS
canvas_video.bind("<Button-4>", on_mouse_wheel)
canvas_video.bind("<Button-5>", on_mouse_wheel)

# Bind aswd keys to canvas
canvas_video.bind_all("w", pan_canvas)
canvas_video.bind_all("s", pan_canvas)
canvas_video.bind_all("a", pan_canvas)
canvas_video.bind_all("d", pan_canvas)

# Create crosshair lines (hidden initially)
crosshair_h = canvas_video.create_line(0, 0, 0, 0, fill="red", width=1)
crosshair_v = canvas_video.create_line(0, 0, 0, 0, fill="red", width=1)

# Bind motion to update the crosshair
canvas_video.bind("<Motion>", update_crosshair)

special_frame_var = StringVar()
special_frame_var.set("----")
label_special_frame = Label(frame_bottom_controls, textvariable=special_frame_var, font=("Arial", 12))
label_special_frame.pack(pady=5)

time_display_var = StringVar()
time_display_var.set("Time: 0.00s | Frame: 0 | Speed: 1.0x")
label_time_display = Label(frame_bottom_controls, textvariable=time_display_var, font=("Arial", 12))
label_time_display.pack(pady=5)

slider_frame = ttk.Scale(frame_bottom_controls, from_=0, to=0, orient=HORIZONTAL, length=video_size_x)
slider_frame.pack(pady=10)
slider_frame.bind("<ButtonRelease-1>", update_frame_from_slider)

# Annotations on the right
frame_annotations = ctk.CTkFrame(frame_main)
frame_annotations.pack(side=LEFT, fill=Y, padx=5, pady=5)


columns = ("ID", "Frame", "Click Type", "Fish Label", "ObjType", "Coordinates")
treeview = ttk.Treeview(frame_annotations, columns=columns, show="headings")
for col in columns:
    treeview.heading(col, text=col)
    treeview.column(col, width=130)  # Adjust column width for readability

treeview.pack(fill=BOTH, expand=True)
treeview.bind("<Double-1>", on_double_click)
root.mainloop()
