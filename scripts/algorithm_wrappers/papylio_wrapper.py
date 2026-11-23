import papylio as pp
import xarray as xr
import numpy as np
from tifffile import imread, imwrite
import os
import yaml
import sys
import time
from pathlib import Path


####VARIABLES###


input_path = Path(sys.argv[1]).resolve()
output_det_path = Path(sys.argv[2]).resolve()
output_sel_path = Path(sys.argv[3]).resolve()

config_path = sys.argv[4]

with open(config_path, 'r') as f:
    full_config = yaml.safe_load(f)

p = full_config["algorithms"]["papylio"]["params"]

max_rolling_donor_intensity = p["max_rolling_donor_intensity"]
min_mean_donor_intensity = p["min_mean_donor_intensity"]
fraction_difference = p["fraction_difference"]
neighbourhood_size_min = p["neighbourhood_size_min"]
neighbourhood_size_max = p["neighbourhood_size_max"]
projection_max_frame = p["projection_max_frame"]
edge_margin = p["edge_margin"]
gaussian_fit_width = p["gaussian_fit_width"]
rolling_window_size = p["rolling_window_size"]


####FOLDER CLEANING AND PATH SETUP###


#Get the path of the current script
script_location = Path(__file__).resolve()
project_root = script_location.parents[2]  # Go up 2 levels
papylio_dir = project_root / "algorithms" / "papylio"
papylio_dir.mkdir(parents=True, exist_ok=True) #Make folder if it doesnt exist
path_DataSimCopy = papylio_dir / "DataSim_copy.tiff"

nc_path = str(path_DataSimCopy.with_suffix(".nc"))
ave_path = str(papylio_dir / f"{path_DataSimCopy.stem}_ave_f0-500_i0.tif")
path_DataSimCopy_str = str(path_DataSimCopy) 
log_path = str(papylio_dir / "DataSim_copy.log")
corrections_path = str(papylio_dir / "DataSim_copy_corrections.nc")  # new: written by the spatial background correction

#Cleaning old files
for p_file in [path_DataSimCopy_str, nc_path, ave_path, log_path, corrections_path]: #now includes background correction too
    if os.path.exists(p_file):
        os.remove(p_file)

# waiting a bit
print("Waiting for file cleanup in Papylio programme folder...")
time.sleep(2)


####PAPYLIO DETECTION####


# Copy input Tiff to Papylio folder
print(f"Reading input: {input_path}")
video = imread(input_path)
imwrite(path_DataSimCopy_str, video[:, 0])  # donor channel only (T, Y, X)
print(f"Saved copy to: {path_DataSimCopy_str}")

#Start Papylio Experiment
exp = pp.Experiment(
    main_path=str(papylio_dir), 
    channels=["donor"],  
    import_all=True
)
if not exp.files:
    raise FileNotFoundError("Papylio did not find the Tiff file copy.")

file = exp.files[0] 

#Apply channel settings 
file.movie.channel_arrangement = [[[0]]]
file.movie.channels = file.movie.channels[0:1]
#Spatial background correction, as in Example.py, which is a median filter (30 px) of the
#average of the first 20 frames is subtracted from every frame, before both detection and trace extraction.
file.movie.determine_spatial_background_correction()

# Find Coordinates
print("Finding coordinates...")
file.find_coordinates(
    channels=["donor"],
    method="by_channel",
    projection_image=dict(
        projection_type="average",
        frame_range=(0, projection_max_frame), #default is first 20 frames (because photobleaching) but sim doesn't do that, so use more frames for best SNR
        illumination=0 #all frames get ALEX illumination index 0 assigned upon loading into Papylio so this selects all frames
    ),
    peak_finding=dict(
        method="local-maximum-auto",
        fraction_difference=fraction_difference, 
        filter_neighbourhood_size_min = neighbourhood_size_min,
        filter_neighbourhood_size_max = neighbourhood_size_max
    ),
    coordinate_optimization=dict(
        coordinates_within_margin=dict(margin=edge_margin), #Same effect as StarDist edge margin
        coordinates_after_gaussian_fit=dict(gaussian_width=gaussian_fit_width)
    )
)

# Get detections
data = xr.load_dataset(nc_path)  #load_dataset reads it into memory and closes the file, so Papylio can still write to it
if "coordinates" not in data:  #Papylio writes no coordinates when it finds 0 molecules
    np.save(output_det_path, np.empty((0, 2)))
    np.save(output_sel_path, np.empty((0, 2)))
    sys.exit(0)

coords = data["coordinates"]
x = coords[:, 0, 0].values
y = coords[:, 0, 1].values
detected_points = np.column_stack((y, x))


###PAPYLIO SELECTION###


file.extract_traces()

#Rolling Average MAX intensity selection: using rolling average because intense brightness at any point is aggregate behaviour
intensity_total_rolling = file.intensity_total.rolling(frame=rolling_window_size).mean().dropna('frame')#dropna is important because the rolling mean will produce NaN values for the first few frames where the window isn't full yet
#^array of rolling mean of total intensity for each molecule over time
selection = (intensity_total_rolling < max_rolling_donor_intensity).all('frame')
file.set_variable(selection, name='selection_intensity_total')

# (Not Rolling!) Average MIN intensity selection 
#Updated this code to be extra transparent about being a whole trace mean minimum intensity, like StarDist, instead of the documented final 5 frames, as the simulator doesn't do photobleaching.
intensity_donor_mean = file.intensity.sel(channel=0).mean('frame')
selection_min_intensity = min_mean_donor_intensity < intensity_donor_mean
file.set_variable(selection_min_intensity, name='selection_intensity_donor')

#Combine masks
combined_selection = selection & selection_min_intensity
file.set_variable(combined_selection, name='selection_combined')

# Get Selected Points
selected = file.coordinates[combined_selection.values]
x = selected[:, 0, 0].values
y = selected[:, 0, 1].values
selected_points = np.column_stack((y, x))
print(f"Papylio Detected: {len(detected_points)}")
print(f"Papylio Selected: {len(selected_points)}")


###SAVING###


print(f"Papylio directory before reset: {os.getcwd()}")
os.chdir(project_root)
print(f"Papylio directory reset to: {os.getcwd()}, the project root.")

#Ensure output directories exist
output_det_path.parent.mkdir(parents=True, exist_ok=True)
output_sel_path.parent.mkdir(parents=True, exist_ok=True)

np.save(output_det_path, detected_points)
np.save(output_sel_path, selected_points)
print(f"Successfully saved to {output_det_path}")