import os
import cv2
import json
import numpy as np
import pyproj
import exifread
import folium
from sklearn.cluster import DBSCAN

# ---------- CONFIGURATION ----------
coco_json_path = "sample image/train/_annotations.coco.json"  # COCO annotations
filtered_image_folder = "sample image/original_images"  # Folder with images
output_folder = "marked_images"  # Folder for visualization
json_output_path = "output_world_coordinates.json"  # JSON file output
map_output_path = "output_map.html"  # 2D map output
os.makedirs(output_folder, exist_ok=True)

# Load COCO JSON file
with open(coco_json_path, 'r') as f:
    coco_data = json.load(f)

# Get image metadata
image_info = {img["id"]: img for img in coco_data["images"]}

# Define coordinate transformation (WGS84 to ENU)
wgs84 = pyproj.CRS("EPSG:4326")  # Latitude, Longitude
ecef = pyproj.CRS("EPSG:4978")  # Earth-Centered, Earth-Fixed
transformer_to_ecef = pyproj.Transformer.from_crs(wgs84, ecef, always_xy=True)
transformer_to_wgs84 = pyproj.Transformer.from_crs(ecef, wgs84, always_xy=True)


# ---------- FUNCTION: Extract Cleaned Filename ----------
def get_clean_filename(file_name):
    """ Extracts base filename from the COCO JSON formatted filename. """
    return file_name.split("_JPG")[0]


# Map cleaned filenames to actual image paths
image_file_map = {}
for real_file in os.listdir(filtered_image_folder):
    if real_file.lower().endswith(".jpg"):
        clean_name = get_clean_filename(real_file)
        image_file_map[clean_name] = os.path.join(
            filtered_image_folder, real_file)


# ---------- FUNCTION: Extract Drone Metadata ----------
def get_exif_metadata(image_path):
    """Extracts GPS coordinates, altitude, focal length, yaw, pitch, roll from EXIF metadata."""
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f, details=False)

    def dms_to_degrees(dms):
        """Convert EXIF GPS DMS to decimal degrees."""
        return float(dms[0].num) / float(dms[0].den) + \
            float(dms[1].num) / (60 * float(dms[1].den)) + \
            float(dms[2].num) / (3600 * float(dms[2].den))

    # Extract GPS
    lat = dms_to_degrees(
        tags['GPS GPSLatitude'].values) if 'GPS GPSLatitude' in tags else None
    lon = dms_to_degrees(
        tags['GPS GPSLongitude'].values) if 'GPS GPSLongitude' in tags else None
    if 'GPS GPSLatitudeRef' in tags and tags['GPS GPSLatitudeRef'].values == 'S':
        lat = -lat
    if 'GPS GPSLongitudeRef' in tags and tags['GPS GPSLongitudeRef'].values == 'W':
        lon = -lon
    alt = float(tags['GPS GPSAltitude'].values[0].num) / float(
        tags['GPS GPSAltitude'].values[0].den) if 'GPS GPSAltitude' in tags else None

    # Extract Focal Length
    focal_length = float(tags['EXIF FocalLength'].values[0].num) / \
        float(
            tags['EXIF FocalLength'].values[0].den) if 'EXIF FocalLength' in tags else 24.0

    # Extract Yaw, Pitch, Roll (defaults to 0 if missing)
    yaw = float(tags.get("FlightYawDegree", [0])[0])
    pitch = float(tags.get("FlightPitchDegree", [0])[0])
    roll = float(tags.get("FlightRollDegree", [0])[0])

    return {
        "lat": lat, "lon": lon, "alt": alt or 0,
        "focal_length": focal_length, "yaw": yaw, "pitch": pitch, "roll": roll
    }


# ---------- FUNCTION: Apply Rotation Matrix ----------
def apply_rotation(yaw, pitch, roll, point):
    """Apply 3D rotation matrix to transform camera-relative coordinates."""
    yaw, pitch, roll = np.radians([yaw, pitch, roll])

    # Rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])

    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                   [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])

    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll), np.cos(roll), 0],
                   [0, 0, 1]])

    return Rz @ Ry @ Rx @ point


# ---------- FUNCTION: Convert Pixel → World Coordinates ----------
def pixel_to_world(image_metadata, pixel_coords, image_shape):
    """Convert pixel coordinates to real-world coordinates using the Pinhole Camera Model."""
    if image_metadata["focal_length"] is None:
        return []  # Skip images with missing focal length

    fx = fy = image_metadata["focal_length"] * \
        image_shape[1]  # Focal length in pixels
    cx, cy = image_shape[1] / 2, image_shape[0] / 2  # Camera center

    real_world_coords = []
    for px, py in pixel_coords:
        x_cam = (px - cx) / fx
        y_cam = (py - cy) / fy
        z_cam = 1  # Assume ground-plane intersection

        cam_space_point = np.array([x_cam, y_cam, z_cam])
        world_point = apply_rotation(
            image_metadata["yaw"], image_metadata["pitch"], image_metadata["roll"], cam_space_point)

        # Convert to GPS
        lat_offset = world_point[1] * 0.00001
        lon_offset = world_point[0] * 0.00001
        real_world_coords.append(
            (image_metadata["lat"] + lat_offset, image_metadata["lon"] + lon_offset))

    return real_world_coords


def remove_duplicate_trees(tree_positions, eps=0.00005, min_samples=1):
    """Apply DBSCAN clustering to remove duplicate tree detections."""
    if not tree_positions:
        return []

    tree_positions = np.array(tree_positions)
    clustering = DBSCAN(eps=eps, min_samples=min_samples,
                        metric='euclidean').fit(tree_positions)

    unique_positions = []
    for label in set(clustering.labels_):
        if label != -1:  # Ignore noise points
            cluster_points = tree_positions[clustering.labels_ == label]
            # Get mean position of the cluster
            cluster_center = np.mean(cluster_points, axis=0)
            unique_positions.append(tuple(cluster_center))

    return unique_positions


# ---------- PROCESS EACH IMAGE ----------
tree_positions_by_category = {}

for annotation in coco_data["annotations"]:
    image_id = annotation["image_id"]
    category_id = annotation["category_id"]
    segmentation = annotation["segmentation"]

    if image_id not in image_info:
        continue

    coco_file_name = image_info[image_id]["file_name"]
    clean_name = get_clean_filename(coco_file_name) + ".JPG"

    if clean_name not in image_file_map:
        continue

    image_path = image_file_map[clean_name]
    image = cv2.imread(image_path)

    if image is None:
        continue

    metadata = get_exif_metadata(image_path)

    # Skip if GPS coordinates are missing
    if metadata["lat"] is None or metadata["lon"] is None:
        continue

    detected_positions = []
    for polygon in segmentation:
        points = np.array(polygon, dtype=np.int32).reshape(-1, 2)
        if len(points) > 0:
            # Get center of detected tree
            detected_positions.append(np.mean(points, axis=0))

    # Convert to world coordinates using yaw, pitch, roll
    world_positions = pixel_to_world(metadata, detected_positions, image.shape)

    # Store positions by category
    if category_id not in tree_positions_by_category:
        tree_positions_by_category[category_id] = []
    tree_positions_by_category[category_id].extend(world_positions)

# ---------- APPLY DBSCAN TO REMOVE DUPLICATES ----------
for category, positions in tree_positions_by_category.items():
    tree_positions_by_category[category] = remove_duplicate_trees(positions)

# Save tree positions
with open(json_output_path, "w") as f:
    json.dump(tree_positions_by_category, f, indent=4)

print(f"✅ Tree positions saved to {json_output_path}")

# ---------- PLOT RESULTS ON MAP ----------
if any(tree_positions_by_category.values()):
    all_positions = [pos for positions in tree_positions_by_category.values()
                     for pos in positions]
    center_lat = np.mean([pos[0] for pos in all_positions])
    center_lon = np.mean([pos[1] for pos in all_positions])

    tree_map = folium.Map(location=[center_lat, center_lon], zoom_start=20)

    for category, positions in tree_positions_by_category.items():
        for lat, lon in positions:
            folium.CircleMarker(
                location=[lat, lon],
                radius=5, color='red', fill=True, fill_color='red'
            ).add_to(tree_map)

    tree_map.save(map_output_path)
    print(f"✅ Map saved as {map_output_path}")
