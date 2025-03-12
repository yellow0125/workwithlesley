import math
import re

def parse_wkt_point(wkt):
    """
    Expects 'SRID=4326;POINT(lon lat)' or 'POINT(lon lat)'
    Returns (lat, lon) or (None, None) if parsing fails.
    """
    match = re.search(r"POINT\(\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*\)", wkt)
    if not match:
        return None, None
    lon = float(match.group(1))
    lat = float(match.group(2))
    return (lat, lon)

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Approx distance (km) between two lat/lon coords using WGS84.
    """
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(d_lon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

    
# Heading refers to the compass direction an aircraft (or camera) is facing, measured in degrees from North (0° or 360°) clockwise.
# 0° / 360° ⇒ Facing North
# 90° ⇒ Facing East
# 180° ⇒ Facing South
# 270° ⇒ Facing West
#By normalizing this angle (e.g., heading % 360), we ensure it’s always in the range [0..360).
def heading_to_cardinal(heading):
    """
    0=North, 90=East, 180=South, 270=West in 90-degree segments.
    """
    angle = heading % 360
    if angle >= 315 or angle < 45:
        return "N"
    elif angle < 135:
        return "E"
    elif angle < 225:
        return "S"
    else:
        return "W"


def retrieve_images_from_csv(
    csv_path,
    query_coords=None,
    max_distance_km=2.0,
    min_pitch=-90,
    max_pitch=0,
    start_date=None,
    end_date=None,
    required_directions=None
):
    """
    Reads CSV with columns including:
      image_coords (WKT),
      gimbal_pitch,
      heading,
      time_of_capture (optional),
      ...
    Filters by distance, pitch, date, direction, etc.
    Returns a list of dicts.
    """
    results = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # parse pitch & heading
            try:
                pitch = float(row.get("gimbal_pitch", "0"))
            except ValueError:
                pitch = 0.0
            try:
                heading = float(row.get("heading", "0"))
            except ValueError:
                heading = 0.0

            # parse time if you need date filtering
            time_str = row.get("time_of_capture", "")
            try:
                dt = datetime.datetime.fromisoformat(time_str.replace(" ", "T"))
                dt = dt.replace(tzinfo=None)
            except ValueError:
                dt = None

            # parse coords
            wkt_str = row.get("image_coords", "")
            lat_img, lon_img = parse_wkt_point(wkt_str)

            # pitch range
            if not (min_pitch <= pitch <= max_pitch):
                continue

            # date range
            if dt:
                if start_date and dt < start_date:
                    continue
                if end_date and dt > end_date:
                    continue

            # distance
            if query_coords and lat_img is not None and lon_img is not None:
                dist = haversine_distance(query_coords[0], query_coords[1], lat_img, lon_img)
                if dist > max_distance_km:
                    continue
            else:
                dist = None


            record = {
                "id": row.get("id"),
                "image_coords": wkt_str,
                "gimbal_pitch": pitch,
                "heading": heading,
                "time_of_capture": time_str,
                "lat": lat_img,
                "lon": lon_img,
            }

    return results

def get_nadir_and_four_directions(address_or_coords, csv_file):
    """
    1) parse_or_geocode -> (lat_query, lon_query).
    2) retrieve images for pitch range [-90..-80] => near-nadir
       pick best among them (closest to lat_query, lon_query).
    3) retrieve images for pitch range [-45..-15], group by heading => pick single best from each of N/E/S/W.
    4) return combined results.
    """

    # 1) parse user input
    coords = parse_or_geocode(address_or_coords)
    if not coords:
        print(f"Could not parse or geocode: {address_or_coords}")
        return None, None
    lat_query, lon_query = coords

    # 2) near-nadir => pitch [-90..-80]
    nadir_candidates = retrieve_images_from_csv(
        csv_path=csv_file,
        query_coords=(lat_query, lon_query),
        max_distance_km=2.0,
        min_pitch=-90,
        max_pitch=-80
    )
    best_nadir = None
    if nadir_candidates:
        # pick the physically closest to user's location
        nadir_candidates.sort(key=lambda x: x["distance_km"] or 999999)
        best_nadir = nadir_candidates[0]
        best_nadir["classification"] = "Nadir"

    # 3) oblique => pitch [-45..-15]
    # ignoring "oblique" label, just collecting 4 directions
    oblique_candidates = retrieve_images_from_csv(
        csv_path=csv_file,
        query_coords=(lat_query, lon_query),
        max_distance_km=2.0,
        min_pitch=-45,
        max_pitch=-15
    )

    # group by heading
    direction_buckets = {"N": [], "E": [], "S": [], "W": []}
    for rec in oblique_candidates:
        dir_label = heading_to_cardinal(rec["heading"])
        direction_buckets[dir_label].append(rec)

    four_dirs = {"N": None, "E": None, "S": None, "W": None}
    # pick single nearest in each direction
    for d in direction_buckets:
        bucket = direction_buckets[d]
        if not bucket:
            continue
        bucket.sort(key=lambda x: x["distance_km"] or 999999)
        # pick best
        best_rec = bucket[0]
        best_rec["classification"] = d
        four_dirs[d] = best_rec

    return best_nadir, four_dirs


address_input = "46.83000, -71.28500"  # numeric lat/lon
csv_file = "NE-metadata-sample.csv"

nadir_img, four_dirs = get_nadir_and_four_directions(address_input, csv_file)

print("----- NEAR NADIR -----")
print(nadir_img)

print("\n----- FOUR DIRECTIONS -----")
for d, img in four_dirs.items():
    print(f"{d}: {img}")
