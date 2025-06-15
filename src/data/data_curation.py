# src/data/data_curation.py

import os
import leafmap
import geopandas as gpd
from tqdm import tqdm

def fetch_high_res_imagery_for_tiles(
    metadata_geojson_path: str,
    output_dir: str,
    zoom_level: int = 18
) -> None:
    """
    Fetches high-resolution Google Maps Satellite imagery for each geometry
    defined in a GeoJSON file.

    This function reads a GeoJSON file containing metadata for low-resolution
    satellite tiles (e.g., from Planet), and for each tile, it downloads a
    corresponding high-resolution GMS image.

    Args:
        metadata_geojson_path (str):
            Path to the GeoJSON file containing tile geometries. Each feature's
            properties should ideally include a unique identifier for the tile.
        output_dir (str):
            Directory where the downloaded high-resolution GeoTIFF images will be saved.
        zoom_level (int, optional):
            The zoom level for Google Maps tiles. Higher values result in
            higher resolution. Defaults to 18.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    gdf = gpd.read_file(metadata_geojson_path)
    # Ensure CRS is WGS 84 (lat/lon) for leafmap
    gdf = gdf.to_crs("EPSG:4326")

    print(f"Found {len(gdf)} tiles to process from {metadata_geojson_path}.")

    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0], desc="Downloading GMS Tiles"):
        # Use a unique identifier from the GeoJSON, e.g., combining x and y tile indices
        # This assumes your GeoJSON has 'x' and 'y' properties. Adapt if necessary.
        try:
            tile_name = f"{row['x']}_{row['y']}"
        except KeyError:
            tile_name = f"tile_{index}"

        output_filename = f"{tile_name}.tif"
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            # print(f"Skipping existing tile: {output_filename}")
            continue

        # Get the bounding box for the tile geometry
        bounds = row['geometry'].bounds
        bbox = [bounds[0], bounds[1], bounds[2], bounds[3]]  # [minx, miny, maxx, maxy]

        try:
            leafmap.map_tiles_to_geotiff(
                output_path,
                bbox,
                zoom=zoom_level,
                source='Google Satellite',
                overwrite=True
            )
        except Exception as e:
            print(f"Could not download tile {tile_name}. Error: {e}")

    print("Data curation process complete.")