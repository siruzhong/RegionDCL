import argparse
import math
import os
import pickle as pkl
import rasterio.features
import geopandas as gpd
import numpy as np
from scipy.spatial import KDTree
from shapely import affinity
from shapely.geometry import Point
from shapely.ops import unary_union
from tqdm import tqdm, trange

from grid import Grid

"""
    This class implements common preprocessing tasks for NYC and Singapore data.
    The preprocessing includes the following steps:
    - Convert POI (points of interest) data and building types into one-hot vectors.
    - Attach POI data to building data based on their spatial relationships.
    - Perform Poisson-disk sampling on the boundary map to generate non-overlapping points.
    - Calculate density and location encodings for each sampled point.
    - Group the buildings and POIs according to spatial patterns.
    - Group the patterns into regions for further analysis.
"""

class Preprocess(object):
    def __init__(self, city):
        # Define input and output paths for the specified city
        in_path = 'data/projected/{}/'.format(city)
        out_path = 'data/processed/{}/'.format(city)
        self.in_path = in_path
        self.out_path = out_path
        
        # Create the output directory if it does not exist
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        # Define file paths for buildings, POI, segmentation, and boundaries
        self.building_in_path = in_path + 'building/building.shp'
        self.poi_in_path = in_path + 'poi/poi.shp'
        self.building_out_path = out_path + 'building.pkl'
        self.poi_out_path = out_path + 'poi.pkl'
        self.segmentation_in_path = in_path + 'segmentation/segmentation.shp'
        self.segmentation_out_path = out_path + 'segmentation'
        self.boundary_in_path = in_path + 'boundary/' + city + '.shp'

        # Load city boundary from the shapefile
        print('Loading boundary from {}'.format(self.boundary_in_path))
        boundary_shapefile = gpd.read_file(self.boundary_in_path)
        
        # If there are multiple polygons, merge them into one boundary using unary_union
        boundary = [boundary_row['geometry'] for index, boundary_row in boundary_shapefile.iterrows()]
        if len(boundary) > 1:
            boundary = unary_union(boundary)
        else:
            boundary = boundary[0]
        self.boundary = boundary

    def get_building_and_poi(self, force=False):
        """
        This function processes buildings and POIs:
        1. Load the building and POI data from Shapefiles.
        2. Convert building types and POI codes/classes into one-hot vectors.
        3. Attach nearby POIs to buildings using spatial relationships.
        4. Save the processed data into pickle files for future use.
        """
        # Check if the building and POI data has already been processed and saved
        if not force and os.path.exists(self.building_out_path):
            print('Loading building from {}'.format(self.building_out_path))
            with open(self.building_out_path, 'rb') as f:
                building = pkl.load(f)
            print('Loading poi from {}'.format(self.poi_out_path))
            with open(self.poi_out_path, 'rb') as f:
                poi = pkl.load(f)
            return building, poi
        
        # Load building and POI data from shapefiles
        print('Preprocessing building and poi data...')
        buildings_shapefile = gpd.read_file(self.building_in_path)
        pois_shapefile = gpd.read_file(self.poi_in_path)
        
        # Convert shapefiles into lists of dictionaries containing shape and attribute data
        building = []
        poi = []
        for index, building_row in tqdm(buildings_shapefile.iterrows(), total=buildings_shapefile.shape[0]):
            output = {}
            # Store polygon geometry and building type
            shape = building_row['geometry']
            output['shape'] = shape
            output['type'] = building_row['type']
            building.append(output)
        
        for index, poi_row in tqdm(pois_shapefile.iterrows(), total=pois_shapefile.shape[0]):
            output = {}
            # Store POI coordinates, code, and functional class
            output['x'] = poi_row['geometry'].x
            output['y'] = poi_row['geometry'].y
            output['code'] = poi_row['code']
            output['fclass'] = poi_row['fclass']
            poi.append(output)
        
        # Convert building types and POI codes/classes into one-hot encoded vectors
        print('Turning building type and poi code/fclass into one-hot...')
        building_type = set([b['type'] for b in building])
        poi_code = set([p['code'] for p in poi])
        poi_fclass = set([p['fclass'] for p in poi])

        building_type_dict = {t: i for i, t in enumerate(building_type)}
        poi_code_dict = {c: i for i, c in enumerate(poi_code)}
        poi_fclass_dict = {f: i for i, f in enumerate(poi_fclass)}

        # Assign one-hot vectors to buildings based on type
        for b in building:
            b['onehot'] = [0] * len(building_type)
            b['onehot'][building_type_dict[b['type']]] = 1
        
        # Assign one-hot vectors to POIs based on code and functional class
        poi_dim = len(poi_code) + len(poi_fclass)
        for p in poi:
            p['onehot'] = [0] * poi_dim
            p['onehot'][poi_code_dict[p['code']]] = 1
            p['onehot'][len(poi_code) + poi_fclass_dict[p['fclass']]] = 1

        # Attach POIs to buildings by checking proximity
        print('Attaching pois to buildings...')
        poi_x = [p['x'] for p in poi]
        poi_y = [p['y'] for p in poi]
        poi_tree = KDTree(np.array([poi_x, poi_y]).T)
        
        attached_poi = []
        batch_size = 50000  # Processes 50,000 buildings in batches, adjustable based on memory availability
        total_buildings = len(building)
        
        # Open the file and prepare to save the results in batches
        with open(self.building_out_path, 'wb') as building_file:
            for batch_start in tqdm(range(0, total_buildings, batch_size)):
                batch_buildings = building[batch_start:batch_start + batch_size]
                
                for b in tqdm(batch_buildings):
                    # Sum up all the POIs within the building's bounding box
                    b['poi'] = [0] * poi_dim
                    bounds = b['shape'].bounds
                    cx = (bounds[0] + bounds[2]) / 2
                    cy = (bounds[1] + bounds[3]) / 2
                    height = bounds[3] - bounds[1]
                    width = bounds[2] - bounds[0]
                    radius = np.sqrt(height ** 2 + width ** 2) / 2

                    # Find all POIs within the calculated radius
                    poi_index = poi_tree.query_ball_point([cx, cy], radius)
                    for i in poi_index:
                        if not b['shape'].contains(Point(poi[i]['x'], poi[i]['y'])):
                            continue
                        b['poi'] = [b['poi'][j] + poi[i]['onehot'][j] for j in range(poi_dim)]
                        attached_poi.append(poi[i])
                
                # Save each batch of building data to a file and write it batch by batch
                pkl.dump(batch_buildings, building_file, protocol=4)
                       
                # Clean up unnecessary memory
                del batch_buildings
                import gc
                gc.collect()

        # Store the POIs that were not attached to any buildings
        poi_not_attached = [p for p in poi if p not in attached_poi]
        print('Saving poi data...')
        with open(self.poi_out_path, 'wb') as f:
            pkl.dump(poi_not_attached, f, protocol=4)

        return building, poi_not_attached

    def poisson_disk_sampling(self, building_list, poi_list, radius, force=False):
        """
        Perform Poisson-disk sampling on the city's boundary map.
        The goal is to generate random points that are evenly distributed, with no overlapping.
        """
        random_point_out_path = self.out_path + 'random_point_' + str(radius) + 'm.pkl'
        
        # Check if the sampling has already been done and saved
        if not force and os.path.exists(random_point_out_path):
            with open(random_point_out_path, 'rb') as f:
                result = pkl.load(f)
            return result
        
        # Use the Grid class to perform Poisson-disk sampling
        grid = Grid(self.boundary, radius, building_list, poi_list)
        result = grid.poisson_disk_sampling()

        # Save the sampled points to a pickle file
        with open(random_point_out_path, 'wb') as f:
            pkl.dump(result, f, protocol=4)

        return result

    def partition(self, building_list, poi_list, random_point_list, radius, force=False):
        """
        Partition the city data based on a segmentation map (e.g., road network or administrative boundaries).
        Buildings, POIs, and random points are grouped into spatial patterns based on these regions.
        """
        # Define output path for partitioned data
        partition_out_path = self.segmentation_out_path + f'_{radius}.pkl'

        # If the partitioning has already been done and saved, load it from disk
        if not force and os.path.exists(partition_out_path):
            with open(partition_out_path, 'rb') as f:
                result = pkl.load(f)
            return result

        print('Partitioning city data by region (e.g., road network or administrative boundaries)...')

        # Load segmentation polygons (e.g., from road network or administrative boundaries)
        segmentation_shapefile = gpd.read_file(self.segmentation_in_path)
        segmentation_polygon_list = [row.geometry for index, row in segmentation_shapefile.iterrows()]

        result = []
        
        # Get the centroid locations of buildings, POIs, and random points
        building_loc = [[b['shape'].centroid.x, b['shape'].centroid.y] for b in building_list]
        poi_loc = [[p['x'], p['y']] for p in poi_list]
        random_point_loc = random_point_list

        # Create KDTree structures for fast spatial querying of buildings, POIs, and random points
        building_tree = KDTree(building_loc)
        poi_tree = KDTree(poi_loc)
        random_point_tree = KDTree(random_point_loc)

        # Iterate through each region defined in the segmentation shapefile
        for i in trange(len(segmentation_polygon_list)):
            shape = segmentation_polygon_list[i]  # Get the polygon representing the current region
            
            # Initialize a pattern for this region, which will contain buildings, POIs, and random points
            pattern = {
                'shape': shape,
                'building': [],     # List of buildings in this region
                'poi': [],          # List of POIs in this region
                'random_point': []  # List of random points in this region
            }

            # Calculate the diameter of the polygon's bounding box to define a search radius
            bounds = shape.bounds
            dx = bounds[2] - bounds[0]  # Width of the bounding box
            dy = bounds[3] - bounds[1]  # Height of the bounding box
            diameter = math.sqrt(dx * dx + dy * dy) / 2

            # Find all buildings within the search radius of the region's centroid
            building_index = building_tree.query_ball_point([shape.centroid.x, shape.centroid.y], diameter)
            for j in building_index:
                # Check if the building actually intersects with the region polygon
                if shape.intersects(building_list[j]['shape']):
                    pattern['building'].append(j)

            # Find all POIs within the search radius of the region's centroid
            poi_index = poi_tree.query_ball_point([shape.centroid.x, shape.centroid.y], diameter)
            for j in poi_index:
                # Check if the POI is contained within the region polygon
                if shape.contains(Point(poi_loc[j][0], poi_loc[j][1])):
                    pattern['poi'].append(j)

            # Find all random points within the search radius of the region's centroid
            random_point_index = random_point_tree.query_ball_point([shape.centroid.x, shape.centroid.y], diameter)
            for j in random_point_index:
                # Check if the random point is contained within the region polygon
                if shape.contains(Point(random_point_loc[j][0], random_point_loc[j][1])):
                    pattern['random_point'].append(j)

            # Skip this pattern if it contains no buildings (regions with no buildings are not of interest)
            if len(pattern['building']) == 0:
                continue

            result.append(pattern)  # Append the pattern for this region to the result list

        # Save the partitioned result (patterns of buildings, POIs, and random points by region) to a pickle file
        with open(partition_out_path, 'wb') as f:
            pkl.dump(result, f, protocol=4)

        return result


    def rasterize_buildings(self, building_list, rotation=True, force=False):
        """
        Rasterize buildings: convert building polygons into 224x224 pixel images.
        Optionally rotate the buildings to align them with the x-axis for better consistency.
        """
        image_out_path = self.out_path + 'building_raster.npz'
        rotation_out_path = self.out_path + 'building_rotation.npz'

        # If rasterized data already exists, load it from disk
        if not force and os.path.exists(image_out_path):
            return np.load(image_out_path)['arr_0']

        print('Rasterizing buildings...')
        images = np.zeros((len(building_list), 224, 224), dtype=np.uint8)
        rotations = np.zeros((len(building_list), 2), dtype=float)

        # Loop over all buildings and rasterize each one
        for i in trange(len(building_list)):
            polygon = building_list[i]['shape']

            if rotation:
                # Rotate the building to align it with the x-axis
                rectangle = polygon.minimum_rotated_rectangle
                xc = polygon.centroid.x
                yc = polygon.centroid.y

                rec_x = []
                rec_y = []
                for point in rectangle.exterior.coords:
                    rec_x.append(point[0])
                    rec_y.append(point[1])

                top = np.argmax(rec_y)
                top_left = top - 1 if top > 0 else 3
                top_right = top + 1 if top < 3 else 0

                x0, y0 = rec_x[top], rec_y[top]
                x1, y1 = rec_x[top_left], rec_y[top_left]
                x2, y2 = rec_x[top_right], rec_y[top_right]

                d1 = np.linalg.norm([x0 - x1, y0 - y1])
                d2 = np.linalg.norm([x0 - x2, y0 - y2])

                if d1 > d2:
                    cosp = (x1 - x0) / d1
                    sinp = (y0 - y1) / d1
                else:
                    cosp = (x2 - x0) / d2
                    sinp = (y0 - y2) / d2

                # Save the rotation matrix
                rotations[i] = [cosp, sinp]
                
                # Apply the rotation transformation to align the building with the x-axis
                matrix = (cosp, -sinp, 0.0,
                          sinp, cosp, 0.0,
                          0.0, 0.0, 1.0,
                          xc - xc * cosp + yc * sinp, yc - xc * sinp - yc * cosp, 0.0)
                polygon = affinity.affine_transform(polygon, matrix)

            # Get the bounding box of the rotated polygon
            min_x, min_y, max_x, max_y = polygon.bounds
            length_x = max_x - min_x
            length_y = max_y - min_y

            # Ensure the bounding box is square by adjusting width and height
            if length_x > length_y:
                min_y -= (length_x - length_y) / 2
                max_y += (length_x - length_y) / 2
            else:
                min_x -= (length_y - length_x) / 2
                max_x += (length_y - length_x) / 2

            # Enlarge the bounding box by 20% to avoid edge clipping
            length = max(length_x, length_y)
            min_x -= length * 0.1
            min_y -= length * 0.1
            max_x += length * 0.1
            max_y += length * 0.1

            # Compute transformation from the polygon bounding box to a 224x224 pixel image
            transform = rasterio.transform.from_bounds(min_x, min_y, max_x, max_y, 224, 224)

            # Rasterize the polygon and save the image to the images array
            image = rasterio.features.rasterize([polygon], out_shape=(224, 224), transform=transform)
            images[i] = image

        # Save the rasterized images and rotation matrices as compressed npz files
        np.savez_compressed(image_out_path, images)
        np.savez_compressed(rotation_out_path, rotations)

def parse_args():
    """
    Parse command-line arguments to specify the city and radius for Poisson disk sampling.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Singapore', help='city name, can be Singapore or NYC')
    parser.add_argument('--radius', type=float, default=100, help='radius of the Poisson Disk Sampling')
    return parser.parse_args()

if __name__ == '__main__':
    # Parse the command-line arguments
    args = parse_args()
    city = args.city
    radius = args.radius
    
    # Ensure the radius is sufficiently large to avoid too many sample points
    assert radius > 50, "Radius is too small, too many points will slow down the process."
    
    # Instantiate the Preprocess class and start processing
    preprocessor = Preprocess(city)
    
    # Load or process buildings and POIs
    building, poi = preprocessor.get_building_and_poi()

    # Perform Poisson disk sampling to generate non-overlapping random points
    random_point = preprocessor.poisson_disk_sampling(building, poi, radius)

    # Rasterize building geometries into images
    preprocessor.rasterize_buildings(building)

    # Partition the city data by regions (e.g., based on the road network)
    preprocessor.partition(building, poi, random_point, radius)

    # Output the number of random points generated
    print(f'Random Points: {len(random_point)}')
