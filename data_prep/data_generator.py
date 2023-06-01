import carla
import math
import random
import time
import queue
import numpy as np
import cv2
import os

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(loc, K, w2c):
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    point_img = np.dot(K, point_camera)

    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2]

client = carla.Client('localhost', 2000)
world = client.get_world()
bp_lib = world.get_blueprint_library()

spawn_points = world.get_map().get_spawn_points()
vehicle_bp = bp_lib.find('vehicle.audi.etron')
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

camera_bp = bp_lib.find('sensor.camera.rgb')
camera_init_trans = carla.Transform(carla.Location(x=10.5, y=0.0, z=0.7))
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

depth_camera_bp = bp_lib.find('sensor.camera.depth')
depth_camera_init_trans = carla.Transform(carla.Location(x=10.5, y=0.0, z=0.7))
depth_camera = world.spawn_actor(depth_camera_bp, depth_camera_init_trans, attach_to=vehicle)

seg_camera_bp = bp_lib.find('sensor.camera.semantic_segmentation')
seg_camera_init_trans = carla.Transform(carla.Location(x=10.5, y=0.0, z=0.7))
seg_camera = world.spawn_actor(seg_camera_bp, seg_camera_init_trans, attach_to=vehicle)

vehicle.set_autopilot(True)

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)


image_queue = queue.Queue()
camera.listen(image_queue.put)

depth_image_queue = queue.Queue()
depth_camera.listen(depth_image_queue.put)

seg_queue = queue.Queue()
seg_camera.listen(seg_queue.put)

world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()
fov = camera_bp.get_attribute("fov").as_float()
K = build_projection_matrix(image_w, image_h, fov)

# Set up the set of bounding boxes from the level
# We filter for traffic lights and traffic signs
#bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
#bounding_box_set.extend(world.get_level_bbs(carla.CityObjectLabel.TrafficSigns))

# Remember the edge pairs
edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

for i in range(10):
    vehicle_bp = random.choice(bp_lib.filter('vehicle'))
    npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    if npc:
        npc.set_autopilot(True)

    pedestrian_bp = random.choice(bp_lib.filter('pedestrian'))
    ped = world.try_spawn_actor(pedestrian_bp, random.choice(spawn_points))
    #if ped:
    #    ped.set_autopilot(True)


# Retrieve the first image
world.tick()
image = image_queue.get()
depth = depth_image_queue.get()
seg = seg_queue.get()

# Reshape the raw data into an RGB array
img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)) 
dmg = np.reshape(np.copy(depth.raw_data), (depth.height, depth.width, 4))
seg.convert(carla.ColorConverter.CityScapesPalette)
smg = np.reshape(np.copy(seg.raw_data), (seg.height, seg.width, 4))

# Display the image in an OpenCV display window
cv2.namedWindow('ImageWindowName', cv2.WINDOW_AUTOSIZE)
cv2.imshow('ImageWindowName',img)
cv2.waitKey(1)

cv2.namedWindow('DepthWindowName', cv2.WINDOW_AUTOSIZE)
cv2.imshow('DepthWindowName', dmg)
cv2.waitKey(1)

cv2.namedWindow('Seg', cv2.WINDOW_AUTOSIZE)
cv2.imshow('Seg', smg)
cv2.waitKey(1)

scenarios = [
    (carla.WeatherParameters.ClearNoon, 'ClearNoon'),
    (carla.WeatherParameters.CloudyNoon, 'CloudyNoon'),
    (carla.WeatherParameters.WetNoon, 'WetNoon'),
    (carla.WeatherParameters.WetCloudyNoon, 'WetCloudyNoon'),
    (carla.WeatherParameters.MidRainyNoon, 'MidRainyNoon'),
    (carla.WeatherParameters.HardRainNoon, 'HardRainNoon'),
    (carla.WeatherParameters.SoftRainNoon, 'SoftRainNoon'),
    (carla.WeatherParameters.ClearSunset, 'ClearSunset'),
    (carla.WeatherParameters.CloudySunset, 'CloudySunset'),
    (carla.WeatherParameters.WetSunset, 'WetSunset'),
    (carla.WeatherParameters.MidRainSunset, 'MidRainSunset'),
    (carla.WeatherParameters.HardRainSunset, 'HardRainSunset'),
    (carla.WeatherParameters.SoftRainSunset, 'SoftRainSunset'),
]

for scenario in scenarios:
    os.mkdir(scenario[1])
    os.mkdir(os.path.join(scenario[1], 'rgb'))
    os.mkdir(os.path.join(scenario[1], 'depth'))
    os.mkdir(os.path.join(scenario[1], 'gt'))

for scenario in scenarios:

    #weather = carla.WeatherParameters(scenario[0])

    world.set_weather(scenario[0])
    print(world.get_weather())
    count = 0

    for i in range(2000):
        # Retrieve and reshape the image
        world.tick()
        image = image_queue.get()
        depth = depth_image_queue.get()
        seg = seg_queue.get()

        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        dmg = np.reshape(np.copy(depth.raw_data), (depth.height, depth.width, 4))
        seg.convert(carla.ColorConverter.CityScapesPalette)
        smg = np.reshape(np.copy(seg.raw_data), (seg.height, seg.width, 4))

        cv2.imwrite(os.path.join(scenario[1], 'rgb', f'{count}.png'), img)
        cv2.imwrite(os.path.join(scenario[1], 'depth', f'{count}.png'), dmg)
        cv2.imwrite(os.path.join(scenario[1], 'gt', f'{count}.png'), smg)
        count += 1


        # Get the camera matrix 
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())


        for npc in world.get_actors().filter('*vehicle*'):

            # Filter out the ego vehicle
            if npc.id != vehicle.id:

                bb = npc.bounding_box
                dist = npc.get_transform().location.distance(vehicle.get_transform().location)

                # Filter for the vehicles within 50m
                if dist < 50:

                # Calculate the dot product between the forward vector
                # of the vehicle and the vector between the vehicle
                # and the other vehicle. We threshold this dot product
                # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
                    forward_vec = vehicle.get_transform().get_forward_vector()
                    ray = npc.get_transform().location - vehicle.get_transform().location

                    if forward_vec.dot(ray) > 1:
                        p1 = get_image_point(bb.location, K, world_2_camera)
                        verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                        for edge in edges:
                            p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                            p2 = get_image_point(verts[edge[1]],  K, world_2_camera)
                            cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (255,0,0, 255), 1)        

        cv2.imshow('ImageWindowName',img)
        if cv2.waitKey(1) == ord('q'):
            break
        cv2.imshow('DepthWindowName',dmg)
        if cv2.waitKey(1) == ord('q'):
            break
        cv2.imshow('Seg', smg)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
