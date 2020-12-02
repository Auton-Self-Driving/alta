import os
import numpy as np
import matplotlib.pyplot as plt

def save_for_detection(
    timestep, dist_to_light, nearest_traffic_actor_state, 
    semantic_image, rgb_image, semantic_path='./semantic_images', rgb_path='./rgb_images'):
    """
    Plugin for saving images for traffic light detection

    Args: 
        timestep: 
            an integer or current timestep
        dist_to_light: 
            a float number, same as obs['dist_to_light']
        nearest_traffic_actor_state: 
            a string, same as obs['nearest_traffic_actor_state']
        semantic_image:
            an H x W shaped semantic sensor output, each value is an integer specifying its class
        rgb_image:
            an H x W x 3 shaped rgb sensor output      
        semantic_path: 
            folder for saving semantic images
        rgb_path: 
            folder for saving RGB images
    """
    user_semantic_path = os.path.expanduser(semantic_path)
    user_rgb_path = os.path.expanduser(rgb_path)

    if not os.path.exists(user_semantic_path):
        print('creating {} ...'.format(user_semantic_path))
        os.makedirs(user_semantic_path)
    if not os.path.exists(user_rgb_path):
        print('creating {} ...'.format(user_rgb_path))
        os.makedirs(user_rgb_path)

    # add str() to everything in case it is None
    save_name = str(timestep) + '_' + str(dist_to_light) + '_' + str(nearest_traffic_actor_state)

    np.save(os.path.join(user_semantic_path, save_name), semantic_image)
    plt.imsave(os.path.join(user_rgb_path, save_name) + '.jpg', rgb_image)

    return save_name


if __name__ == '__main__':
    dummy_semantic = np.random.randint(1, 23, size=(512,512))
    dummy_rgb = np.random.randint(0, 256, size=(512,512, 3)).astype(np.uint8)

    save_for_detection(123, 3.141592654123, 'Green', dummy_semantic, dummy_rgb,
        semantic_path='~/Desktop/savetest/semantic_images', rgb_path='~/Desktop/savetest/rgb_images')

