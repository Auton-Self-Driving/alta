import matplotlib.pyplot as plt
import numpy as np
import subprocess
from PIL import Image, ImageFont, ImageDraw
import os
import glob
import time
from collections import defaultdict
import cv2

class Visualizer:
    def __init__(self, images_path, video_path, videos=True):
        self.images_path = images_path
        self.video_path = video_path
        self.time = lambda: time.strftime('%Y-%m-%d %H:%M:%S')
        self.savetime = lambda: time.strftime('%b%d%I%M%p%S')
        # Keeps track of internal image ID
        self.image_idx = defaultdict(int)
        os.makedirs(self.images_path, exist_ok=True)
        if videos and not os.path.exists(self.video_path):
            os.makedirs(self.video_path, exist_ok=True)

    def save_image(self, image, sub_folder=''):
        self.image_idx[sub_folder] += 1
        _path = os.path.join(self.images_path, str(sub_folder))
        if not os.path.exists(_path):
            os.mkdir(_path)
        img_id = "{:08d}".format(self.image_idx[sub_folder])
        im_path = os.path.join(_path, img_id + '.png')
        plt.imsave(im_path, image)

    def save_semantic_image(self, image, step_number):
        im_path = os.path.join(self.images_path, str(step_number))
        np.save(im_path, image)

    def convert_image(self, image):
        return Image.fromarray(image, 'RGB').convert('RGBA')

    def modify_image(self, image, step_info):
        overlay = Image.new('RGBA', image.size, (255,255,255,0))
        draw_overlay = ImageDraw.Draw(overlay)
        draw_overlay.text((10, 10),
                          "Sp={:0.2f}, TSp={:0.2f}\nTh={:0.2f}, St={:0.2f}\nBr={:0.2f}, Or={:0.2f}\nd={:0.2f}\nObs_d={:0.2f}\nObs_s={:0.2f}\nTL={:0.2f}".format(
                              step_info['speed'] * 3.6,
                              step_info['target_speed'],
                              step_info['control_throttle'],
                              step_info['control_steer'],
                              step_info['control_brake'],
                              step_info['next_orientation'],
                              step_info['dist_to_trajectory'],
                              step_info['obstacle_dist'],
                              step_info['obstacle_speed'],
                              step_info['red_light_dist']),
                          fill=(255,255,255,128))

        return Image.alpha_composite(image, overlay)

    def save_pil_image(self, image, step_number, step_info, sub_folder=''):
        _path = os.path.join(self.images_path, str(sub_folder))
        if not os.path.exists(_path):
            os.mkdir(_path)
        image = self.convert_image(image)
        image = self.modify_image(image, step_info)
        self.image_idx[sub_folder] += 1
        img_id = "{:08d}".format(self.image_idx[sub_folder])
        im_path = os.path.join(_path, img_id+'.png')
        image.save(im_path)

    # @profile
    def generate_video(self, sub_folder='', suffix=''):
        vid_prefix = 'video' if not sub_folder else sub_folder
        vid_suffix = '' if not suffix else '_' + str(suffix)

        file_name = str(vid_prefix) + '_' + vid_suffix + '.mp4'
        vid_path = os.path.join(self.video_path, file_name)
        image_names = os.listdir(os.path.join(self.images_path, sub_folder))
        
        frame_template = cv2.imread(os.path.join(self.images_path, sub_folder,image_names[0]))
        out = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (frame_template.shape[1],frame_template.shape[0]))
        for im_p in image_names:
            if ".png" not in im_p:
                continue
            frame = cv2.imread(os.path.join(self.images_path, sub_folder,im_p))
            out.write(frame)
        out.release()
        cv2.destroyAllWindows()

    # @profile
    def create_combined_image(self, sub_folders):
        """
        Creates a combined video from multiple cameras
        """

        folder_name = str(sub_folders[0].split("_")[0]) 
        out_path = os.path.join(self.images_path, folder_name)
        os.makedirs(out_path,exist_ok=True)

        for im_p in os.listdir(os.path.join(self.images_path, sub_folders[0])):
            frames = []
            for sub_folder in sub_folders: 
                im_path = os.path.join(self.images_path, sub_folder,im_p)
                frames.append(cv2.imread(im_path))
            out_frame = np.concatenate(frames, axis=1)
            cv2.imwrite(os.path.join(out_path,im_p),out_frame)

        return folder_name

    # @profile
    def remove_images(self, sub_folder=''):
        # rm_img_command = ["rm", "-f", "{}/*.png".format(self.images_path)]
        # rm_img_process = subprocess.Popen(rm_img_command, preexec_fn=os.setsid, stdout=open(os.devnull, "w"))
        #TODO: Faster way remove images?
        _path = os.path.join(self.images_path, str(sub_folder))
        images = glob.glob("{}/*.png".format(_path))
        for image in images:
            os.remove(image)
        anything_else = glob.glob("{}/*.*".format(_path))
        if not anything_else: os.rmdir(_path)
        # Reset image idx (ffmpeg starts from index 0)? Bug where there was no video generation past episode 1
        self.image_idx[sub_folder] = 0

    def create_directories_if_not_exist(self,*directories):
        for d in directories:
            if d is not None:
                if not os.path.exists(d):
                    os.makedirs(d)

    def plot_episode_info(path,
                    target_speeds_array,
                    speeds_array,
                    throttles_array,
                    steers_array,
                    brakes_array,
                    obstacle_dist_array,
                    step_reward_array,
                    collision_reward_array,
                    dist_to_trajectory_reward_array,
                    red_light_dist_array,
                    episode_num):

        # Override path
        path = self.video_path

        if not os.path.exists(path):
            os.makedirs(path)

        observations = np.arange(len(target_speeds_array))

        target_speeds_array = np.array(target_speeds_array)
        speeds_array = np.array(speeds_array)
        throttles_array = np.array(throttles_array)
        steers_array = np.array(steers_array)
        brakes_array = np.array(brakes_array)
        step_reward_array = np.array(step_reward_array)
        collision_reward_array = np.array(collision_reward_array)
        obstacle_dist_array = np.array(obstacle_dist_array)
        dist_to_trajectory_reward_array = np.array(dist_to_trajectory_reward_array)
        red_light_dist_array = np.array(red_light_dist_array)

        fig, axs = plt.subplots(5, 2, figsize=(12, 12))
        fig.suptitle('Episode info plots for episode idx {} '.format(episode_num))

        axs[0, 0].plot(observations, target_speeds_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
        axs[0, 0].set_xlabel('Timesteps')
        axs[0, 0].set_ylabel('Target Speed - Stochastic')

        axs[1, 0].plot(observations, speeds_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
        axs[1, 0].set_xlabel('Timesteps')
        axs[1, 0].set_ylabel('Actual Speed - Stochastic')


        axs[2, 0].plot(observations, throttles_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
        axs[2, 0].set_xlabel('Timesteps')
        axs[2, 0].set_ylabel('Throttle')

        axs[3, 0].plot(observations, step_reward_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
        axs[3, 0].set_xlabel('Timesteps')
        axs[3, 0].set_ylabel('Step reward')

        axs[4, 0].plot(observations, dist_to_trajectory_reward_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
        axs[4, 0].set_xlabel('Timesteps')
        axs[4, 0].set_ylabel('dist_to_trajectory reward')


        axs[0, 1].plot(observations, steers_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
        axs[0, 1].set_xlabel('Timesteps')
        axs[0, 1].set_ylabel('Steer - Stochastic')


        axs[1, 1].plot(observations, obstacle_dist_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
        axs[1, 1].set_xlabel('Timesteps')
        axs[1, 1].set_ylabel('Obstacle Distance')

        axs[2, 1].plot(observations, brakes_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
        axs[2, 1].set_xlabel('Timesteps')
        # axs[2, 1].set_ylabel('Break')
        axs[2, 1].set_ylabel('Orientation')

        axs[3, 1].plot(observations, collision_reward_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
        axs[3, 1].set_xlabel('Timesteps')
        axs[3, 1].set_ylabel('collision_reward')

        axs[4, 1].plot(observations, red_light_dist_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
        axs[4, 1].set_xlabel('Timesteps')
        axs[4, 1].set_ylabel('Dist to red light')

        axs[0,0].grid(True)
        axs[0,1].grid(True)
        axs[1,0].grid(True)
        axs[1,1].grid(True)
        axs[2,0].grid(True)
        axs[2,1].grid(True)
        axs[3,0].grid(True)
        axs[3,1].grid(True)
        axs[4,0].grid(True)
        axs[4,1].grid(True)

        plt.grid(True)
        plt.savefig(os.path.join(path,'{}.png'.format(episode_num)))
        plt.close()

    def plot_episode_info_2(self,path,
                    target_speeds_array,
                    speeds_array,
                    throttles_array,
                    steers_array,
                    brakes_array,
                    obstacle_dist_array,
                    step_reward_array,
                    collision_reward_array,
                    dist_to_trajectory_reward_array,
                    red_light_dist_array,
                    episode_num):

        # Override path
        path = self.video_path

        if not os.path.exists(path):
            os.makedirs(path)

        stats = {
            "target_speeds":target_speeds_array,
            "speeds":speeds_array,
            "throttles":throttles_array,
            "steers":steers_array,
            "brakes":brakes_array,
            "obstacle_dist":obstacle_dist_array,
            "step_reward":step_reward_array,
            "collision_reward":collision_reward_array,
            "dist_to_traj_reward":dist_to_trajectory_reward_array,
            "red_light_dist_reward":red_light_dist_array,
        }

        np.save(os.path.join(path,'{}_stats.npy'.format(episode_num)), stats) 
        # read_dictionary = np.load('my_file.npy',allow_pickle='TRUE').item() # TO READ

        observations = np.arange(len(target_speeds_array))

        target_speeds_array = np.array(target_speeds_array)
        speeds_array = np.array(speeds_array)
        throttles_array = np.array(throttles_array)
        steers_array = np.array(steers_array)
        brakes_array = np.array(brakes_array)
        step_reward_array = np.array(step_reward_array)
        collision_reward_array = np.array(collision_reward_array)
        obstacle_dist_array = np.array(obstacle_dist_array)
        dist_to_trajectory_reward_array = np.array(dist_to_trajectory_reward_array)
        red_light_dist_array = np.array(red_light_dist_array)

        fig, axs = plt.subplots(6, 2, figsize=(14, 14))
        fig.suptitle('Episode info plots for episode idx {} '.format(episode_num))

        axs[0, 0].plot(observations, target_speeds_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
        axs[0, 0].set_xlabel('Timesteps')
        axs[0, 0].set_ylabel('Target Speed - Stochastic')

        axs[1, 0].plot(observations, speeds_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
        axs[1, 0].set_xlabel('Timesteps')
        axs[1, 0].set_ylabel('Actual Speed - Stochastic')


        axs[2, 0].plot(observations, throttles_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
        axs[2, 0].set_xlabel('Timesteps')
        axs[2, 0].set_ylabel('Throttle')

        axs[3, 0].plot(observations, step_reward_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
        axs[3, 0].set_xlabel('Timesteps')
        axs[3, 0].set_ylabel('Step reward')

        axs[4, 0].plot(observations, dist_to_trajectory_reward_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
        axs[4, 0].set_xlabel('Timesteps')
        axs[4, 0].set_ylabel('dist_to_trajectory reward')


        axs[0, 1].plot(observations, steers_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
        axs[0, 1].set_xlabel('Timesteps')
        axs[0, 1].set_ylabel('Steer - Stochastic')


        axs[1, 1].plot(observations, obstacle_dist_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
        axs[1, 1].set_xlabel('Timesteps')
        axs[1, 1].set_ylabel('Obstacle Distance')

        axs[2, 1].plot(observations, brakes_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
        axs[2, 1].set_xlabel('Timesteps')
        # axs[2, 1].set_ylabel('Break')
        axs[2, 1].set_ylabel('Orientation')

        axs[3, 1].plot(observations, collision_reward_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
        axs[3, 1].set_xlabel('Timesteps')
        axs[3, 1].set_ylabel('collision_reward')

        axs[4, 1].plot(observations, red_light_dist_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
        axs[4, 1].set_xlabel('Timesteps')
        axs[4, 1].set_ylabel('Dist to red light')

        axs[5, 0].hist(throttles_array, density=True, bins=300)
        axs[5, 0].set_xlabel('Throttle')
        axs[5, 0].set_ylabel('Probability')

        axs[5, 1].hist(steers_array, density=True, bins=300)
        axs[5, 1].set_xlabel('Steering ')
        axs[5, 1].set_ylabel('Probability')

        axs[0,0].grid(True)
        axs[0,1].grid(True)
        axs[1,0].grid(True)
        axs[1,1].grid(True)
        axs[2,0].grid(True)
        axs[2,1].grid(True)
        axs[3,0].grid(True)
        axs[3,1].grid(True)
        axs[4,0].grid(True)
        axs[4,1].grid(True)
        axs[5,0].grid(True)
        axs[5,1].grid(True)

        plt.grid(True)
        plt.savefig(os.path.join(path,'{}.png'.format(episode_num)))
        plt.close()


