import matplotlib.pyplot as plt
import numpy as np
import subprocess
from PIL import Image, ImageFont, ImageDraw
import os
import glob

class Visualizer:
    def __init__(self, images_path, video_path, frame_skip=1, videos=True):
        self.frame_skip = frame_skip
        self.images_path = images_path
        self.video_path = video_path
        # Keeps track of internal image ID
        self.image_idx = 0
        if videos:
            self.create_directories_if_not_exist(images_path, video_path)

    def save_image(self, image, step_number):
        if(step_number % self.frame_skip == 0):
            self.image_idx += 1
            img_id = "{:08d}".format(self.image_idx)
            im_path = os.path.join(self.images_path, img_id+'.png')
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

    def save_pil_image(self, image, step_number, step_info):
        image = self.convert_image(image)
        image = self.modify_image(image, step_info)
        if(step_number % self.frame_skip == 0):
            self.image_idx += 1
            img_id = "{:08d}".format(self.image_idx)
            im_path = os.path.join(self.images_path, img_id+'.png')
            image.save(im_path)

    # @profile
    def generate_video(self, episode_number, total_steps, index):
        file_name = 'Ep_' + str(episode_number) + '_step_' + str(total_steps) + "_ind_" + str(index) + '.mp4'
        vid_path = os.path.join(self.video_path, file_name)
        im_path = os.path.join(self.images_path, "%08d.png")
        gen_vid_command = ["ffmpeg", "-y", "-i", im_path ,"-c:v", "libx264", "-r", "30", "-pix_fmt", "yuv420p",
        vid_path]
        gen_vid_process = subprocess.Popen(gen_vid_command, preexec_fn=os.setsid, stdout=open(os.devnull, "w"))
        gen_vid_process.wait()

    # @profile
    def remove_images(self):
        # rm_img_command = ["rm", "-f", "{}/*.png".format(self.images_path)]
        # rm_img_process = subprocess.Popen(rm_img_command, preexec_fn=os.setsid, stdout=open(os.devnull, "w"))
        #TODO: Faster way remove images?
        images = glob.glob("{}/*.png".format(self.images_path))
        for image in images:
            os.remove(image)
        # Reset image idx (ffmpeg starts from index 0)? Bug where there was no video generation past episode 1
        self.image_idx = 0

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
        plt.savefig(path + '{}.png'.format(episode_num))
        plt.close()


