import matplotlib.pyplot as plt
import subprocess
from PIL import Image, ImageFont, ImageDraw
import os
import glob

class vis():
    def __init__(self, images_path, video_path, frame_skip, videos=True):
        self.frame_skip = frame_skip
        self.images_path = images_path
        self.video_path = video_path
        # Keeps track of internal image ID
        self.image_idx = 0
        if videos:
            self.create_directories_if_not_exist(images_path, video_path)

    def save_image(self, image, step_number):
        if(step_number % self.frame_skip == 0):
            img_id = "{:04d}".format(self.image_idx)
            im_path = os.path.join(self.images_path, 'img'+img_id+'.png')
            plt.imsave(im_path, image)
            self.image_idx += 1
    
    def save_semantic_image(self, image, step_number):
        im_path = os.path.join(self.images_path, str(step_number))
        np.save(im_path, image)

    def convert_image(self, image):
        return Image.fromarray(image, 'RGB').convert('RGBA')
    
    def modify_image(self, image, step_info):
        overlay = Image.new('RGBA', image.size, (255,255,255,0))
        draw_overlay = ImageDraw.Draw(overlay)
        draw_overlay.text((10, 10), 
                          "Sp={:0.3f}\nTSp={:0.3f}\nTh={:0.3f}\nSt={:0.3f}\nBr={:0.3f}\nd={:0.3f}\nObs_d={:0.3f}\nObs_s={:0.3f}".format(
                              step_info['speed'] * 3.6, 
                              step_info['target_speed'],
                              step_info['control_throttle'], 
                              step_info['control_steer'], 
                              step_info['control_brake'], 
                              step_info['dist_to_trajectory'],
                              step_info['obstacle_dist'],
                              step_info['obstacle_speed']), 
                          fill=(255,255,255,128))
        
        return Image.alpha_composite(image, overlay)

    def save_pil_image(self, image, step_number, step_info):
        image = self.convert_image(image)
        image = self.modify_image(image, step_info)
        if(step_number % self.frame_skip == 0):
            img_id = "{:04d}".format(self.image_idx)
            im_path = os.path.join(self.images_path, 'img'+img_id+'.png')
            image.save(im_path)
            self.image_idx += 1

    def generate_video(self, episode_number, total_steps, index):
        file_name = 'E_' + str(episode_number) + '_t_' + str(total_steps) + "_i_" + str(index) + '.mp4'
        vid_path = os.path.join(self.video_path, file_name)
        im_path = os.path.join(self.images_path, "img%04d.png")
        gen_vid_command = ["ffmpeg", "-y", "-i", im_path ,"-c:v", "libx264", "-r", "30", "-pix_fmt", "yuv420p",
        vid_path]
        gen_vid_process = subprocess.Popen(gen_vid_command, preexec_fn=os.setsid, stdout=open(os.devnull, "w"))
        gen_vid_process.wait()

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
            if not os.path.exists(d):
                os.makedirs(d)
