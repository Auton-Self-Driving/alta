import matplotlib.pyplot as plt
import subprocess
import os

class vis:
    def __init__(self, images_path, video_path, frame_skip):
        self.frame_skip = frame_skip
        self.images_path = images_path
        self.video_path = video_path
        # Keeps track of internal image ID
        self.image_idx = 0
    
    def save_image(self, image, step_number):
        if(step_number % self.frame_skip == 0):
            img_id = "{:04d}".format(self.image_idx)
            plt.imsave(str(self.images_path)+'img_'+img_id+'.png', image)
            self.image_idx += 1 

    
    def generate_video(self, episode_number):
        gen_vid_command = ["ffmpeg -i img_%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p", 
        str(self.video_path)+"log_{}.mp4".format(episode_number)]
        gen_vid_process = subprocess.Popen(gen_vid_command, preexec_fn=os.setsid, stdout=open(os.devnull, "w"))
        # Add function to remove images from directory
        rm_img_command = ["rm -rf", "{}/*.png".format(self.images_path)]
        rm_img_process = subprocess.Popen(rm_img_command, preexec_fn=os.setsid, stdout=open(os.devnull, "w"))
        
