import matplotlib.pyplot as plt
import subprocess
import os
import glob

class vis():
    def __init__(self, images_path, video_path, frame_skip):
        self.frame_skip = frame_skip
        self.images_path = images_path
        self.video_path = video_path
        # Keeps track of internal image ID
        self.image_idx = 0
        self.create_directories_if_not_exist(images_path, video_path)

    def save_image(self, image, step_number):
        if(step_number % self.frame_skip == 0):
            img_id = "{:04d}".format(self.image_idx)
            im_path = os.path.join(self.images_path, 'img'+img_id+'.png')
            plt.imsave(im_path, image)
            self.image_idx += 1

    def generate_video(self, episode_number):
        vid_path = os.path.join(self.video_path, "log_{}.mp4".format(episode_number))
        im_path = os.path.join(self.images_path, "img%04d.png")
        gen_vid_command = ["ffmpeg", "-y", "-i", im_path ,"-c:v", "libx264", "-r", "25", "-pix_fmt", "yuv420p",
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
