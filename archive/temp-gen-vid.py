import agents.tf.vis_module as vism

images_path = '/home/shubhand/carla-images'
video_path = '/home/shubhand/carla-images/video'

frame_skip = 1
vism_obj = vism.vis(images_path, video_path, frame_skip)
vism_obj.generate_video(0)
