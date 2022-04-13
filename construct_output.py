import os
import sys
import numpy as np
import cv2

# function to construct output videos
def construct_output(results_dir, name):
    frm_dir = os.path.join(results_dir, name, 'test_latest', 'images')
    vid_dir = os.path.join(results_dir, name, 'test_latest', 'videos')
    if not os.path.exists(vid_dir):
        os.makedirs(vid_dir)

    # make video writer objects for each type
    video_writers = {}
    vid_types = ['real_A', 'real_B', 'fake_A', 'fake_B', 'rec_A', 'rec_B']
    prefix = sorted(os.listdir(frm_dir))[0].split('_')[0]
    for vid_type in vid_types:
        video_writers[vid_type] = cv2.VideoWriter(os.path.join(vid_dir, f'{prefix}_{vid_type}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (256, 256))

    # write frames to output videos
    for frm_name in sorted(os.listdir(frm_dir)):
        for vid_type in vid_types:
            if vid_type in frm_name and frm_name[:len(prefix)] == prefix:
                frm = cv2.imread(os.path.join(frm_dir, frm_name))
                video_writers[vid_type].write(frm)

    # release files
    for vid_type in vid_types:
        video_writers[vid_type].release()

# run script
if __name__ ==  '__main__':
    results_dir = sys.argv[1]
    name = sys.argv[2]
    construct_output(results_dir, name)
