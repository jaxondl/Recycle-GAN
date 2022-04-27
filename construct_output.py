import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

# function to construct output videos
def construct_output(results_dir, name, phase='test', epoch='latest', num_frames=6, interval=30):
    frm_dir = os.path.join(results_dir, name, f'{phase}_{epoch}', 'images')
    vid_dir = os.path.join(results_dir, name, f'{phase}_{epoch}', 'videos')
    if not os.path.exists(vid_dir):
        os.makedirs(vid_dir)
    print(f'Frame directory: {frm_dir}')
    print(f'Video directory: {vid_dir}')

    # load banners
    banner_h = 36
    file_path = os.path.dirname(os.path.realpath(__file__))
    vid_banner = cv2.imread(os.path.join(file_path, 'banners', 'vid_banner.png'))
    vid_banner = cv2.resize(vid_banner, (3*256, banner_h))
    img_banner = cv2.imread(os.path.join(file_path, 'banners', 'img_banner.png'))
    img_banner = cv2.resize(img_banner, (banner_h, 3*256))

    # set up objects
    video_writers = {}
    vid_types = ['real_A', 'real_B', 'fake_A', 'fake_B', 'rec_A', 'rec_B']

    concat_writers = {}
    concat_data = {}
    concat_data['concat_A-B-A'] = {'types': ['real_A', 'fake_B', 'rec_A'],
                                    'cur_frame': np.zeros((banner_h + 256, 3*256, 3), dtype=np.uint8),
                                    'image': np.zeros((3*256, banner_h + num_frames*256, 3), dtype=np.uint8),
                                    'track': 3}
    concat_data['concat_B-A-B'] = {'types': ['real_B', 'fake_A', 'rec_B'],
                                    'cur_frame': np.zeros((banner_h + 256, 3*256, 3), dtype=np.uint8),
                                    'image': np.zeros((3*256, banner_h + num_frames*256, 3), dtype=np.uint8),
                                    'track': 3}
    counters = {}
    for vid_type in vid_types:
        counters[vid_type] = 0

    # set up writers
    prefix = sorted(os.listdir(frm_dir))[0].split('_')[0]
    for vid_type in vid_types:
        video_writers[vid_type] = cv2.VideoWriter(os.path.join(vid_dir, f'{prefix}_{vid_type}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (256, 256))
    for concat_type in concat_data:
        concat_writers[concat_type] = cv2.VideoWriter(os.path.join(vid_dir, f'{prefix}_{concat_type}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (3*256, banner_h + 256))
        concat_data[concat_type]['image'][:, :banner_h, :] = img_banner

    # write frames to output videos
    for i, frm_name in enumerate(sorted(os.listdir(frm_dir))):
        for vid_type in vid_types:
            if vid_type in frm_name: # and frm_name[:len(prefix)] == prefix:
                if i % (25) == 0:
                    print(f'Frame: {frm_name}')
                frm = cv2.imread(os.path.join(frm_dir, frm_name))
                video_writers[vid_type].write(frm)

        for concat_type in concat_data:
            for j, vid_type in enumerate(concat_data[concat_type]['types']):
                if vid_type in frm_name:
                    frm = cv2.imread(os.path.join(frm_dir, frm_name))
                    concat_data[concat_type]['cur_frame'][banner_h:, j*256:(j+1)*256, :] = frm
                    concat_data[concat_type]['track'] -= 1
                    
                    if counters[vid_type] % interval == 0 and counters[vid_type] < interval*num_frames:
                        k = int(counters[vid_type]/interval)
                        concat_data[concat_type]['image'][j*256:(j+1)*256, banner_h+k*256:banner_h+(k+1)*256, :] = frm
                    counters[vid_type] += 1

            if concat_data[concat_type]['track'] == 0:
                concat_data[concat_type]['cur_frame'][:banner_h, :, :] = vid_banner
                concat_writers[concat_type].write(concat_data[concat_type]['cur_frame'])
                concat_data[concat_type]['cur_frame'] = np.zeros((banner_h + 256, 3*256, 3), dtype=np.uint8)
                concat_data[concat_type]['track'] = 3

    # release files
    for vid_type in vid_types:
        video_writers[vid_type].release()
    for concat_type in concat_data:
        concat_writers[concat_type].release()
        cv2.imwrite(os.path.join(vid_dir, f'{prefix}_{concat_type}.png'), concat_data[concat_type]['image'])

# run script
if __name__ ==  '__main__':
    results_dir = sys.argv[1]
    name = sys.argv[2]
    epoch = sys.argv[3]
    construct_output(results_dir, name, phase='test', epoch=epoch)
