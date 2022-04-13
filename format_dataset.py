import os
import sys
import numpy as np
import cv2

# function to center crop image
def center_crop(img):
    h, w, _ = img.shape
    hc = int(h/2)
    wc = int(w/2)
    delta = min(hc, wc)
    return img[hc-delta:hc+delta,wc-delta:wc+delta,:]

# function to format dataset
def format_dataset(vid_dir, save_dir, domain):
    if domain != 'A' and domain != 'B':
        raise ValueError(f'Domain ({domain}) should be A or B')
  
    # make output directories
    train_dir = os.path.join(save_dir, f'train{domain}')
    test_dir = os.path.join(save_dir, f'test{domain}')
    for frm_dir in [train_dir, test_dir]:
        if not os.path.exists(frm_dir):
            os.makedirs(frm_dir)
    print(f'Train directory: {train_dir}')
    print(f'Test directory: {test_dir}')

    # iterate through videos
    vid_names = sorted(os.listdir(vid_dir))
    for vid_name in vid_names:
        if '.mp4' in vid_name:
            print(f'Video: {vid_name}')
            vidcap = cv2.VideoCapture(os.path.join(vid_dir, vid_name))

            # iterate through frames
            count = 0
            frm_0 = None
            frm_1 = None
            success, frm_2 = vidcap.read()
            while success:
                # crop and resize
                frm_2 = center_crop(frm_2)
                frm_2 = cv2.resize(frm_2, (256, 256))

                # save triplet and single frame
                if frm_0 is not None:
                    file_name = f'{int(vid_name.split(".")[0]):03d}_{count:06d}.png'
                    cv2.imwrite(os.path.join(test_dir, file_name), frm_0)

                    triplet = np.concatenate([frm_0, frm_1, frm_2], axis=1)
                    cv2.imwrite(os.path.join(train_dir, file_name), triplet)
                    count += 1

                # next frame
                if frm_1 is not None:
                    frm_0 = frm_1.copy()
                frm_1 = frm_2.copy()
                success, frm_2 = vidcap.read()

# run script
if __name__ ==  '__main__':
    vid_dir = sys.argv[1]
    save_dir = sys.argv[2]
    domain = sys.argv[3]
    format_dataset(vid_dir, save_dir, domain)
