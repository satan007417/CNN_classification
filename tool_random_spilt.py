

import os
import random
import shutil
from sklearn.model_selection import train_test_split
from utils.scan_files import scan_dirs_folder, scan_files_folder, scan_files_subfolder

input('enter any key to start ...')

'''
spilt data from src_folder to spilt_folder
'''
src_folder = r'D:\BackendAOI\Data\PassFail_Images\4HW1_4HW2\1'
spilt_folder = r'D:\BackendAOI\Data\PassFail_Images\4HW1_4HW2\1_test'
src_keep_ratio = 0.5
spilt_keep_ratio = 1 - src_keep_ratio
search_exts = ['jpg','jpeg', 'bmp', 'png']

for root, dirs, files in os.walk(src_folder):
    # create dir
    new_root = root.replace(src_folder, spilt_folder)
    os.makedirs(new_root, exist_ok=True)
    # filtered extensions
    files = [f for f in files if any(f.endswith(ext) for ext in search_exts)]
    # spilt
    if len(files) > 0:
        files_src, files_spilt = train_test_split(files, test_size=spilt_keep_ratio, random_state=42)
        for f in files_spilt:
            shutil.move(f'{root}\\{f}', f'{new_root}\\{f}')
                