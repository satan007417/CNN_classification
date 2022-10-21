
import os
import random
import shutil
from utils.scan_files import scan_files_subfolder

input('enter any key to start ...')

files = scan_files_subfolder(r'D:\BackendAOI\Photo\Offline\Developing\A73A\4HZ4-2801-TR1C\GT_ADC\GT_P', ['jpg','jpeg', 'bmp', 'png'])
random.shuffle(files)
for f in files[:2000]:
    dir = os.path.dirname(f)
    fn, ext = os.path.splitext(os.path.basename(f))
    txt = f'{dir}\\{fn}.txt'

    target_dir = r'\\10.80.100.190\d\BackendAOI\data\classification\4HZ4\1_chipping\1.Pass'

    shutil.move(f, target_dir + f'\\{fn}{ext}')
    
    # if os.path.exists(txt) and os.path.getsize(txt) == 0:
    #     os.remove(txt)

    # if os.path.exists(txt):
    #     shutil.move(txt, target_dir + f'\\{fn}.txt')
    #     shutil.move(f, target_dir + f'\\{fn}{ext}')

    # if not os.path.exists(txt):
    #     shutil.move(f, target_dir + f'\\{fn}{ext}')

    # if os.path.exists(txt) and os.path.getsize(txt) > 50:
    #     shutil.move(txt, target_dir + f'\\{fn}.txt')
    #     shutil.move(f, target_dir + f'\\{fn}{ext}')