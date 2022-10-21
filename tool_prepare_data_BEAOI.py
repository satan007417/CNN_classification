import os
import glob
import shutil
import json
import random
from sklearn.model_selection import train_test_split
from utils.scan_files import get_path_info, scan_files_subfolder


SRC_ROOT = r'D:\BackendAOI\Photo\Offline\Developing2\UA5A\PK5576-B1-TSM-CC-C\PK5576-B1-TSM-CC-C_PUA5A08N0080001_Bin2_Pic1'
OUTPUT_PASS_ROOT = r'D:\BackendAOI\Data\PassFail_Images\PK5576\2\1.Pass\Bin2'
OUTPUT_FAIL_ROOT = r'D:\BackendAOI\Data\PassFail_Images\PK5576\2\2.Fail\Bin2'
OUTPUT_UNLABELED_ROOT = r'D:\BackendAOI\Data\PassFail_Images\PK5576\2\Unlabeled\Bin2'
SEARCH_EXTS = ['jpg','jpeg', 'bmp', 'png']


def prepare_pass_fail(files):
    for i, file in enumerate(files):
        dir, fn, ext = get_path_info(file) 

        with open(f'{dir}\\{fn}.json', 'r') as f:
            json_obj = json.loads(f.read())

        try:
            label_id, label_name = str(json_obj['Info']['GroundTruth']).split(',')
            if label_name == 'Pass':
                # pass
                shutil.copy(file, f"{OUTPUT_PASS_ROOT}\\{fn}{ext}")
            elif label_name == 'Fail':
                # fail
                shutil.copy(file, f"{OUTPUT_FAIL_ROOT}\\{fn}{ext}")
            
            if i%100 == 0:
                print(f'[{i}] label_id, label_name = {label_id}, {label_name}, {file}')
        except Exception as ex:
            print(ex)


def prepare_pp_ff_pf_fp(files):
    max_count = 20000
    files_pp = []
    files_ff = []
    files_pf = []
    files_fp = []
    for i, file in enumerate(files):
        dir, fn, ext = get_path_info(file)

        with open(f'{dir}\\{fn}.json', 'r') as f:
            json_obj = json.loads(f.read())

        label_id, label_name = str(json_obj['Info']['GroundTruth']).split(',')
        pred_name = str(json_obj['Results'][0]['Judge']['State']) if len(json_obj['Results']) > 0 else label_name

        if label_name == 'Pass' and pred_name == 'Pass':
            files_pp.append(file)
        elif label_name == 'Fail' and pred_name == 'Fail':
            files_ff.append(file)
        elif label_name == 'Pass' and pred_name == 'Fail':
            files_pf.append(file)
        elif label_name == 'Fail' and pred_name == 'Pass':
            files_fp.append(file)

    random.shuffle(files_pp)
    random.shuffle(files_ff)
    random.shuffle(files_pf)
    random.shuffle(files_fp)

    for file in files_pp[:max_count]:
        dir, fn, ext = get_path_info(file) 
        shutil.copy(file, f"{OUTPUT_PASS_ROOT}\\{fn}{ext}")
        if i%100 == 0: print(f'[{i}] files_pp')

    for file in files_ff[:max_count]:
        dir, fn, ext = get_path_info(file) 
        shutil.copy(file, f"{OUTPUT_FAIL_ROOT}\\{fn}{ext}")
        if i%100 == 0: print(f'[{i}] files_ff')

    for file in files_pf[:max_count]:
        dir, fn, ext = get_path_info(file) 
        shutil.copy(file, f"{OUTPUT_UNLABELED_ROOT}\\{fn}{ext}")
        if i%100 == 0: print(f'[{i}] files_pf')

    for file in files_fp[:max_count]:
        dir, fn, ext = get_path_info(file) 
        shutil.copy(file, f"{OUTPUT_UNLABELED_ROOT}\\{fn}{ext}")
        if i%100 == 0: print(f'[{i}] files_fp')


def Main():    
    # check src folder
    if not os.path.isdir(SRC_ROOT):
        print(SRC_ROOT + ' is not exist!')
        return

    # create dir
    os.makedirs(OUTPUT_PASS_ROOT, exist_ok=True)
    os.makedirs(OUTPUT_FAIL_ROOT, exist_ok=True)
    os.makedirs(OUTPUT_UNLABELED_ROOT, exist_ok=True)

    # scan
    files = scan_files_subfolder(SRC_ROOT, SEARCH_EXTS)
    
    # prepare
    # prepare_pass_fail(files)
    prepare_pp_ff_pf_fp(files)
             
    print('done')


if __name__ == '__main__':
    Main() 