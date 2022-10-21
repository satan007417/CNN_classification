import os
import glob

def scan_dirs_folder(folder):
    return [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]


def scan_files_subfolder(root_folder, search_exts):
    searched_files = []
    for root, dirs, files in os.walk(root_folder):
        for f in files:
            if any(f.endswith(ext) for ext in search_exts):
                searched_files.append(os.path.join(root, f))
    return searched_files


def scan_files_folder(root_folder, search_exts):
    files = [x for x in glob.glob(root_folder + '\\*.*') if any(x.endswith(ext) for ext in search_exts)]
    return files


def Main():
    img_path_root = r'D:\BackendAOI\Data\ForeignMaterial\1\FM_Black'
    search_exts = ['jpg','jpeg', 'bmp', 'png']

    # split
    files = scan_files_subfolder(img_path_root, search_exts)
    
    for f in files:
        dir = os.path.dirname(f)
        fn, ext = os.path.splitext(os.path.basename(f))
        txt = f'{dir}\\{fn}.txt'
        if not os.path.exists(txt):
            os.remove(f)
        elif os.path.getsize(txt) == 0:
            os.remove(f)
            os.remove(txt)
    
    print('done')
    
if __name__ == '__main__':
    Main() 