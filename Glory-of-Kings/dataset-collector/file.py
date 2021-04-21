import os
import shutil

def rename_files(
    dir,
    prefix="f",
    start_index = 0,
    log = False):
    files = os.listdir(dir)
    index = start_index
    if log:
        print("Renaming images")
    for f in files:
        if f.startswith("."):
            if log:
                print(f"Escape {f}")
            continue
        src = os.path.join(dir, f)
        name = rename(f, index, prefix)
        dst = os.path.join(dir, name)
        os.rename(src, dst)
        if log:
            print(f"{src} has been renamed to {dst}")
        index = index + 1
    if log:
        print("Finished renaming")

def rename(path, index, prefix='f') -> str:
    try:
        _, extension = os.path.splitext(path)
        result = f"{prefix}{index}{extension}"
        return result
    except Exception as e:
        print(f"Rename: {path} err: {e}")
        return prefix

def make_dir(dir, force_replace=False) -> str:
    if len(dir) < 1:
        dir = os.getcwd()

    if force_replace:
        if os.path.isdir(dir):
            shutil.rmtree(dir)
    try:
        if not os.path.isdir(dir):
            os.makedirs(dir)
    except:
        pass

    return dir


def handle_src_dest_dir(src_dir, dest_dir, force_replace=False):
    if force_replace and src_dir != dest_dir:
        if os.path.isdir(dest_dir):
            shutil.rmtree(dest_dir)
    if len(dest_dir) < 1:
        dest_dir = src_dir
    dest_dir = make_dir(dest_dir)
    return dest_dir

if __name__ == '__main__':
    make_dir("/Users/catchzeng/desktop/ttt", force_replace=True)