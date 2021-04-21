from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
import os
import ntpath
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
    from file import make_dir, handle_src_dest_dir
except ImportError:
    from .file import make_dir, handle_src_dest_dir

image_extensions = ["jpe", "jpeg", "JPG", "jfif", "exif",
                    "tiff", "bmp", "png", "webp", "jpg"]
available_formats = ["jpg", "png", "webp"]


def reformat_images(
    src_dir,
    dest_dir="",
    format="jpg",
    remove_old=False,
    remove_unavailable=False,
    force_replace=False,
    log=False,
):
    try:
        dest_dir = handle_src_dest_dir(src_dir, dest_dir, force_replace)

        files = os.listdir(src_dir)
        for f in files:
            src = os.path.join(src_dir, f)
            name, ext = os.path.splitext(f)
            ext = ext.replace('.', '', 1)
            if ext in image_extensions:
                dest = os.path.join(dest_dir, name+"."+format)
                reformat(src, dest, format=format,
                         remove_old=remove_old, remove_unavailable=remove_unavailable, log=log)
            else:
                print(f"{src} is unavailable")
                if remove_unavailable:
                    os.remove(src)
                    if log:
                        print(f"Remove {src}")
    except Exception as e:
        print(f"Reformat {src_dir} failed with err {e}.")


def reformat(
        path,
        dest="",
        format="jpg",
        remove_old=False,
        remove_unavailable=False,
        log=False):
    try:
        if format not in available_formats:
            raise Exception(f"{format} is unavailable")
        im = Image.open(path).convert("RGB")
        save_format = format
        if format == "jpg":
            save_format = "jpeg"
        im.save(dest, save_format)
        if log:
            print(f"{path} has been converted to {dest}")
        if remove_old and path != dest:
            os.remove(path)
            if log:
                print(f"Remove {path}")
    except Exception as e:
        print(f"Reformat {path} failed with err {e}.")
        if remove_unavailable:
            if os.path.exists(path):
                os.remove(path)
                print(f"Remove {path}")
            if os.path.exists(dest):
                os.remove(dest)
                print(f"Remove {dest}")


def resize(
    path,
    dest_dir,
    max_w=500,
    max_h=500,
    log=False,
):
    im = Image.open(path)
    w, h = im.size
    if w <= max_w and h <= max_h:
        return
    resize_im(im, path, dest_dir, max_w, max_h, log)


def resize_im(
    im,
    path,
    dest_dir,
    max_w=500,
    max_h=500,
    log=False,
):
    try:
        im.thumbnail((max_w, max_h), Image.ANTIALIAS)
        _, tail = ntpath.split(path)
        dest_dir = make_dir(dest_dir)
        dest = os.path.join(dest_dir, tail)
        im.save(dest)
        if log:
            print(f"Resized {path}")
    except Exception as e:
        print(f"Resize {path} failed with err {e}.")


def resize_images(
    src_dir,
    dest_dir='',
    max_w=500,
    max_h=500,
    force_replace=False,
    log=False,
):
    try:
        dest_dir = handle_src_dest_dir(src_dir, dest_dir, force_replace)

        files = os.listdir(src_dir)
        for f in files:
            src = os.path.join(src_dir, f)
            dest = os.path.join(dest_dir, f)
            im = Image.open(src)
            w, h = im.size
            if w <= max_w and h <= max_h:
                if dest_dir == src_dir:
                    if log:
                        print(f"Escape {src}")
                else:
                    copyfile(src, dest)
                    if log:
                        print(f"Copy {src} to {dest}")
                continue
            resize_im(im, src, dest_dir, max_w, max_h, log)
    except Exception as e:
        print(f"Resize {src_dir} failed with err {e}.")


datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')


def generate_in_dir(
        src_dir,
        dest_dir='',
        generator=datagen,
        flag='_aug_',
        target_size=(224, 224),
        num=10,
        force_replace=False,
        log=False):
    try:
        dest_dir = handle_src_dest_dir(src_dir, dest_dir, force_replace)

        files = os.listdir(src_dir)
        for f in files:
            path = os.path.join(src_dir, f)
            generate(path, dest_dir, generator, flag, target_size, num, log)
    except Exception as e:
        print(f'Generate {src_dir} failed with err {e}.')


def generate(
        path,
        dest_dir='',
        generator=datagen,
        flag='_aug_',
        target_size=(224, 224),
        num=10,
        log=False):
    try:
        dir, name = os.path.split(path)
        _, extension = os.path.splitext(path)
        pure_name = name.replace(extension, '', 1)
        if dest_dir == '':
            dest_dir = dir
        dest_dir = make_dir(dest_dir)

        img = image.load_img(path, target_size=target_size)
        x = image.img_to_array(img)
        x = x.reshape((1,) + x.shape)
        num = num - 1
        if log:
            print(f'Generating images for {path}')
        for i, batch in enumerate(generator.flow(x, batch_size=1)):
            draw = image.array_to_img(batch[0])
            new_name = f"{pure_name}{flag}{i}{extension}"
            save_path = os.path.join(dest_dir, new_name)
            draw.save(save_path)
            if log:
                print(f'Generated {save_path} for {path}')
            if i == num:
                break

    except Exception as e:
        print(f'Generate {path} failed with err {e}.')


if __name__ == '__main__':
    src_dir = '/Users/catchzeng/Desktop/test3'
    generate_in_dir(src_dir, src_dir, log=True, force_replace=True)
