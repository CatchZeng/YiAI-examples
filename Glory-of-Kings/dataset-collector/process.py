from file import rename_files
from image import reformat_images, resize_images, generate_in_dir, datagen
import random
import string

heros = {
    "铠": "kai",
    "后羿": "houyi",
    "王昭君": "wangzhaojun"
}

def process_images(
    src_dir,
    dest_dir="",
    format="jpg",
    remove_old=False,
    remove_unavailable=False,
    max_w=512,
    max_h=512,
    rename_prefix="f",
    rename_start_index=0,
    generator=datagen,
    gen_flag='_aug_',
    gen_target_size=(224, 224),
    gen_num=10,
    force_replace=False,
    log=False,
):
    rename_files(dest_dir, ran_name(), rename_start_index, False)
    reformat_images(src_dir, dest_dir, format,
                    remove_old, remove_unavailable, force_replace, log)
    resize_images(dest_dir, dest_dir, max_w, max_h, force_replace, log)
    rename_files(dest_dir, ran_name(), rename_start_index, False)
    generate_in_dir(dest_dir, dest_dir, generator, gen_flag,
                    gen_target_size, gen_num, force_replace, log)
    rename_files(dest_dir, rename_prefix, rename_start_index, log)


def ran_name():
    name = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    return name


if __name__ == '__main__':
    for (key, value) in heros.items():
        src_dir = f'../dataset/{value}'
        process_images(src_dir, src_dir, remove_old=True,
                       remove_unavailable=True, rename_prefix=value, rename_start_index=1)
