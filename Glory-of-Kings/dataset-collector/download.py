from bing_images import bing
from file import rename_files

heros = {
    "铠": "kai",
    "后羿": "houyi",
    "王昭君": "wangzhaojun"
}

if __name__ == '__main__':
    # for (key, value) in heros.items():
    #     query = f'王者荣耀 {key}'
    #     output_dir = f'../dataset/{value}'
    #     bing.download_images(query,
    #                     150,
    #                     output_dir= output_dir,
    #                     pool_size=5,
    #                     force_replace=True)
                        
    for value in heros.values():
        output_dir = f'/Users/catchzeng/Desktop/Glory of Kings/train/{value}'
        rename_files(output_dir,value,1)