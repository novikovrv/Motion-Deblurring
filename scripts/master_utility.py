import argparse
import os
import subprocess

def main(root_folder):
    # Проходим по всем подпапкам в указанной папке
    for dirpath, _, filenames in os.walk(root_folder):
        #print(filenames)
        # Проверяем наличие необходимых файлов
        image_file = os.path.join(dirpath, 'image.csv')
        video_file = os.path.join(dirpath, 'video.csv')

        if all(os.path.isfile(file) for file in [image_file,  video_file]):
            # Запускаем deblurring_utility.py с указанными файлами
            subprocess.run(['python', 'deblurring_utility.py',
                            f'--image_csv_path={image_file}',
                            f'--video_csv_path={video_file}'], shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', required=True, help='Путь к папке с подкаталогами')
    args = parser.parse_args()  # Парсим аргум
    main(args.root_folder)  # D:\source\PycharmProjects\Motion_Deblur\Data
