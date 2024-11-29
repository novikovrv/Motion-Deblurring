import subprocess
import argparse
import csv
import os
import time

def time_logger(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Запоминаем время начала
        result = func(*args, **kwargs)  # Выполняем функцию
        end_time = time.time()  # Запоминаем время окончания
        elapsed_time = end_time - start_time
        log_message = f"Функция '{func.__name__}' выполнена за {elapsed_time:.4f} секунд"
        print(log_message)  # Выводим в консоль
        return result
    return wrapper


def read_csv(file_path):
    with open(file_path, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)  # Возвращаем список словарей

@time_logger
def run_file1(args):
    subprocess.run(['python', 'motion_deblurring.py',
                    f'--video_name={args["video_name"]}',
                    f'--path={args["path"]}',
                    f'--image_name={args["image_name"]}',
                    f'--image_time={args["image_time"]}',
                    f'--end_video_time={args["end_video_time"]}',
                    f'--exp_time={args["exp_time"]}',
                    f'--kernel_size={args["kernel_size"]}',
                    f'--period={args["period"]}',
                    f'--block_size={args["block_size"]}',
                    f'--win_size={args["win_size"]}',
                    f'--kernel_width={args["kernel_width"]}',
                    f'--kernel_noise={args["kernel_noise"]}'], shell=True)


parser = argparse.ArgumentParser()
parser.add_argument('--image_csv_path', required=True)
parser.add_argument('--video_csv_path', required=True)

if __name__ == "__main__":
    arg = parser.parse_args()
    parent_path = os.path.dirname(arg.image_csv_path)
    video_data = read_csv(arg.video_csv_path)[0]  # Предполагаем, что в каждом файле одна строка
    im_data = read_csv(arg.image_csv_path)
    param1_range = range(60, 76, 30)
    param2_range = range(80, 102, 60)



    # Цикл для итерации по параметрам

    start_time = time.time()  # Запоминаем время начала
    for i in range(len(param1_range)):
        for time_data in im_data:
            block_size = param1_range[i]
            win_size = param2_range[i]
            period = 1
            kernel_size = 500
            kernel_noise = 0.01
            kernel_width = 1
            # Объединяем данные в один словарь
            args = {
                'video_name': video_data['video_name'],
                'path': parent_path,
                'image_name': time_data['image_name'],
                'image_time': time_data['image_time'],
                'end_video_time': video_data['end_video_time'],
                'exp_time': time_data['exp_time'],
                'kernel_size': kernel_size,
                'period': period,
                'block_size': block_size,
                'win_size': win_size,
                'kernel_width': kernel_width,
                'kernel_noise': kernel_noise
            }
            run_file1(args)

    log = os.path.join(parent_path + "general_time.log")
    end_time = time.time()  # Запоминаем время окончания
    elapsed_time = end_time - start_time
    log_message = f"Функция deblurring_utility выполнена за {elapsed_time:.4f} секунд"
    with open(log, 'a') as log_file:
        log_file.write(log_message + '\n')  # Записываем в лог-файл




