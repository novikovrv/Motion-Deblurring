import argparse
import cv2
import os
import numpy as np
from datetime import datetime, timedelta
from numpy.fft import fft2, ifft2
import time

# Декоратор для измерения времени выполнения функции
def time_logger(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Запоминаем время начала
        result = func(*args, **kwargs)  # Выполняем функцию
        end_time = time.time()  # Запоминаем время окончания
        elapsed_time = end_time - start_time
        log_message = f"Функция '{func.__name__}' выполнена за {elapsed_time:.4f} секунд"
        print(log_message)  # Выводим в консоль
        with open('function_timing.log', 'a') as log_file:
            log_file.write(log_message + '\n')  # Записываем в лог-файл
        return result
    return wrapper

@time_logger
def wiener_filter(img, kernel, K):
    kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = fft2(dummy)
    kernel = fft2(kernel, s=img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(ifft2(dummy))
    return dummy

@time_logger
def find_area_with_most_corners(old_gray, grid_size=100):

    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, maxCorners=50, qualityLevel=0.2, minDistance=10)

    # Проверка наличия угловых точек
    if p0 is None:
        return None

    # Преобразуем углы в двумерный массив координат
    points = np.int32(p0)

    # Создаём сетку для анализа
    grid_counts = {}

    for point in points:
        x, y = point.ravel()
        grid_x = x // grid_size
        grid_y = y // grid_size
        grid_key = (grid_x, grid_y)

        if grid_key not in grid_counts:
            grid_counts[grid_key] = 0
        grid_counts[grid_key] += 1

    # Находим ячейку с наибольшим количеством угловых точек
    max_count = 0
    max_grid = None

    for grid_key, count in grid_counts.items():
        if count > max_count:
            max_count = count
            max_grid = grid_key

    # Вычисляем координаты центра области с наибольшим количеством углов
    if max_grid is not None:
        center_x = (max_grid[0] * grid_size) + (grid_size // 2)
        center_y = (max_grid[1] * grid_size) + (grid_size // 2)
        print(center_x, center_y)
        return (center_x, center_y)


    return None


def find_similar_blocks(img1, img2, block_size=100, w_size=120):

    center_x, center_y = find_area_with_most_corners(img1)
    center_block = img1[center_y - block_size:center_y + block_size, center_x - block_size:center_x + block_size]


    start_x2 = center_x - w_size
    start_y2 = center_y - w_size
    end_x2 = center_x + w_size
    end_y2 = center_y + w_size
    center_region = img2[start_y2:end_y2, start_x2:end_x2]

    result = cv2.matchTemplate(center_region, center_block, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    best_offset = (max_loc[0] - (w_size - block_size), max_loc[1] - (w_size - block_size))
    min_diff = max_val


    return [best_offset]


def extract_frames(video_capture, start_time, end_time, path):
    start_t = time.time()  # Запоминаем время начала
    frames = []

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        current_time = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if start_time <= current_time <= end_time:
            frames.append(frame)
        if current_time > end_time:
            break
    video_capture.release()
    e_time = time.time()  # Запоминаем время окончания
    elapsed_time = e_time - start_t
    with open(path, 'a') as log_file:
        log_message = f"Выбор {len(frames)} кадров: {elapsed_time} секунд"
        log_file.write(log_message + '\n')

    return frames

def kernel_points(frames, period, block_size, win_size, path):
    start_time = time.time()  # Запоминаем время начала
    first = frames[0]
    first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    coords = []
    for i in range(1, len(frames), period):
        frame = frames[i]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        coords.append(find_similar_blocks(first, frame, block_size, win_size))
        #first = frame
    end_time = time.time()  # Запоминаем время окончания
    elapsed_time = end_time - start_time
    with open(path, 'a') as log_file:
        log_message = f"Функция kernel_points(ищет смещения между {len(coords)} кадрами видео и возвращает координаты для отрисовки ядра) выполнена за {elapsed_time:.6f} секунд"
        log_file.write(log_message + '\n')
    print(coords)
    return coords

@time_logger
def mat_array_normalization(array_of_matrix):
    normalized_matrix = np.empty_like(array_of_matrix, dtype=np.float32)
    for i in range(array_of_matrix.shape[0]):
        matr = array_of_matrix[i]
        sum_values = np.sum(matr)
        if sum_values != 0:
            normalized_matrix[i] = matr.astype(np.float32) / sum_values
        else:
            normalized_matrix[i] = matr.astype(np.float32)
    return normalized_matrix

@time_logger
def mat_sum(array_of_matrix):
    sum_matrix = np.sum(array_of_matrix, axis=0)
    return sum_matrix

@time_logger
def draw_line(matr, x, y, width=1):
    cv2.line(matr, x, y, 255, width, lineType=cv2.LINE_AA)
    return matr

def draw_coordinates(image_array, center_x, center_y, a, width, path):
    start_time = time.time()  # Запоминаем время начала
    count_iter = 0
    x0 = 0
    y0 = 0
    for shift in a:
        x, y = shift[0]
        x = x - x0
        y = y - y0
        end_x = center_x + (x)
        end_y = center_y + (y * 2)
        x1 = (center_x, center_y)
        y1 = (end_x, end_y)
        image_array[count_iter] = draw_line(image_array[count_iter], x1, y1, width)
        #print(center_x, center_y, end_x, end_y)
        with open(path, 'a') as log_file:
            log_message = f"{(center_x, center_y, end_x, end_y)}"
            log_file.write(log_message + '\n')
        center_x = end_x
        center_y = end_y
        count_iter += 1
        x0 = x + x0
        y0 = y + y0
    end_time = time.time()  # Запоминаем время окончания
    elapsed_time = end_time - start_time
    with open(path, 'a') as log_file:
        log_message = f"Функция draw_coordinates выполнена за {elapsed_time:.6f} секунд"
        log_file.write(log_message + '\n')
    return image_array



parser = argparse.ArgumentParser()
parser.add_argument('--video_name', required=True)
parser.add_argument('--path', required=True)
parser.add_argument('--image_name', required=True)
parser.add_argument('--image_time', required=True)
parser.add_argument('--end_video_time', required=True)
parser.add_argument('--exp_time', required=True)
parser.add_argument('--kernel_size', required=True)
parser.add_argument('--period', default=1)
parser.add_argument('--block_size', default=100)
parser.add_argument('--win_size', default=150)
parser.add_argument('--kernel_width', default=1)
parser.add_argument('--kernel_noise', default=0.01)

if __name__ == '__main__':
    args = parser.parse_args()
    video_location = os.path.join(args.path, args.video_name)
    cap = cv2.VideoCapture(video_location)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    image_time = datetime.strptime(args.image_time, "%H:%M:%S.%f")
    video_end_time = datetime.strptime(args.end_video_time, "%H:%M:%S.%f")
    video_start_time = (video_end_time - timedelta(seconds=duration))

    start_frame_time = (image_time - video_start_time).total_seconds()
    end_frame_time = start_frame_time + float(args.exp_time)

    image_name = os.path.splitext(os.path.basename(args.image_name))[0]
    new_folder_kernel = os.path.join(args.path, image_name + "_kernels")
    new_folder_image = os.path.join(args.path, image_name + "_deconvolved")
    if not os.path.exists(new_folder_kernel):
        os.mkdir(new_folder_kernel)
    if not os.path.exists(new_folder_image):
        os.mkdir(new_folder_image)
    output_path = os.path.join(new_folder_kernel, f"{image_name}_{int(args.block_size)*2}_{int(args.win_size)*2}_{int(args.period)}_kernel.png")
    log_path = os.path.join(new_folder_kernel, f"{image_name}_{int(args.block_size)*2}_{int(args.win_size)*2}_{int(args.period)}_kernel.log")


    selected_frames = extract_frames(cap, start_frame_time, end_frame_time, log_path)
    coords = kernel_points(selected_frames, int(args.period),
                           int(args.block_size), int(args.win_size), log_path)
    kernel_size = int(args.kernel_size)
    array_of_matrix = np.zeros((len(coords), kernel_size, kernel_size), dtype=np.uint8)
    k = draw_coordinates(array_of_matrix, kernel_size // 2, kernel_size // 2, coords, int(args.kernel_width), log_path)
    k = mat_array_normalization(k)
    kernel = mat_sum(k)

    cv2.imwrite(output_path, kernel * 1500.0)
    im_path = os.path.join(args.path, args.image_name)
    image = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2GRAY)
    kernel = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
    kernel = kernel.astype(np.float32)

    deconvolved_image = wiener_filter(image, kernel, float(args.kernel_noise))
    deconv_output_path = os.path.join(new_folder_image, f"{image_name}_{int(args.block_size)*2}_{int(args.win_size)*2}_{int(args.period)}_deconv.png")
    cv2.imwrite(deconv_output_path, deconvolved_image)