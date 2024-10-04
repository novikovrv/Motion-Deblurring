import argparse
import cv2
import numpy as np

def find_similar_blocks_2(img1, img2, block_size=100, w_size = 120):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Проверяем, что размер блока не превышает размер изображений
    if block_size * 2 > min(h1, w1, h2, w2):
        raise ValueError("Размер блока не должен превышать размер изображений")

    # Вычисляем координаты центрального блока в img1
    center_x = w1 // 2
    center_y = h1 // 2
    center_block = img1[center_y - block_size:center_y + block_size, center_x - block_size:center_x + block_size]

    # Вычисляем координаты центральной области в img2
    center_x2 = w2 // 2
    center_y2 = h2 // 2
    start_x2 = center_x2 - w_size
    start_y2 = center_y2 - w_size
    end_x2 = center_x2 + w_size
    end_y2 = center_y2 + w_size
    center_region = img2[start_y2:end_y2, start_x2:end_x2]

    # Используем метод matchTemplate для поиска
    result = cv2.matchTemplate(center_region, center_block, cv2.TM_CCOEFF_NORMED)

    # Находим максимальное значение корреляции и его координаты
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Координаты лучшего совпадения
    print(max_loc)
    best_offset = (max_loc[0] - (w_size - block_size), max_loc[1] - (w_size - block_size))
    min_diff = max_val

    return[best_offset]


def find_similar_blocks(img1, img2, block_size=100, w_size = 120):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Проверяем, что размер блока не превышает размер изображений
    if block_size*2 > min(h1, w1, h2, w2):
        raise ValueError("Размер блока не должен превышать размер изображений")

    # Вычисляем координаты центрального блока в img1
    center_x = w1 // 2
    center_y = h1 // 2
    center_block = img1[center_y-block_size:center_y+block_size, center_x-block_size:center_x+block_size]

    # Вычисляем координаты центральной области в img2
    center_x2 = w2 // 2
    center_y2 = h2 // 2
    start_x2 = center_x2 - w_size
    start_y2 = center_y2 - w_size
    end_x2 = center_x2 + w_size
    end_y2 = center_y2 + w_size
    center_region = img2[start_y2:end_y2, start_x2:end_x2]

    # Находим наиболее похожий блок в center_region
    min_diff = -2
    #best_offset = (0, 0)
    for y2 in range(block_size, center_region.shape[0] - block_size + 1, 1):
        for x2 in range(block_size, center_region.shape[1] - block_size + 1, 1):
            block2 = center_region[y2-block_size:y2+block_size, x2-block_size:x2+block_size]
            corr = np.corrcoef(center_block.flatten(), block2.flatten())[0, 1]
            #print(corr)
            if corr > min_diff:
                min_diff = corr
                best_offset = (x2 - center_x + start_x2, y2 - center_y + start_y2)

    return [best_offset]






parser = argparse.ArgumentParser()
parser.add_argument('--prev_image_path', required=True)
parser.add_argument('--image_path', required=True)
if __name__ == '__main__':
    args = parser.parse_args()

    prev_image = cv2.imread(args.prev_image_path)
    image = cv2.imread(args.image_path)
    coords = find_similar_blocks_2(prev_image, image)
    print(coords)
