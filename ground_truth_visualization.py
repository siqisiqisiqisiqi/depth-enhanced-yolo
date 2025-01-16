import os
import sys
from glob import glob

import numpy as np
import cv2

from src.utils.color import colors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


def read_txt_file(file_path):
    try:
        with open(file_path, 'r') as file:
            contents = file.readlines()
        return contents
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def kpts_conversion(kpts, dim):
    X = dim[0]
    Y = dim[1]
    kpts_array = np.array(kpts).reshape((-1, 3))
    kpts_new = np.zeros_like((kpts_array))
    kpts_new[:, 0] = X * kpts_array[:, 0]
    kpts_new[:, 1] = Y * kpts_array[:, 1]
    kpts_new[:, 2] = kpts_array[:, 2]
    return kpts_new


def visualization(img, keypoint):
    keypoint = keypoint[~np.all(keypoint == 0, axis=1)]
    keypoint = keypoint.astype(int)
    num_point = keypoint.shape[0]
    for i in range(num_point):
        point = keypoint[i][:2]
        cv2.circle(img, point, 3, colors[i], -1)
        if i < keypoint.shape[0] - 1:
            cv2.line(img, keypoint[i][:2], keypoint[i + 1]
                     [:2], colors[5], thickness=2)

    cv2.imshow("Image", img)
    k = cv2.waitKey(0)
    return k


def main():

    import re

    image_dir = "./test/test_data/images"
    label_dir = "./test/test_data/labels"

    for image_path in glob(image_dir + "/*"):
        num = re.findall(r'\d+', image_path)[-1]
        label_path = label_dir + f"/Image_{num}.txt"

        img = cv2.imread(image_path)
        (Y, X, _) = img.shape
        contents = read_txt_file(label_path)

        for i, content in enumerate(contents):

            content_list = content.replace('\n', ' ').split(" ")[:-1]
            content_list = [float(value) for value in content_list[:-1]]
            category = int(content_list[0])

            # keypoints visualization
            keypoint = content_list[5:5 + 3 * 3]
            keypoint = kpts_conversion(keypoint, (X, Y))
            k = visualization(img, keypoint)
            if k == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
