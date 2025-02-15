
import cv2
import numpy as np
from glob import glob

from src.utils.color import colors
from src.utils.model_load import ModelExt
from src.predict import PoseRTDETRPredict

image_path_list = glob("./test/test_data/images/*")


def visualization(image_path, results):
    img = cv2.imread(image_path)
    keypoints = results[0].keypoints.data.detach(
    ).cpu().numpy().astype(np.int32)
    num_object = keypoints.shape[0]
    for keypoint in keypoints:
        keypoint = keypoint[~np.all(keypoint == 0, axis=1)]
        num_point = keypoint.shape[0]
        for i in range(num_point):
            point = keypoint[i][:2]
            cv2.circle(img, point, 3, colors[i], -1)
            # cv2.putText(img, str(i + 1), (point[0] + 10, point[1]),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
            if i < keypoint.shape[0]-1:
                cv2.line(img, keypoint[i][:2], keypoint[i + 1]
                         [:2], colors[5], thickness=2)
        # keypoint = keypoint[keypoint[:, -1] != 0]

        cv2.imshow("Image", img)
        k = cv2.waitKey(0)
    return k


for image_path in image_path_list:

    ############################## rgb inference ################################
    m = ModelExt('./exp/runs/pose/train24/weights/best.pt')
    # m = ModelExt('./exp/runs/pose/train14/weights/best.pt')
    predictor = PoseRTDETRPredict(m)
    results = predictor.predict(image_path)

    ############################## visualization ################################
    # for r in results:
    #     im_array = r.plot()
    #     cv2.imshow("image", im_array)
    #     k = cv2.waitKey(0)
    # if k == ord("q"):
    #     break

    k = visualization(image_path, results)
    if k == ord("q"):
        break

cv2.destroyAllWindows()
