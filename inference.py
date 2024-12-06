import yaml

from src.utils.model_load import ModelExt
from src.predict import PosePredict
import cv2

############################## rgb inference ################################
# m = ModelExt('./weights/rgb.pt')
# predictor = PosePredict(m)
# results = predictor.predict("./test/test_data/images/Image_1.jpg")

# for r in results:
#     im_array = r.plot()
#     cv2.imshow("image", im_array)
#     cv2.waitKey(0)

############################## rgbd concat inference ################################
# m = ModelExt('./weights/rgbd_concat.pt')
# predictor = PosePredict(m, depth=True)
# results = predictor.predict("./test/test_data/images/Image_1.jpg")

# for r in results:
#     im_array = r.plot()
#     cv2.imshow("image", im_array)
#     cv2.waitKey(0)

############################## rgbd dformer inference ################################
m = ModelExt('./weights/rgbd_dformer.pt')
predictor = PosePredict(m, depth=True)
results = predictor.predict("./test/test_data/images/Image_1.jpg")

for r in results:
    im_array = r.plot()
    cv2.imshow("image", im_array)
    cv2.waitKey(0)



