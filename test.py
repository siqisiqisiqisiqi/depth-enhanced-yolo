import cv2
import numpy as np
import matplotlib.pyplot as plt

# image = np.load("test_img.npy")
# # image = np.transpose(image, (1,2,0))
# depth = image[...,3]
# image = image[:,:,0:3]
# # image = image[..., ::-1]
# # cv2.imshow("test", image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# plt.imshow(depth, cmap='viridis')  # Use a perceptually uniform colormap
# plt.colorbar(label='Depth (arbitrary units)')
# plt.title('Depth Image')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

image = np.load("test_img.npy")
image = image[3]*255
image = image.astype(np.uint8)
image = np.transpose(image, (1,2,0))
depth = image[...,0]
image = image[:,:,1:4]
image = image[..., ::-1]
cv2.imshow("test", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.imshow(depth, cmap='viridis')  # Use a perceptually uniform colormap
# plt.colorbar(label='Depth (arbitrary units)')
# plt.title('Depth Image')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()