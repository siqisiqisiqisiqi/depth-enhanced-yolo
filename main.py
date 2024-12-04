import yaml

from src.utils.model_load import ModelExt
from src.models.model import PoseModel
from src.train import PoseTrain

# temporary data
# data = {'test': None, 'kpt_shape': [7, 3], 'flip_idx': [0, 1, 2, 5, 6, 3, 4],
#         'names': {0: 'green onion'}, 'translate': 0.2, 'scale': 0.2,
#         'shear': 0.2, 'perspective': 0.1, 'flipud': 0.7, 'fliplr': 0.5,
#         'mosaic': 0.3, 'mixup': 0.1, 'nc': 1}

####################################### fine tunning #########################################
####################################### This is a test ########################################
# m = ModelExt("./exp/checkpoints/yolo11m-pose.pt")
# cfg = m.model.yaml
# # Parse the YOLO model.yaml model into a nn.module model
# model = PoseModel(
#     cfg, ch=3, nc=data["nc"], data_kpt_shape=data["kpt_shape"], verbose=True)
# model.load(m.model)

############################## dataset construction ################################
dataset_path = "./dataset/green_onion_skeleton_three_points.yaml"
with open(dataset_path) as file:
    data_config = yaml.safe_load(file)
# with open("./dataset/green_onion_skeleton.yaml") as file:
#     data_config = yaml.safe_load(file)
if data_config['depth']:
    ch = 4
else:
    ch = 3
nc = len(data_config['names'])
######################### create model from the yaml file ###########################
m = ModelExt('./exp/model_config/yolo11m-pose.yaml')
model = PoseModel(
    m, ch=ch, nc=nc, data_kpt_shape=data_config["kpt_shape"], verbose=True)

############################## train the model ################################
trainer = PoseTrain(model)
results = trainer.train(
    data=dataset_path, batch=4, epochs=1, imgsz=1080)
print("This is a test!")
