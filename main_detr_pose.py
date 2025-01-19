import yaml

from src.utils.model_load import ModelExt
from src.models.model import RTDETRPoseModel
from src.train import RTDETRTrain
from src.utils.transfer_learning import weight_transfer

############################## dataset construction ################################
dataset_path = "./dataset/green_onion_skeleton_three_points.yaml"
with open(dataset_path) as file:
    data_config = yaml.safe_load(file)

if data_config['depth']:
    ch = 4
else:
    ch = 3
nc = len(data_config['names'])

######################### create model from the yaml file ###########################
m = ModelExt('./exp/model_config/rtdetr-l-pose.yaml')
m2 = ModelExt('./exp/model_weights/rtdetr-l-green_onion.pt')
model = RTDETRPoseModel(m, ch=ch, nc=nc, verbose=True)
model = weight_transfer(model, m2.model)
del m2

############################## train the model ################################
trainer = RTDETRTrain(model)
results = trainer.train(
    data=dataset_path, batch=4, epochs=200, imgsz=640)
