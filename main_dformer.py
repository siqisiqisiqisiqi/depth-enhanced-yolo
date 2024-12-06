import yaml

from src.utils.model_load import ModelExt
from src.models.model_deformer import PoseModel
from src.train import PoseTrain


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
m = ModelExt('./exp/model_config/yolo11m-pose-v2.yaml')
model = PoseModel(
    m, ch=ch, nc=nc, data_kpt_shape=data_config["kpt_shape"], verbose=True)

############################## train the model ################################
trainer = PoseTrain(model)
results = trainer.train(
    data=dataset_path, batch=4, epochs=200, imgsz=640)
print("This is a test!")
