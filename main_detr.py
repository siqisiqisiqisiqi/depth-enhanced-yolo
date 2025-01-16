import yaml

from src.utils.model_load import ModelExt
from src.models.model import RTDETRDetectionModel
from src.train import RTDETRTrain

############################## dataset construction ################################
dataset_path = "./dataset/coco8.yaml"
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
m = ModelExt('./exp/model_config/rtdetr-l.yaml')
model = RTDETRDetectionModel(m, ch=ch, nc=nc, verbose=True)

############################## train the model ################################
trainer = RTDETRTrain(model)
results = trainer.train(
    data=dataset_path, batch=4, epochs=2, imgsz=640)
print("This is a test!")
