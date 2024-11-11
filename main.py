from src.utils.model_load import ModelExt
from src.models.model import PoseModel
from src.train import PoseTrain

# temporary data
data = {'test': None, 'kpt_shape': [7, 3], 'flip_idx': [0, 1, 2, 5, 6, 3, 4],
        'names': {0: 'green onion'}, 'translate': 0.2, 'scale': 0.2,
        'shear': 0.2, 'perspective': 0.1, 'flipud': 0.7, 'fliplr': 0.5,
        'mosaic': 0.3, 'mixup': 0.1, 'nc': 1}

####################################### fine tunning #########################################
####################################### This is a test ########################################
# m = ModelExt("./exp/checkpoints/yolo11m-pose.pt")
# cfg = m.model.yaml
# # Parse the YOLO model.yaml model into a nn.module model
# model = PoseModel(
#     cfg, ch=3, nc=data["nc"], data_kpt_shape=data["kpt_shape"], verbose=True)
# model.load(m.model)
############################## dataset construction ################################


############################## create model from the yaml file ################################
m = ModelExt('./exp/model_config/yolo11-pose.yaml')
model = PoseModel(
    m, ch=3, nc=data["nc"], data_kpt_shape=data["kpt_shape"], verbose=True)

############################## train the model ################################
trainer = PoseTrain(model)
results = trainer.train(
    data="./dataset/green_onion_skeleton.yaml", batch=4, epochs=20, imgsz=1080)
print("This is a test!")
