import torch

# a = torch.randn(5,8)
# b = a.split((3,2,3),dim=-1)
# print(b)

m = "rtdertdecoder"

if m in ["pose", "rtdetrposedecoder"]:
    print("hello")

if m == "pose" or m == "rtdetrposedecoder":
    print("wrong answer")