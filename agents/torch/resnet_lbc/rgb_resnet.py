import sys
import torch
from models import modified_res

#Returns pretrained Learning by cheating resnet model
def get_resnet():
    model_args = {
            "backbone": "resnet34",
            "model": "image_ss"
        }
    model = modified_res.ModifiedImagePolicyModelSS(**model_args)

    weights = torch.load('model-10.th')
    old_weights=model.state_dict()

    #Load weights only for the feature extractor
    for name,params in old_weights.items():
        try:
            #print("Loading Weights For : ",name)
            old_weights[name] = weights[name]
        except:
            print("Couldn't Find Key : ",name)
            continue
    model.load_state_dict(old_weights)
    return model

