import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

class DMSModel(nn.Module):
    def __init__(self):
        super(DMSModel, self).__init__()
        
        # Use MobileNetV2 as the backbone
        mobilenet = mobilenet_v2(pretrained=False)
        self.features = mobilenet.features
        
        # In the paper, features from intermediate blocks are concatenated.
        # For simplicity in this dummy architecture, we will extract features
        # from the end of the MobileNetV2 backbone (1280 channels) and project them.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 209 elements output in total, but separated into 6 distinct heads
        # as expected by the inference code.
        self.head_landmark = nn.Linear(1280, 196) # raw[0]
        self.head_euler = nn.Linear(1280, 3)      # raw[1]
        self.head_eyes = nn.Linear(1280, 4)       # raw[2] (Openness + Viz)
        self.head_mouth = nn.Linear(1280, 3)      # raw[3]
        self.head_act = nn.Linear(1280, 3)        # raw[4]
        self.head_act_score = nn.Linear(1280, 1)  # raw[5] (unused but expected)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        landmark = self.head_landmark(x)
        euler = self.head_euler(x)
        eyes = self.head_eyes(x)
        mouth = self.head_mouth(x)
        act = self.head_act(x)
        act_score = self.head_act_score(x)
        
        return landmark, euler, eyes, mouth, act, act_score

def export_onnx():
    print("Building DMS model...")
    model = DMSModel()
    model.eval()
    
    # Dummy input based on inference code expectations (160x160 RGB)
    # The pre-processing in inference: shape is (1, 3, 160, 160)
    dummy_input = torch.randn(1, 3, 160, 160)
    
    onnx_path = "./models/reborn.onnx"
    print(f"Exporting to {onnx_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        input_names=['input'], 
        output_names=['landmark', 'euler', 'eyes', 'mouth', 'act', 'act_score'],
        opset_version=11
    )
    print("Export complete.")

if __name__ == '__main__':
    export_onnx()
