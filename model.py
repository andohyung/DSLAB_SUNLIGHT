
import torch
import torch.nn as nn
from torchvision import models

# class ResNet34Model(nn.Module):
#     def __init__(self, num_classes=10):
#         super(ResNet34Model, self).__init__()
#         # ResNet34 불러오기
#         self.model = models.resnet34(pretrained=True)
        
#         # 마지막 FC 레이어를 교체
#         num_features = self.model.fc.in_features
#         self.model.fc = nn.Linear(num_features, num_classes)

#     def forward(self, x):
#         return self.model(x)

# # 모델 인스턴스 생성
# model = ResNet34Model(num_classes=2) #우리가 구별해줘야 하는 클래스가 2개이기때문에 2로 설정
        
        
class ResNet50Model(nn.Module):
    def __init__(self, num_classes=2):  # 클래스가 2개이므로 num_classes를 2로 설정
        super(ResNet50Model, self).__init__()
        # ResNet50 불러오기
        self.model = models.resnet50(pretrained=True)
        
        # 마지막 FC 레이어를 교체
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)

# 모델 인스턴스 생성
model = ResNet50Model(num_classes=2)



# class ResNet18Model(nn.Module):
#     def __init__(self, num_classes=2):  # Assuming 2 classes
#         super(ResNet18Model, self).__init__()
#         # ResNet18 pre-trained model
#         self.model = models.resnet18(pretrained=True)
        
#         # Replace the last fully connected layer
#         num_features = self.model.fc.in_features
#         self.model.fc = nn.Linear(num_features, num_classes)

#     def forward(self, x):
#         return self.model(x)

# # Instantiate the ResNet-18 model
# model = ResNet18Model(num_classes=2)