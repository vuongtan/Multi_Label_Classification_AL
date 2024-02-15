import torch
import torch.nn as nn
import torchvision.models as models
# from torchvision.models.resnet import ResNet50_Weights
from torchvision.models import resnet50, ResNet50_Weights
# from torchvision.models import resnet50
class ClassWiseAttention(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(ClassWiseAttention, self).__init__()
        self.num_classes = num_classes
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            ) for _ in range(num_classes)
        ])
    
    def forward(self, features):
        # Ensure features tensor has the expected shape 
        batch_size, _, H, W = features.size()
        attention_maps = torch.stack([head(features) for head in self.attention_heads], dim=1)
        return attention_maps

class MultiClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(MultiClassificationModel, self).__init__()
        # Initialize ResNet without the final layer
        original_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # original_model = torch.hub.load('pytorch/vision:0.9.1', 'resnext50_32x4d', pretrained=True)
        # original_model=models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*(list(original_model.children())[:-2]))
        
        # Assuming the feature maps size from modified ResNet50 is 2048 channels
        self.class_wise_attention = ClassWiseAttention(num_classes, 2048)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifiers = nn.ModuleList([nn.Linear(2048, 1) for _ in range(num_classes)])
        
        # Assuming the feature maps size from ResNet50 is 2048
        self.class_wise_attention = ClassWiseAttention(num_classes, 2048)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifiers = nn.ModuleList([
            nn.Linear(2048, 1) for _ in range(num_classes)
        ])
    
    def forward(self, x):
        feature_maps = self.feature_extractor(x) # This should now be a 4D tensor
        if feature_maps.dim() != 4:
            raise ValueError(f"Expected feature_maps to be a 4D tensor, got {feature_maps.shape}")
        attention_maps = self.class_wise_attention(feature_maps)
        attended_features = feature_maps.unsqueeze(1) * attention_maps
        pooled_features = self.pooling(attended_features).view(attended_features.size(0), attended_features.size(1), -1)
        outputs = torch.cat([classifier(pooled_features[:, i]) for i, classifier in enumerate(self.classifiers)], dim=1)
        return outputs

# Example model initialization
if __name__ == "__main__":
    num_classes = 10  # Example number of classes
    model = MultiClassificationModel(num_classes)
    print(model)
