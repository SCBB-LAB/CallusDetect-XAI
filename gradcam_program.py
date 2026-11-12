import torch
import torch.nn as nn  # Add this import to avoid 'nn is not defined' error
import torch.nn.functional as F
import numpy as np
import cv2, sys, os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 50
VAL_SPLIT = 0.2
IMAGE_SIZE = 1000
# Define the same model structure as before (DenseNet161)
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = nn.functional.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
            self.add_module(f'denselayer{i+1}', layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet161(nn.Module):
    def __init__(self, growth_rate=48, block_config=(6, 12, 36, 24),
                 num_init_features=96, bn_size=4, drop_rate=0, num_classes=1):
        super(DenseNet161, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2

        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

# -----------------------------------------------------------
# Load the saved model
# -----------------------------------------------------------
def load_model(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

# -----------------------------------------------------------
# Load a single image and apply transformations
# -----------------------------------------------------------
def load_single_image(image_path, image_size=1000):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image = datasets.folder.pil_loader(image_path)  # Load the image
    image = transform(image)  # Apply transformations
    return image

# -----------------------------------------------------------
# Grad-CAM and Grad-CAM++ Implementation
# (Same as in previous message, no change)
# -----------------------------------------------------------
def get_last_conv_layer(model):
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv


def register_hooks(layer):
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    layer.register_forward_hook(forward_hook)
    layer.register_backward_hook(backward_hook)

    return activations, gradients


def grad_cam(model, image, target_class=0):
    model.eval()

    conv_layer = get_last_conv_layer(model)
    activations, gradients = register_hooks(conv_layer)

    image = image.unsqueeze(0).to(DEVICE)
    out = model(image)

    model.zero_grad()
    out[0, target_class].backward()

    grads = gradients[0].detach().cpu().numpy()
    acts = activations[0].detach().cpu().numpy()

    weights = np.mean(grads, axis=(1, 2), keepdims=True)  # GAP over gradients
    cam = np.sum(weights * acts, axis=0)
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    return cam


def grad_cam_pp(model, image, target_class=0):
    model.eval()

    conv_layer = get_last_conv_layer(model)
    activations, gradients = register_hooks(conv_layer)

    image = image.unsqueeze(0).to(DEVICE)
    out = model(image)

    model.zero_grad()
    out[0, target_class].backward(retain_graph=True)

    grads = gradients[0].detach().cpu().numpy()
    acts = activations[0].detach().cpu().numpy()

    grads2 = grads ** 2
    grads3 = grads ** 3

    eps = 1e-8
    alpha_num = grads2
    alpha_den = 2 * grads2 + (acts * grads3).sum(axis=(1, 2), keepdims=True) + eps
    alpha = alpha_num / alpha_den

    weights = (alpha * np.maximum(grads, 0)).sum(axis=(1, 2), keepdims=True)
    cam = np.sum(weights * acts, axis=0)
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    return cam

def overlay_cam(image_tensor, cam):
    # Ensure the image tensor is in the correct format (0 to 1)
    img = image_tensor.numpy().transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min())

    # If 'cam' has multiple channels, average across the depth (last dimension)
    if len(cam.shape) == 3:
        cam = np.mean(cam, axis=-1)  # Take the mean across the channels (depth)

    # Ensure 'cam' is a 2D array (height, width)
    if len(cam.shape) != 2:
        raise ValueError(f"Expected a 2D cam, but got {cam.shape}. Ensure cam is 2D.")

    # Scale 'cam' values to the range [0, 255] and convert to uint8
    cam = np.uint8(255 * cam)  # Ensure 'cam' is of type uint8

    # Apply the color map to the cam
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
#    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_INFERNO)
    # Normalize the heatmap to match the image scale (0 to 1)
    heatmap = heatmap.astype(np.float32) / 255.0  # Normalize heatmap values to [0, 1]

    # Combine the original image with the heatmap (overlay effect)
    result = (0.4 * heatmap + img)  # 0.4 is the transparency factor
    return np.uint8(result * 255)  # Convert result back to uint8 for display


def show_gradcams(model, image_path, idx=0):
    image = load_single_image(image_path, IMAGE_SIZE)
    cam1 = grad_cam(model, image, target_class=0)
    cam2 = grad_cam_pp(model, image, target_class=0)

    overlay1 = overlay_cam(image, cam1)
    overlay2 = overlay_cam(image, cam2)

    plt.figure(figsize=(14, 6))

#    plt.subplot(1, 3, 1)
#    plt.imshow(image.numpy().transpose(1, 2, 0))
#    plt.title("Original")
#    plt.axis("off")

    plt.subplot(1, 2, 1)
    plt.imshow(overlay1)
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay2)
    plt.title("Grad-CAM++")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(str(sys.argv[2]), format='png', dpi=300)
    plt.close()

# -----------------------------------------------------------
# Load the saved model, for example, 'best_model_early_stopping.pth'
# -----------------------------------------------------------
model = DenseNet161(num_classes=1).to(DEVICE)
model = load_model(model, 'model/best_model_early_stopping.pth')

# -----------------------------------------------------------
# Run Grad-CAM and Grad-CAM++ on a single image (Provide image path)
# -----------------------------------------------------------
image_path = str(sys.argv[1])  # Change this to the image path
show_gradcams(model, image_path)

