import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import os
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import wandb
from transformers import BertTokenizer

# Login to wandb
wandb.login(key="ab35ea8191eba471c2b58a844910531625b00550")
wandb.init(project="Untitled10", entity="mblogge785-work")  # Replace with your wandb username

# Define the YOLOv3 model configuration
config = [
    (32, 3, 1),
    (128, 3, 1),
    (64, 3, 2),
    ["list", 1],
    (128, 3, 2),
    ["list", 2],
    (256, 3, 2),
    ["list", 8],
    (512, 3, 2),
    ["list", 8],
    (1024, 3, 2),
    ["list", 4],
    (512, 1, 1),
    (1024, 3, 1),
    "sp",
    (256, 1, 1),
    "up",
    (256, 1, 1),
    (512, 3, 1),
    "sp",
    (128, 1, 1),
    "up",
    (128, 1, 1),
    (256, 3, 1),
    "sp",
]

# Define the CNN block
class CNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super(CNN_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act
    
    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)

# Define the Residual block
class Residual_Block(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super(Residual_Block, self).__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNN_Block(channels, channels//2, kernel_size=1),
                    CNN_Block(channels//2, channels, kernel_size=3, padding=1)
                )
            ]
        self.use_residual = use_residual
        self.num_repeats = num_repeats
    
    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)
        return x

# Define the Prediction Scale
class Prediction_Scale(nn.Module):
    def __init__(self, in_channels, NumClasses):
        super(Prediction_Scale, self).__init__()
        self.pred = nn.Sequential(
            CNN_Block(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNN_Block(2 * in_channels, (NumClasses + 5) * 3, bn_act=False, kernel_size=1),
        )
        self.NumClasses = NumClasses
    
    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.NumClasses + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )

# Define the YOLOv3 model
class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, NumClasses=20):
        super(YOLOv3, self).__init__()
        self.NumClasses = NumClasses
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()
    
    def forward(self, x):
        outputs = []
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, Prediction_Scale):
                outputs.append(layer(x))
                continue
            x = layer(x)
            if isinstance(layer, Residual_Block) and layer.num_repeats == 8:
                route_connections.append(x)
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
        return outputs
    
    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(CNN_Block(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1 if kernel_size == 3 else 0
                ))
                in_channels = out_channels
            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(Residual_Block(in_channels, num_repeats=num_repeats))
            elif isinstance(module, str):
                if module == "sp":
                    layers += [
                        Residual_Block(in_channels, use_residual=False, num_repeats=1),
                        CNN_Block(in_channels, in_channels//2, kernel_size=1),
                        Prediction_Scale(in_channels//2, NumClasses=self.NumClasses)
                    ]
                    in_channels = in_channels // 2
                elif module == "up":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3
        return layers

# Define utility functions
def WeidthHeight(boxa, boxb):
    intersection = torch.min(boxa[..., 0], boxb[..., 0]) * torch.min(
        boxa[..., 1], boxb[..., 1]
    )
    union = (
        boxa[..., 0] * boxa[..., 1] + boxb[..., 0] * boxb[..., 1] - intersection
    )
    return intersection / union

def calculate_metrics(all_preds, all_labels, num_classes):
    all_preds = torch.cat([torch.flatten(p) for p in all_preds])
    all_labels = torch.cat([torch.flatten(l) for l in all_labels])
    precision, recall, f1_score = [], [], []
    for c in range(num_classes):
        tp = ((all_preds == c) & (all_labels == c)).sum().item()
        fp = ((all_preds == c) & (all_labels != c)).sum().item()
        fn = ((all_preds != c) & (all_labels == c)).sum().item()
        p = tp / (tp + fp + 1e-6)
        r = tp / (tp + fn + 1e-6)
        f1 = 2 * (p * r) / (p + r + 1e-6)
        precision.append(p)
        recall.append(r)
        f1_score.append(f1)
    return precision, recall, f1_score

def train(model, train_loader, optimizer, criterion, DEVICE):
    model.train()
    total_loss = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(DEVICE)
        targets = [target.to(DEVICE) for target in targets]
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")
    return total_loss / len(train_loader)

def evaluate(model, test_loader, criterion, DEVICE):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            data = data.to(DEVICE)
            targets = [target.to(DEVICE) for target in targets]
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            for i in range(len(outputs)):
                preds = outputs[i]
                labels = targets[i]
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
    return total_loss / len(test_loader), all_preds, all_labels

def non_max_suppression(boxx, iou_threshold, threshold, box_format="corners"):
    assert type(boxx) == list
    boxx = [box for box in boxx if box[1] > threshold]
    boxx = sorted(boxx, key=lambda x: x[1], reverse=True)
    boxx_after_nms = []
    while boxx:
        chosen_box = boxx.pop(0)
        boxx = [
            box
            for box in boxx
            if box[0] != chosen_box[0]
            or InterctionOverUnion(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]
        boxx_after_nms.append(chosen_box)
    return boxx_after_nms

def InterctionOverUnion(PredsBox, TargetBox, box_format="corners"):
    if box_format == "midpoint":
        box1_x1 = PredsBox[..., 0:1] - PredsBox[..., 2:3] / 2
        box1_y1 = PredsBox[..., 1:2] - PredsBox[..., 3:4] / 2
        box1_x2 = PredsBox[..., 0:1] + PredsBox[..., 2:3] / 2
        box1_y2 = PredsBox[..., 1:2] + PredsBox[..., 3:4] / 2
        box2_x1 = TargetBox[..., 0:1] - TargetBox[..., 2:3] / 2
        box2_y1 = TargetBox[..., 1:2] - TargetBox[..., 3:4] / 2
        box2_x2 = TargetBox[..., 0:1] + TargetBox[..., 2:3] / 2
        box2_y2 = TargetBox[..., 1:2] + TargetBox[..., 3:4] / 2
    elif box_format == "corners":
        box1_x1 = PredsBox[..., 0:1]
        box1_y1 = PredsBox[..., 1:2]
        box1_x2 = PredsBox[..., 2:3]
        box1_y2 = PredsBox[..., 3:4]
        box2_x1 = TargetBox[..., 0:1]
        box2_y1 = TargetBox[..., 1:2]
        box2_x2 = TargetBox[..., 2:3]
        box2_y2 = TargetBox[..., 3:4]
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    inter_area = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    return inter_area / (box1_area + box2_area - inter_area + 1e-6)

class CustomDataset(Dataset):
    def __init__(self, csv_file, image_dir, label_dir, anchors, image_size=416, S=[13, 26, 52], C=20, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        image_path = os.path.join(self.image_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(image_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        for box in bboxes:
            iou_anchors = WeidthHeight(torch.tensor(box[2:4]), self.anchors)
            iou_sorted, indices = iou_anchors.sort(descending=True)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3
            for iou_idx in indices:
                scale_idx = iou_idx // self.num_anchors_per_scale
                anchor_on_scale = iou_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True
                elif not anchor_taken and iou_idx == 0:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i
                    width_cell, height_cell = width * S, height * S
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)

        return image, tuple(targets)

def get_loaders(train_csv_path, test_csv_path):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ]
    )
    train_dataset = CustomDataset(
        csv_file=train_csv_path,
        image_dir=DirImage,
        label_dir=DirLable,
        anchors=ANCHORS,
        transform=transform,
    )
    test_dataset = CustomDataset(
        csv_file=test_csv_path,
        image_dir=DirImage,
        label_dir=DirLable,
        anchors=ANCHORS,
        transform=transform,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=SizeOfBatch,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=SizeOfBatch,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, test_loader

# Initialize the model
model = YOLOv3(NumClasses=20).to(DEVICE)

# Optimizer and loss criterion
optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Initialize data loaders
DirImage = "path_to_images"  # Replace with your image directory path
DirLable = "path_to_labels"  # Replace with your labels directory path
train_csv_path = "path_to_train_csv"  # Replace with your train CSV path
test_csv_path = "path_to_test_csv"  # Replace with your test CSV path
SizeOfBatch = 16
ANCHORS = [
    [(10, 13), (16, 30), (33, 23)],  # Scale 1
    [(30, 61), (62, 45), (59, 119)],  # Scale 2
    [(116, 90), (156, 198), (373, 326)],  # Scale 3
]
train_loader, test_loader = get_loaders(train_csv_path, test_csv_path)

# wandb configuration
wandb.config.update({
    "learning_rate": 1e-3,
    "epochs": 30,
    "batch_size": 64,
    "encoder_embedding_dim": 512,
    "decoder_embedding_dim": 512,
    "hidden_dim": 1024,
    "num_layers": 3,
    "encoder_dropout": 0.5,
    "decoder_dropout": 0.5,
    "num_classes": 20
})

# Main training loop
for epoch in range(wandb.config.epochs):
    train_loss = train(model, train_loader, optimizer, criterion, DEVICE)
    val_loss, all_preds, all_labels = evaluate(model, test_loader, criterion, DEVICE)
    
    precision, recall, f1_score = calculate_metrics(all_preds, all_labels, wandb.config.num_classes)
    
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    })
    
    print(f"Epoch [{epoch+1}/{wandb.config.epochs}], Train Loss: {train_loss}, Val Loss: {val_loss}")
