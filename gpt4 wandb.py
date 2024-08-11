import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import wandb  # Import wandb
from transformers import BertTokenizer

# Login with the API KEY
wandb.login(key="ab35ea8191eba471c2b58a844910531625b00550")

# Initialize wandb
wandb.init(project="Untitled10", entity="mblogge785-work")

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define configuration for YOLOv3
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
                x = torch.cat([x, route_connections.pop()], dim=1)
        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(CNN_Block(
                    in_channels, out_channels,
                    kernel_size=kernel_size, stride=stride,
                    padding=1 if kernel_size == 3 else 0
                ))
                in_channels = out_channels
            elif isinstance(module, list):
                layers.append(Residual_Block(in_channels, num_repeats=module[1]))
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

class YOLODataset(Dataset):
    def __init__(self, csv_file, ImgDir, LableDir, anchors,
                 ImageSize=416, sp=[13,26,52], cp=20, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.ImgDir = ImgDir
        self.LableDir = LableDir
        self.transform = transform
        self.sp = sp
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.cp = cp
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.LableDir, self.annotations.iloc[index, 1])
        boxx = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        img_path = os.path.join(self.ImgDir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        targets = [torch.zeros((self.num_anchors // 3, sp, sp, 6)) for sp in self.sp]
        for box in boxx:
            iou_anchors = WeidthHeight(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False, False, False]
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                sp = self.sp[scale_idx]
                i, j = int(sp*y), int(sp*x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = sp*x - j, sp*y - i
                    width_cell, height_cell = width*sp, height*sp
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1
        return image, tuple(targets)

class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0
        no_object_loss = self.bce((predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]))
        anchors = anchors.reshape(1,3,1,1,2)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        ious = InterctionOverUnion(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.bce((predictions[..., 0:1][obj]), (ious * target[..., 0:1][obj]))
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
        target[..., 3:5] = torch.log((1e-6 + target[..., 3:5] / anchors))
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])
        class_loss = self.entropy((predictions[..., 5:][obj]), (target[..., 5][obj].long()))
        return (self.lambda_box * box_loss + self.lambda_obj * object_loss + self.lambda_noobj * no_object_loss + self.lambda_class * class_loss)

# Utility functions
def WeidthHeight(boxa, boxb):
    intersection = torch.min(boxa[..., 0], boxb[..., 0]) * torch.min(boxa[..., 1], boxb[..., 1])
    union = (boxa[..., 0] * boxa[..., 1] + boxb[..., 0] * boxb[..., 1] - intersection)
    return intersection / union

def non_max_suppression(boxx, iou_threshold, threshold, box_format="corners"):
    assert type(boxx) == list
    boxx = [box for box in boxx if box[1] > threshold]
    boxx = sorted(boxx, key=lambda x: x[1], reverse=True)
    boxx_after_nms = []
    while boxx:
        chosen_box = boxx.pop(0)
        boxx = [box for box in boxx if box[0] != chosen_box[0] or InterctionOverUnion(
            torch.tensor(chosen_box[2:]), torch.tensor(box[2:]), box_format=box_format) < iou_threshold]
        boxx_after_nms.append(chosen_box)
    return boxx_after_nms

def InterctionOverUnion(PredsBox, lableBox, box_format="midpoint"):
    if box_format == "midpoint":
        box1_a1 = PredsBox[..., 0:1] - PredsBox[..., 2:3] / 2
        box1_b1 = PredsBox[..., 1:2] - PredsBox[..., 3:4] / 2
        box1_a2 = PredsBox[..., 0:1] + PredsBox[..., 2:3] / 2
        box1_b2 = PredsBox[..., 1:2] + PredsBox[..., 3:4] / 2
        box2_a1 = lableBox[..., 0:1] - lableBox[..., 2:3] / 2
        box2_y1 = lableBox[..., 1:2] - lableBox[..., 3:4] / 2
        box2_a2 = lableBox[..., 0:1] + lableBox[..., 2:3] / 2
        box2_y2 = lableBox[..., 1:2] + lableBox[..., 3:4] / 2
    if box_format == "corners":
        box1_a1 = PredsBox[..., 0:1]
        box1_b1 = PredsBox[..., 1:2]
        box1_a2 = PredsBox[..., 2:3]
        box1_b2 = PredsBox[..., 3:4]
        box2_a1 = lableBox[..., 0:1]
        box2_y1 = lableBox[..., 1:2]
        box2_a2 = lableBox[..., 2:3]
        box2_y2 = lableBox[..., 3:4]
    x1 = torch.max(box1_a1, box2_a1)
    y1 = torch.max(box1_b1, box2_y1)
    x2 = torch.min(box1_a2, box2_a2)
    y2 = torch.min(box1_b2, box2_y2)
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    box1_area = (box1_a2 - box1_a1) * (box1_b2 - box1_b1)
    box2_area = (box2_a2 - box2_a1) * (box2_y2 - box2_y1)
    return intersection / (box1_area + box2_area - intersection)

def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", NumClasses=4):
    average_precisions = []
    epsilon = 1e-6
    for c in range(NumClasses):
        detections = []
        ground_truths = []
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)
        amount_boxx = Counter([gt[0] for gt in ground_truths])
        for key, val in amount_boxx.items():
            amount_boxx[key] = torch.zeros(val)
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_boxx = len(ground_truths)
        if total_true_boxx == 0:
            continue
        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            num_gts = len(ground_truth_img)
            best_iou = 0
            for idx, gt in enumerate(ground_truth_img):
                iou = InterctionOverUnion(torch.tensor(detection[3:]), torch.tensor(gt[3:]), box_format=box_format)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            if best_iou > iou_threshold:
                if amount_boxx[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_boxx[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_boxx + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))
    return sum(average_precisions) / len(average_precisions)

def get_evaluation_boxx(loader, model, iou_threshold, anchors, threshold, box_format="midpoint"):
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, labels) in enumerate(loader):
        x = x.float().to(DEVICE)
        with torch.no_grad():
            predictions = model(x)
        batch_size = x.shape[0]
        boxx = [[] for _ in range(batch_size)]
        for i in range(3):
            sp = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(DEVICE) * sp
            boxes_scale_i = cells_to_boxx(predictions[i], anchor, sp=sp, is_preds=True)
            for idx, box in enumerate(boxes_scale_i):
                boxx[idx] += box
        true_boxx = cells_to_boxx(labels[2], anchor, sp=sp, is_preds=False)
        for idx in range(batch_size):
            nms_boxes = non_max_suppression(boxx[idx], iou_threshold=iou_threshold, threshold=threshold, box_format=box_format)
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)
            for box in true_boxx[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)
            train_idx += 1
    model.train()
    return all_pred_boxes, all_true_boxes

def cells_to_boxx(predictions, anchors, sp, is_preds=True):
    SizeOfBatch = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]
    cell_indices = (
        torch.arange(sp)
        .repeat(predictions.shape[0], 3, sp, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / sp * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / sp * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / sp * box_predictions[..., 2:4]
    converted_boxx = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(SizeOfBatch, num_anchors * sp * sp, 6)
    return converted_boxx.tolist()

def plot_image(image, boxes):
    cmap = plt.get_cmap("tab20b")
    class_labels = AllClacess
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape
    fig, ax = plt.subplots(1)
    ax.imshow(im)
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        UpperLeft_x = box[0] - box[2] / 2
        UpperLeft_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (UpperLeft_x * width, UpperLeft_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        ax.add_patch(rect)
        plt.text(
            UpperLeft_x * width,
            UpperLeft_y * height,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )
    plt.show()

# Instantiate the model
model = YOLOv3(NumClasses=NumClasses).to(DEVICE)

# Compile the model
optimizer = torch.optim.Adam(model.parameters(), lr=RateOfLearning)
loss_fn = YoloLoss()

# Scaler
scaler = torch.cuda.amp.GradScaler()

# Transform
transform = transforms.Compose([transforms.Resize((416, 416)), transforms.ToTensor()])

# Train-Test Loader
def get_loaders(train_csv_path, test_csv_path):
    train_dataset = YOLODataset(train_csv_path, transform=transform, sp=[ImageSize // 32, ImageSize // 16, ImageSize // 8],
                                ImgDir=DirImage, LableDir=DirLable, anchors=ANCHORS)
    test_dataset = YOLODataset(test_csv_path, transform=transform, sp=[ImageSize // 32, ImageSize // 16, ImageSize // 8],
                               ImgDir=DirImage, LableDir=DirLable, anchors=ANCHORS)
    train_loader = DataLoader(dataset=train_dataset, batch_size=SizeOfBatch, shuffle=True, drop_last=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=SizeOfBatch, shuffle=False, drop_last=False)
    return train_loader, test_loader

train_loader, test_loader = get_loaders(
    train_csv_path='/kaggle/input/pascalvoc-yolo/8examples.csv', test_csv_path='/kaggle/input/pascalvoc-yolo/8examples.csv'
)

# Anchors
scaled_anchors = (
    torch.tensor(ANCHORS) * torch.tensor([13,26,52]).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
).to(DEVICE)

# Training loop
history_loss = []

for epoch in tqdm(range(epochsno), desc="Epochs"):
    model.train()
    losses = []
    start_time = time.time()
    for batch_idx, (x,y) in enumerate(train_loader):
        x = x.to(DEVICE)
        y0, y1, y2 = (y[0].to(DEVICE), y[1].to(DEVICE), y[2].to(DEVICE))
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (loss_fn(out[0], y0, scaled_anchors[0]) +
                    loss_fn(out[1], y1, scaled_anchors[1]) +
                    loss_fn(out[2], y2, scaled_anchors[2]))
        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    end_time = time.time()
    epoch_duration = end_time - start_time
    history_loss.append(sum(losses)/len(losses))
    precision, recall, f1 = compute_metrics(pred_boxes, true_boxes)
    if (epoch+1) % 10 == 0:
        tqdm.write(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds")
        print(f"Epoch [{epoch+1}/{epochsno}], Loss: {sum(losses)/len(losses):.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "loss": sum(losses)/len(losses),
            "epoch_duration": epoch_duration,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
        torch.save(model.state_dict(), f'/kaggle/working/Yolov3_epoch{epoch+1}.pth')

wandb.finish()
