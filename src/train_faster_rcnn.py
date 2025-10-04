import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm.notebook import tqdm
import os
from PIL import Image
import pandas as pd


class CocoDataset(Dataset):
    def __init__(self, root, annFile, transforms=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]['file_name']

        # Load image
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        # Bounding boxes
        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in anns:
            xmin, ymin, w, h = ann['bbox']
            boxes.append([xmin, ymin, xmin + w, ymin + h])
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)


def get_dataloaders(train_root, train_ann, val_root, val_ann, test_root, test_ann, batch_size=4):
    transform = T.Compose([T.ToTensor()])

    train_dataset = CocoDataset(train_root, train_ann, transforms=transform)
    val_dataset = CocoDataset(val_root, val_ann, transforms=transform)
    test_dataset = CocoDataset(test_root, test_ann, transforms=transform)

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train(model, train_loader, device, num_epochs=10, lr=0.005):
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        print(f"Epoch {epoch+1} done, Avg Loss: {epoch_loss/len(train_loader):.4f}")

    return model


def evaluate_and_save(model, data_loader, annFile, device, csv_path="metrics.csv"):
    model.eval()
    results = []
    img_ids = []

    coco_gt = COCO(annFile)

    with torch.no_grad():
        for imgs, targets in data_loader:
            imgs = [img.to(device) for img in imgs]
            outputs = model(imgs)

            for target, output in zip(targets, outputs):
                image_id = int(target["image_id"].item())
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    xmin, ymin, xmax, ymax = box
                    w, h = xmax - xmin, ymax - ymin
                    results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [xmin, ymin, w, h],
                        "score": float(score)
                    })
                img_ids.append(image_id)

    if len(results) == 0:
        print("Không có detection nào được tạo!")
        return

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP_50_95 = coco_eval.stats[0]
    mAP_50 = coco_eval.stats[1]

    df = pd.DataFrame([{
        "mAP50-95": mAP_50_95,
        "mAP50": mAP_50
    }])
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")
    return df


def save_model(model, path="models/fasterrcnn.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f" Model saved to {path}")


def main():
    train_root = "dataset_coco_format/train"
    train_ann = os.path.join(train_root, "_annotations.coco.json")
    val_root = "dataset_coco_format/valid"
    val_ann = os.path.join(val_root, "_annotations.coco.json")
    test_root = "dataset_coco_format/test"
    test_ann = os.path.join(test_root, "_annotations.coco.json")

    num_classes = 3 
    num_epochs = 10
    batch_size = 4

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(train_root, train_ann, val_root, val_ann,test_root, test_ann, batch_size=batch_size)

    # Model
    model = get_model(num_classes)

    # Train
    model = train(model, train_loader, device, num_epochs=num_epochs)

    # Save
    save_model(model, "models/fasterrcnn.pth")

    # Evaluate & Save metrics
    evaluate_and_save(model, test_loader, test_ann, device, csv_path="models/metrics.csv")


if __name__ == "__main__":
    main()
