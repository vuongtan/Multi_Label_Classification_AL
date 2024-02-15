import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import model
import data_preprocessor
import numpy as np
from torchvision.utils import save_image
import os
import torchvision.transforms.functional as TF
import json
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Args:
            data (list of tuples): A list where each tuple is of the form (image_path, label).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Return image, label, and filename
        return image, label, os.path.basename(img_path)

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_batches = len(train_loader)  # Total number of batches in the loader
    for batch_idx, (data, target,filenames) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Calculate the percentage of the epoch that has been completed
        percentage_completed = 100. * (batch_idx + 1) / total_batches

        if batch_idx % 10 == 0:  # Print every 10 batches
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({percentage_completed:.0f}%)]\tLoss: {loss.item():.6f}')

def draw_labels_on_image(image, pred_labels, actual_labels, label_names):
    if isinstance(image, torch.Tensor):
        image = TF.to_pil_image(image)

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    # print(pred_labels)
    # print(actual_labels)

    # Convert label indices to names, adjusting for indexing starting at 1
    pred_label_names = [label_names[i+1] for i, label in enumerate(pred_labels) if label]
    actual_label_names = [label_names[i+1] for i, label in enumerate(actual_labels) if label == 1.0]

    # Formatting text for drawing
    text_pred = 'Pred: ' + ', '.join(pred_label_names) if pred_label_names else 'Pred: None'
    text_actual = 'Actual: ' + ', '.join(actual_label_names) if actual_label_names else 'Actual: None'
    text = text_pred + '\n' + text_actual

    # Assuming 'draw' and 'image' are already defined as in your original snippet
    # Example for drawing the text:
    draw.multiline_text((10, image.height - 30), text, font=font, fill=(255, 255, 255))

    return image


def save_metrics_to_csv(metrics, csv_path="validation_metrics.csv"):
    df = pd.DataFrame([metrics])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, mode='w', header=True, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

def update_best_model(model, current_metrics, best_metrics, model_path="best_model.pth"):
    if current_metrics['accuracy_positive'] > best_metrics.get('accuracy_positive', 0):
        torch.save(model.state_dict(), model_path)
        print(f"Updated best model weights saved to {model_path}")
        return current_metrics  # Update the best metrics
    return best_metrics


def validate(model, device, validation_loader, criterion,epoch,label_names, output_dir="validation_output"):
    model.eval()
    validation_loss = 0
    total_correct = 0
    total_labels = 0
    positive_labels=0
    total_correct_positive=0
    total_images = 0  # Variable to count the total number of images processed
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    with torch.no_grad():
       for batch_idx, (data, target,filenames) in enumerate(validation_loader):
            batch_size = data.size(0)
            total_images += batch_size  # Increment total images processed
            data, target = data.to(device), target.to(device)
            output = model(data)
            validation_loss += criterion(output, target).item()  # Sum up batch loss
            pred = torch.sigmoid(output) > 0.6  # Convert output probabilities to binary predictions
            target_binary = target > 0.6
            positive_labels += target_binary.sum().item()
            total_labels += target.numel()
            total_correct += pred.eq(target.byte()).sum().item()
            total_correct_positive += (pred & target_binary).sum().item()  # Logical AND to count true positives
        
            if batch_idx < 5:  # Save images from the first batch
                for i in range(data.size(0)):
                    img = data[i].cpu()
                    pred_labels = pred[i].cpu().numpy()
                    actual_labels = target[i].cpu().numpy()
                    original_img_name = filenames[i].replace('.png', '').replace('.jpg', '')  # Adjust extension as needed
                    img_with_labels = draw_labels_on_image(img, pred_labels, actual_labels, label_names)
                    img_name = f"epoch_{epoch}_{original_img_name}_pred.png"
                    img_path = os.path.join(output_dir, img_name)
                    img_with_labels.save(img_path)


    validation_loss /= total_images  # Adjust average loss calculation
    accuracy_positive = 100. * total_correct_positive / positive_labels if positive_labels > 0 else 0
    accuracy = 100. * total_correct / total_labels
    print(f'Validation set [{total_images} images]: Average loss: {validation_loss:.4f}, '
          f'Accuracy: {accuracy:.0f}% ({total_correct}/{total_labels})\n'
          f'Accuracy_Positive: {accuracy_positive:.0f}%')
    # Return validation metrics
    return {
        'epoch': epoch,
        'validation_loss': validation_loss,
        'accuracy': accuracy,
        'accuracy_positive': accuracy_positive,
        'total_images': total_images
    }



# Load COCO annotations to get category names
def load_coco_labels(annotation_file):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Creating a dictionary for label index to name mapping
    label_names = {category['id']: category['name'] for category in annotations['categories']}
    return label_names

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])
    # Load and preprocess data
    data_dir_train = '/home/tan/VisDrone/train2017'
    # annotation_file_train = '/content/drive/MyDrive/PPALVuongTan/PPAL/data/coco/annotations/instances_train2017.json'
    annotation_file_train = '/home/tan/VisDrone/coco_557_labeled_1.json'

    processed_data_train = data_preprocessor.coco_to_multilabel(data_dir_train, annotation_file_train)
    dataset_train = CustomDataset(processed_data_train, transform=transform)
    
    # Load and preprocess data
    data_dir_val = '/home/tan/VisDrone/val2017'
    annotation_file_val ='/home/tan/VisDrone/instances_val2017.json'
    processed_data_val = data_preprocessor.coco_to_multilabel(data_dir_val, annotation_file_val)
    dataset_val = CustomDataset(processed_data_val, transform=transform)
    label_names = load_coco_labels(annotation_file_train)
    print(label_names)
    # Define DataLoaders for training and validation
    train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True)
    validation_loader = DataLoader(dataset_val, batch_size=16, shuffle=False)


    # Initialize model
    num_classes = 10  # Set your number of classes
    net = model.MultiClassificationModel(num_classes).to(device)
    
    # Define loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.002)

    # Training and validation loop
    epochs = 200
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        train(net, device, train_loader, optimizer, criterion, epoch)
        metrics = validate(net, device, validation_loader, criterion, epoch,label_names,'output_image')
        save_metrics_to_csv(metrics)
        best_metrics = update_best_model(net, metrics, best_metrics, model_path="best_model.pth")
if __name__ == "__main__":
    main()
