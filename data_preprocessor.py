import json
import os
from collections import defaultdict
from PIL import Image
import torch
from torchvision.datasets import CocoDetection

def coco_to_multilabel(data_dir, annotation_file):
    """
    Convert COCO annotations to a multi-label format.
    Args:
        data_dir (str): Directory containing the images.
        annotation_file (str): COCO JSON annotation file path.
    Returns:
        list of tuples: Each tuple contains (image_path, multi_label), where multi_label is a tensor of zeros and ones.
    """
    # Load COCO annotations
    coco = CocoDetection(root=data_dir, annFile=annotation_file)

    # Initialize label dictionary
    label_dict = {}
    for idx, ann in enumerate(coco.coco.cats):
        label_dict[ann] = idx

    # Initialize data list
    data = []
    for img_id in coco.coco.imgs:
        img_info = coco.coco.loadImgs(img_id)[0]
        annotation_ids = coco.coco.getAnnIds(imgIds=img_info['id'])
        annotations = coco.coco.loadAnns(annotation_ids)

        # Create a multi-label vector
        multi_label = torch.zeros(len(label_dict))
        for ann in annotations:
            cat_id = ann['category_id']
            if cat_id in label_dict:
                multi_label[label_dict[cat_id]] = 1

        img_path = os.path.join(data_dir, img_info['file_name'])
        data.append((img_path, multi_label))

    return data

# Example usage
if __name__ == "__main__":
    data_dir = 'path/to/your/images'
    annotation_file = 'path/to/your/annotations/instances.json'
    processed_data = coco_to_multilabel(data_dir, annotation_file)
    print(f"Processed {len(processed_data)} images.")
