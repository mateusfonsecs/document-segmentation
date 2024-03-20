from ultralytics import YOLO
import cv2
import torch
import numpy as np

def load_model(path, image):
    model = YOLO(path)  # load a custom model
    results = model.predict(image, conf =0.6, save_conf=True)  # predict on an image

    return results

def show_results(results):
    for result in results:
        result.show()
        result.save('result.jpg')

def show_masks_with_cv2(results):
    merged_mask = None
    conf = 0
    for result in results:
        masks = result.masks.data
        boxes = result.boxes.data

        clss = boxes[:, 5]
        unique_classes = torch.unique(clss)
        if result.boxes.conf[0].item() > conf:
            conf = result.boxes.conf[0].item()
            merged_mask = None
            for cls in unique_classes:
                class_indices = torch.where(clss == cls)
                class_masks = masks[class_indices]
                class_mask = torch.any(class_masks, dim=0).int() * 255
                if merged_mask is None:
                    merged_mask = class_mask
                else:
                    merged_mask = torch.max(merged_mask, class_mask)

    merged_mask_np = merged_mask.cpu().numpy().astype(np.uint8) 
    resized_mask = cv2.resize(merged_mask_np, (640, 720))

    cv2.imshow('Resized Mask', resized_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    dir_image = "/home/dg/aleatorio/ufv/document-detection/faturas/dme1.png"
    image = cv2.imread(dir_image)
    dir_model = '/home/dg/aleatorio/ufv/document-detection/model_teste_6/best.pt'
    results = load_model(dir_model, image)
    show_results(results)
    show_masks_with_cv2(results)