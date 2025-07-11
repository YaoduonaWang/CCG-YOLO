from ultralytics import YOLO

def main():
    # Load the trained model and specify the device
    model = YOLO('/root/awsyolo/runs/detect/Rhododendron_Lundy_V2_NIR_RE_cbam3_00012/weights/best.pt')  # replace with your model path

    # Evaluate the model on the validation split (using CUDA)
    metrics = model.val(
        data='/root/awsyolo/Rhododendron_Lundy_V2_NIR_RE/Rhododendron_Lundy_V2_NIR_RE.yaml',    # path to the dataset configuration file
        split='test',        # which data split to use for evaluation
        imgsz=640,           # image size
        batch=6,             # batch size
        conf=0.25,           # confidence threshold
        iou=0.6,             # IoU threshold
        device='0'           # use (GPU)
    )

if __name__ == "__main__":
    main()
