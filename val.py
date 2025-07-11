from ultralytics import YOLO

model = YOLO('/root/awsyolo/runs/detect/Rhododendron_Lundy_V2_NIR_RE_cbam1_00013/weights/best.pt')
metrics = model.val(data='/root/awsyolo/Rhododendron_Lundy_V2_NIR_RE/Rhododendron_Lundy_V2_NIR_RE.yaml',batch=12,half=True)

