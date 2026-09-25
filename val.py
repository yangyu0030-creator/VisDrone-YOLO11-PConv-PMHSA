from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/visdrone_pconv_pmhsa_100/weights/best.pt')
    metrics = model.val(
        data='ultralytics/cfg/datasets/VisDrone2019-DET.yaml',
        split='val',
        imgsz=1024,
        batch=4,
        device='cuda',
        workers=4,
        plots=True,
    )
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"P:        {metrics.box.mp:.4f}")
    print(f"R:        {metrics.box.mr:.4f}")