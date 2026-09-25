from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/visdrone_pconv_pmhsa/weights/best.pt')
    model.train(
        data='ultralytics/cfg/datasets/VisDrone2019-DET.yaml',
        batch=4,
        epochs=100,
        imgsz=1024,
        workers=4,
        device='cuda',
        cache='disk',
        name='visdrone_pconv_pmhsa_100',
        exist_ok=True,
    )