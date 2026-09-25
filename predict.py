from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'C:\Users\yy\Desktop\ultralytics-main\runs\detect\visdrone_pconv_pmhsa_100\weights\best.pt')

    results = model.predict(
        source='datasets/VisDrone2019-DET/images/val/9999990_00000_d_0000007.jpg',
        imgsz=1024,
        conf=0.25,
        save=True,
        device='cuda',
        show=True,
        max_det=300,      # 每张图最多检测 300 个目标
    )