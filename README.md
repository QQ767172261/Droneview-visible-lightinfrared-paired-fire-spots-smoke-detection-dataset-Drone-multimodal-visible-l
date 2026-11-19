无人机视角可见光-红外配对的火点，烟雾检测数据集![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b46fe7c83a10464abb53a0d9d692f136.png)

共1200余对可见光-红外图像（640*512分辨率），拍摄森林火情，采用yolo格式标注![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/89213b1179e24db3a112832fdc1eda51.png)
表格

| 项目 | 详情 |
| --- | --- |
| 数据集名称 | 无人机视角可见光 - 红外配对火点、烟雾检测数据集 |
| 图像数量 | 1200 余对 |
| 图像分辨率 | 640*512 |
| 数据形式 | 可见光图像 + 红外图像配对 |
| 标注格式 | YOLO |
| 场景类型 | 森林火情 |

### 应用价值

用于无人机森林火灾巡检、火点定位、烟雾识别、早期火情预警，适合多模态融合目标检测算法研发，提升复杂背景下小火点、烟雾遮挡、暗光红外目标的检出能力。

```
# YOLOv11n 训练脚本
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov11n.pt")
    model.train(
        data="fire_smoke.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0
    )
```

```
# 多模态推理入口示例
import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
vis = cv2.imread("visible.jpg")
ir = cv2.imread("ir.jpg")
results = model.predict(source=[vis, ir], conf=0.35, iou=0.45)
```

