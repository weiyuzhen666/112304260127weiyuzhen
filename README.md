# 交通标志检测实验

本项目使用 YOLOv8 模型完成交通标志目标检测任务。

## 项目结构

```
├── data.yaml          # 数据集配置文件
├── train_gpu.py       # GPU训练脚本
├── infer.py           # 推理脚本
├── submission.csv     # 提交文件
├── test/              # 测试集目录
│   └── images/        # 测试图片
├── runs/              # 训练结果目录
│   └── detect/        # 检测相关结果
└── README.md          # 项目说明
```

## 环境要求

- Python 3.9+
- PyTorch 2.6.0+cu124
- ultralytics 8.0.228
- NVIDIA GPU（推荐）

## 安装依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics
```

## 训练模型

```bash
python train_gpu.py
```

训练参数：
- epochs: 50
- batch size: 4
- imgsz: 640
- optimizer: SGD
- lr0: 0.01

## 生成提交文件

```bash
python infer.py --model runs/detect/runs/train/weights/best.pt \
                --test-dir test/images \
                --output submission.csv \
                --conf 0.01 \
                --device cpu
```

## 实验结果

- 验证集 mAP50: 0.94875
- 比赛提交分数: 0.722832
- 排行榜名次: 第32名

## 数据集

数据集包含15种交通标志类别：
1. Green Light
2. Red Light
3. Yellow Light
4. Speed limit 30
5. Speed limit 40
6. Speed limit 50
7. Speed limit 60
8. Speed limit 70
9. Speed limit 80
10. No entry
11. No left turn
12. No right turn
13. Stop
14. Yield
15. Roundabout

## 实验报告

详细的实验报告请查看：[第四次实验报告模板.md](第四次实验报告模板.md)

## 作者

- 姓名：韦余圳
- 学号：112304260127
