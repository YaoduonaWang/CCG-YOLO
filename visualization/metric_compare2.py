import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#
csv_files = [
    '/Users/diona/Desktop/plot/ccgyolo.csv',
    '/Users/diona/Desktop/plot/yolov11.csv'
]
labels = ['CCG-YOLO', 'YOLOv11']

# Read data and clean up header spaces
dfs = []
for path in csv_files:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()  # 去除表头空格
    dfs.append(df)

# 
loss_metrics = [
    'train/box_loss',
    'train/dfl_loss',
    'train/cls_loss',
    'val/box_loss',
    'val/dfl_loss',
    'val/cls_loss'
]
titles = [
    'Train Box Loss', 'Train DFL Loss', 'Train CLS Loss',
    'Val Box Loss', 'Val DFL Loss', 'Val CLS Loss'
]

plt.figure(figsize=(18, 8))
for idx, (metric, title) in enumerate(zip(loss_metrics, titles), 1):
    plt.subplot(2, 3, idx)
    for df, label in zip(dfs, labels):
        y = df[metric].astype(np.float32).replace(np.inf, np.nan).interpolate()
        plt.plot(y, label=label)
    plt.xlabel('epoch')
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig('loss_compare.png', dpi=200)
plt.show()
print('Loss对比曲线已保存为 loss_compare.png')
