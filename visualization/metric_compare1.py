import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 
csv_files = [
    '/Users/diona/Desktop/plot/ccgyolo.csv',
    '/Users/diona/Desktop/plot/yolov11.csv'
]
labels = ['CCG-YOLO', 'YOLOv11']

#Read data and clean up header spaces
dfs = []
for path in csv_files:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    dfs.append(df)

#Draw comparison curves of precision, recall, mAP50, and mAP50-95
plt.figure(figsize=(12, 10))

# precision
plt.subplot(2, 2, 1)
for df, label in zip(dfs, labels):
    y = df['metrics/precision(B)'].astype(np.float32).replace(np.inf, np.nan).interpolate()
    plt.plot(y, label=label)
plt.xlabel('epoch')
plt.ylabel('precision')
plt.title('Precision')
plt.legend()
plt.grid(True)

# recall
plt.subplot(2, 2, 2)
for df, label in zip(dfs, labels):
    y = df['metrics/recall(B)'].astype(np.float32).replace(np.inf, np.nan).interpolate()
    plt.plot(y, label=label)
plt.xlabel('epoch')
plt.ylabel('recall')
plt.title('Recall')
plt.legend()
plt.grid(True)

# mAP50
plt.subplot(2, 2, 3)
for df, label in zip(dfs, labels):
    y = df['metrics/mAP50(B)'].astype(np.float32).replace(np.inf, np.nan).interpolate()
    plt.plot(y, label=label)
plt.xlabel('epoch')
plt.ylabel('mAP_0.5')
plt.title('mAP_0.5')
plt.legend()
plt.grid(True)

# mAP50-95
plt.subplot(2, 2, 4)
for df, label in zip(dfs, labels):
    y = df['metrics/mAP50-95(B)'].astype(np.float32).replace(np.inf, np.nan).interpolate()
    plt.plot(y, label=label)
plt.xlabel('epoch')
plt.ylabel('mAP_0.5:0.95')
plt.title('mAP_0.5:0.95')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('metric_compare.png', dpi=200)
plt.show()
print('metric_compare.png')
