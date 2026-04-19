# 全景图拼接实验

---

唐雅妍23320143

### 项目结构

```
H1/
├── results/1/                       # 结果文件
├── images/1/                        # 测试文件
├── harris.py                        # 角点检测
├── hog.py                           # HOG 特征提取及匹配
├── sift.py                          # SIFT 特征提取及匹配
├── hog_rotate.py                    # 改进版的 HOG 特征
├── affine.py                        # 图像拼接
└── main.py                          # 程序入口
```

结果文件说明：

* `_hog`表示使用hog特征提取，`_sift`表示使用sift特征提取
* `_selfaffine`表示使用仿射变换，`_selfhom`表示使用透视变换
* `_testhog`使用透视变换+改进hog得到的用于测试

### 快速开始

环境：

* numpy
* cv2
* scipy

```bash
python main.py
```
