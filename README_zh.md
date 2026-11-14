# ISGM 可运行研究演示
本例训练小型图像/文本对齐、视觉支持门控与关系纠正接口，并生成两轮离散文本。所有图像与事实均为自制合成几何记录，不是胸片、不是医疗报告系统，输出没有临床用途，不能填入论文实验结果。

```powershell
python -m pip install -r requirements.txt
python run_demo.py --output outputs
python -m unittest discover -s tests -v
```
支持任意 cwd 用绝对脚本启动，资源相对脚本位置，output相对调用者位置；固定CPU单线程。`python -m src.data`重建样例。输出真实训练loss、adapter梯度、候选来源/支持概率、初稿及修订文本。无结果筛选，生成错误也保留。

SourceGraph只接受train记录，训练query同患者在建邻接前排除；测试只给图像和排除用patient ID。测试参考只在推理完成后附加到评估输出。MIMIC/报告下载与资质不在本例范围，正式接入不得绕过授权。
