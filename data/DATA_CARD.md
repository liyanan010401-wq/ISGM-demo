# Synthetic demo data card

This release contains only self-generated, nonmedical geometric data. The names of findings are symbolic tokens, not observations of real diseases. No images, reports, identifiers, or annotations from MIMIC-CXR, MS-CXR, or real patients are included.

- Generator: src/data.py; NumPy seed 4404.
- Sample: 15 grayscale 8x8 arrays, one synthetic examination and two atomic clauses per synthetic identifier.
- Split: identifiers 0-9 train; 10-11 reserved development; 12-14 test. All identifiers are invented.
- Each clause preserves its source examination/clause and the half-open character span of the original string.
- Side codes: 0=left, 1=right. Assertion codes: 0=present, 1=absent, 2=uncertain. The implementation also reserves 3=unmentioned and masks it from assertion supervision.
- Rebuild from the repository root: python -m src.data. The public generator uses LF newlines to make fixture hashes portable.
- These examples test software interfaces only. They cannot support medical or manuscript performance claims.

## Original Chinese data documentation

# 自制非医疗测试数据卡
`src/data.py`种子4404生成15个8×8灰度几何阵列和规则事实。patient0–9训练，10–11开发保留，12–14测试，各一exam两原子clause。字段包含patient/exam/clause、不可变原字符串offset、finding/location/side/assertion。图块亮区及词语只是接口合成符号，不代表真实疾病或诊断规律。像素无量纲；side0/1=left/right，assertion0/1/2=present/absent/uncertain，模型还保留unmentioned3语义。未提供数据集原图、病例或受限报告。
所有数据为本项目自行制作，可修改使用；无患者隐私，无外部数据许可。JSON可字节确定性重建，run记录seed与SHA256。训练使用源train标签，测试预测不接收reference clauses。不得把样例文本、门控概率或loss误称真实临床证据。
