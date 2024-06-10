# nlp_spellcorrection
HUST NLP homework.

# 项目文件说明
- ans.txt: 标准答案
- correction.py: UI中用于纠正拼写错误的核心模块，大部分代码与main.txt相同
- count_1edit.txt：生成嘈杂信道模型所使用的的1编辑距离的词典
- edit_distance.py: 符合一定编辑距离的候选词生成器
- eval.py: 评估模块
- eval_enhanced.py: 可以输出详细信息的评估模块
- main.py: 主模块，输出答案
- ngram.py: 生成ngram模型
- result.txt: 由main.py输出的答案
- testdata.txt: 测试集
- UI.py: UI用户界面核心
- vocab.txt: 词典，用于纠正非词错误

# 如何运行？
运行main.py可以获得答案result.txt，之后运行eval.py可以获得评估结果。

运行UI.py可以进入用户界面，输入句子，点击纠错按钮可以获得纠错结果。
