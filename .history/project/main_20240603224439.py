# %%
import nltk
import re
import numpy as np
from ngram import generate_ngram_models
from ngram import calculate_smoothed_probability
from nltk.corpus import reuters
from tqdm import tqdm
from edit_distance import CandidatesGenerator

# %% [markdown]
# # 读取字典库、语料库

# %%
vocab = {line.rstrip() for line in open("vocab.txt")}
# 用set来存储不用list是因为查找的时候set时间复杂度是O(1),List是O(n)

# %%
nltk.download("reuters")
nltk.download("punkt")

# %% [markdown]
# # 据语料库生成ngram模型

# %%
categories = reuters.categories()  # 路透社语料库的类别
corpus = reuters.sents(categories=categories)  # sents()指定分类中的句子

# 构建语言模型：bigram
term_count, bigram_count = generate_ngram_models(corpus, 2)

# %% [markdown]
# # 从count_1edit.txt获取channel probability

# %%
# 用户打错的概率统计 - channel probability
# 创建一个字典来存储channel probabilities
channel_prob = {}
total_errors = 0

i = 0
# 解析错误数据并计算总错误次数
for line in open("count_1edit.txt"):
    i += 1
    # Step1:解析数据
    # 正则表达式找到错误次数
    count = re.findall(r"\d+", line)[-1]

    # 从末尾剥离数字
    line = line.replace(count, "")
    # 剥离制表符
    if "\t" in line:
        line = line.replace("\t", "")
    # 判断空格在不在后段
    first, last = line.split("|")

    if " " in last:
        # 去除多个空格为一个
        if re.match(r" {2,}", line):
            multi_spaces = re.findall(r" {2,}", line)
            for space in multi_spaces:
                line = line.replace(space, " ")

    # 正常情况
    correct, mistake = line.split("|")

    count = int(count)
    # Step2:计算错误次数
    if correct not in channel_prob:
        channel_prob[correct] = {}

    channel_prob[correct][mistake] = count
    total_errors += count

# 计算每种错误的概率
for correct in channel_prob:
    for mistake in channel_prob[correct]:
        channel_prob[correct][mistake] /= total_errors

# %% [markdown]
# # 测试

# %%
# 创建生成者
CG = CandidatesGenerator(vocab=vocab)
print(CG.generate_candidates("HKES"))
# 设置最大编辑距离
max_distance = 1

# %% [markdown]
# ### 未实现的用例
# 241 246


# %%
def count_lines(filename):
    with open(filename, "r") as f:
        return sum(1 for line in f)


# 计算文件的行数
file_path = "testdata.txt"
total_lines = count_lines(file_path)
V = len(term_count.keys())
# 打开文件
with open(file_path, "r") as file:
    results = []
    i = 1
    bar = tqdm(file, total=total_lines, desc="Processing lines")
    # 开始测试
    for line in bar:

        # 语句结尾保存

        line = re.sub(r"([,])([^\d])", r" \1 \2", line)

        line = re.sub(r"([^sn])(['])", r"\1 \2", line)
        line = re.sub(r"([s])(['])", r"\1 \2 ", line)
        line = re.sub(r"([n])(['])([t])", r" \1\2\3 ", line)
        line = re.sub(r"([n])(['])([^t])", r"\1 \2\3", line)

        line = re.sub(r"([.?!]\s*$)", r" \1 ", line)
        # items = line.rstrip().split("\t")
        items = line.split("\t")
        line = items[2].split()
        corrected_line = line
        j = 0
        # 遍历句子单词
        for word in line:
            if word not in vocab:
                if word == "HKES":
                    print("find")
                # 需要替换word成正确的单词
                # Step1: 生成所有的(valid)候选集合
                # 获得编辑距离小于2的候选列表
                candidates = CG.generate_candidates(word, max_distance=max_distance)
                if word == "HKES":
                    print(candidates)
                    print(CG.generate_candidates("HKES", max_distance=max_distance))
                candidates = list(candidates)
                if word == "HKES":
                    print(candidates)
                probs = []

                # 对于每一个candidate, 计算它的score
                # score = p(correct)*p(mistake|correct)
                #       = log p(correct) + log p(mistake|correct)
                # 返回score最大的candidate
                for candi in candidates:
                    prob = 0
                    # 计算channel probability
                    if candi in channel_prob and word in channel_prob[candi]:
                        prob += np.log(channel_prob[candi][word])
                    else:
                        prob += np.log(0.0001)

                    # 计算语言模型的概率
                    # 以s=I like playing football.为例line=['I','like','playing','football']
                    # word为playing时
                    if j > 0:
                        forward_word = (
                            line[j - 1] + " " + candi
                        )  # 考虑前一个单词,出现like playing的概率
                        prob += calculate_smoothed_probability(
                            bigram_count, term_count, V, forward_word, line[j - 1]
                        )
                    if j + 1 < len(line):
                        backward_word = (
                            candi + " " + line[j + 1]
                        )  # 考虑后一个单词，出现playing football的概率
                        prob += calculate_smoothed_probability(
                            bigram_count, term_count, V, backward_word, candi
                        )
                    probs.append(prob)

                if probs:
                    max_idx = probs.index(max(probs))
                    if len(word) == 1:
                        corrected_line[j] = word  # 不替换单个字母
                    else:
                        corrected_line[j] = candidates[max_idx]
            j += 1

        corrected_sentence = " ".join(corrected_line)
        corrected_sentence = re.sub(
            r"\s*(['])\s*", r"\1", corrected_sentence
        )  # 去除标点前的空格
        corrected_sentence = re.sub(r"(s')", r"\1 ", corrected_sentence)  # 恢复s'的情况

        corrected_sentence = re.sub(
            r"\s([.])\s", r"\1", corrected_sentence
        )  # 去除标点前的空格

        corrected_sentence = re.sub(
            r"\s([,])", r"\1", corrected_sentence
        )  # 去除标点前的空格（保留逗号后面的空格）
        corrected_sentence = re.sub(
            r"(\d)([,])\s+(\d)", r"\1\2\3", corrected_sentence
        )  # 去除数据中的空格
        # 保存年份的逗号, 因为数字每三位要加逗号，年份一般是4位数
        corrected_sentence = re.sub(
            r"(\d{4,})([,])(\d)", r"\1\2 \3", corrected_sentence
        )
        corrected_sentence = re.sub(r"([,])(\d{4,})", r"\1 \2", corrected_sentence)

        # 句点补全
        corrected_sentence = re.sub(r"([.?!])\s*$", r"\1", corrected_sentence)

        if (
            corrected_sentence[-1] != "."
            and corrected_sentence[-1] != "?"
            and corrected_sentence[-1] != "!"
        ):
            corrected_sentence += "."

        results.append(f"{i}\t{corrected_sentence}")
        i += 1

# %%
with open("result.txt", "w") as file:
    file.write("\n".join(results))
