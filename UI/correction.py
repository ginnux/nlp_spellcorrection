import nltk
import re
import numpy as np
from project.ngram import generate_ngram_models
from project.ngram import calculate_smoothed_probability
from nltk.corpus import reuters
from tqdm import tqdm
from project.edit_distance import CandidatesGenerator


def generate_noisy_channel_model():
    channel_prob = {}
    total_errors = 0

    i = 0
    # 解析错误数据并计算总错误次数
    for line in open('../project/count_1edit.txt'):
        i += 1
        # Step1:解析数据
        # 正则表达式找到错误次数
        count = re.findall(r'\d+', line)[-1]

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

    return channel_prob


class spell_correction():
    def __init__(self):
        self.vocab = {line.rstrip() for line in open('../project/vocab.txt')}

        nltk.download('reuters')
        nltk.download('punkt')
        categories = reuters.categories()  # 路透社语料库的类别
        corpus = reuters.sents(categories=categories)  # sents()指定分类中的句子

        # 构建语言模型：bigram
        self.term_count, self.bigram_count = generate_ngram_models(corpus, 2)

        # 构造channel模型
        self.channel_prob = generate_noisy_channel_model()

        # 构造候选词生成类
        # 创建生成者
        self.CG = CandidatesGenerator(vocab=self.vocab)
        # 设置最大编辑距离
        self.max_distance = 1
        self.V = len(self.term_count.keys())

    def correction(self, line):
        # 分开逗号两边的词
        line = re.sub(r"([,])([^\d])", r" \1 \2", line)

        # 单引号分开原则：特别注意s'和n't情况，以及作为成对引号
        # 先考虑成对引号，第一个引号前一般是空格，展开这对单引号，两侧加入空格
        line = re.sub(r"(\s)(['])([^']+)(['])", r"\1\2 \3 \4 ", line)
        # 一般情况，一般来说单引号前是一个完整的单词
        line = re.sub(r"([^sn\s])(['])", r"\1 \2", line)
        # 若单引号前是复数s，单引号后是缩写，如s'，则不分开
        line = re.sub(r"([s])(['])", r"\1 \2 ", line)
        # 若单引号是n't的情况，如don't，won't等，则不应该分开n't，而应该分开n't两边的词，比如do
        line = re.sub(r"([n])(['])([t])", r" \1\2\3 ", line)
        # 若单引号是前是n，后面不是t，则就是一般情况，分开n和单引号
        line = re.sub(r"([n])(['])([^t])", r"\1 \2\3", line)

        # 识别句尾标点符号并分开
        line = re.sub(r"([.?!]\s*$)", r" \1 ", line)

        line = line.split()
        corrected_line = line
        j = 0
        # 遍历句子单词
        for word in line:
            if word not in self.vocab:
                # 需要替换word成正确的单词
                # Step1: 生成所有的(valid)候选集合
                # 获得编辑距离小于2的候选列表
                candidates = self.CG.generate_candidates(word, max_distance=max_distance)
                candidates = list(candidates)
                probs = []

                # 对于每一个candidate, 计算它的score
                # score = p(correct)*p(mistake|correct)
                #       = log p(correct) + log p(mistake|correct)
                # 返回score最大的candidate
                for candi in candidates:
                    prob = 0
                    # 计算channel probability
                    if candi in self.channel_prob and word in self.channel_prob[candi]:
                        prob += np.log(self.channel_prob[candi][word])
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
                            self.bigram_count, self.term_count, self.V, forward_word, line[j - 1]
                        )
                    if j + 1 < len(line):
                        backward_word = (
                                candi + " " + line[j + 1]
                        )  # 考虑后一个单词，出现playing football的概率
                        prob += calculate_smoothed_probability(
                            self.bigram_count, self.term_count, self.V, backward_word, candi
                        )
                    probs.append(prob)

                if probs:
                    max_idx = probs.index(max(probs))
                    if len(word) == 1:
                        corrected_line[j] = word  # 不替换单个字母
                    else:
                        corrected_line[j] = candidates[max_idx]
            j += 1

        # 因为对所有词间加入了标点符号，要对句子标点符号修正
        corrected_sentence = " ".join(corrected_line)

        # 先将单引号周围一侧的空格去掉，再按情况恢复，但两侧都有空格的是成对的引号，不去除
        corrected_sentence = re.sub(r"\s+(['])([^\s])", r"\1\2", corrected_sentence)
        corrected_sentence = re.sub(r"([^\s])(['])\s+", r"\1\2", corrected_sentence)
        # 单引号周围是空格的情况一般只有s'，恢复
        corrected_sentence = re.sub(r"(s')", r"\1 ", corrected_sentence)
        # 去除成对单引号剩余的空格，前单引号后面的空格去掉，后单引号前面的空格去掉
        corrected_sentence = re.sub(
            r"(\s)(')(\s)([^']*)(\s)(')(\s)", r"\1\2\4\6\7", corrected_sentence
        )

        # 句尾标点旁边的空格去除
        corrected_sentence = re.sub(r"\s([.])\s", r"\1", corrected_sentence)

        # 处理逗号附近的空格
        # 逗号前面不可能有空格，去掉
        corrected_sentence = re.sub(r"\s([,])", r"\1", corrected_sentence)
        # 如果两边都有数字，则只有是数字中的逗号和年份月份分开的情况，先去除所有空格
        corrected_sentence = re.sub(r"(\d)([,])\s+(\d)", r"\1\2\3", corrected_sentence)
        # 恢复年份的逗号空格, 因为数字每三位要加逗号，年份一般是4位数
        corrected_sentence = re.sub(
            r"(\d{4,})([,])(\d)", r"\1\2 \3", corrected_sentence
        )
        corrected_sentence = re.sub(r"([,])(\d{4,})", r"\1 \2", corrected_sentence)

        # 句点补全，句子原来有什么标点，就补全什么标点，否则就补句号
        # 先去掉句尾空格
        corrected_sentence = re.sub(r"([.?!])\s*$", r"\1", corrected_sentence)
        # 补全
        if (
                corrected_sentence[-1] != "."
                and corrected_sentence[-1] != "?"
                and corrected_sentence[-1] != "!"
        ):
            corrected_sentence += "."
        return corrected_sentence
