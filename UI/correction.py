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
        line = re.sub(r"([,])([^\d])", r" \1 \2", line)
        line = re.sub(r"([^s])(['])", r"\1 \2", line)
        line = re.sub(r"([s])(['])", r"\1 \2 ", line)
        line = re.sub(r"([.]$)", r" \1 ", line)

        line = line.split(' ')
        corrected_line = line
        j = 0
        # 遍历句子单词
        for word in line:
            if word not in self.vocab:
                # 需要替换word成正确的单词
                # Step1: 生成所有的(valid)候选集合
                # 获得编辑距离小于2的候选列表
                candidates = self.CG.generate_candidates(word, max_distance=self.max_distance)
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
                        forward_word = line[j - 1] + " " + candi  # 考虑前一个单词,出现like playing的概率
                        prob += calculate_smoothed_probability(self.bigram_count, self.term_count, self.V, forward_word,
                                                               line[j - 1])
                    if j + 1 < len(line):
                        backward_word = candi + " " + line[j + 1]  # 考虑后一个单词，出现playing football的概率
                        prob += calculate_smoothed_probability(self.bigram_count, self.term_count, self.V,
                                                               backward_word, candi)
                    probs.append(prob)

                if probs:
                    max_idx = probs.index(max(probs))
                    if len(word) == 1:
                        corrected_line[j] = word  # 不替换单个字母
                    else:
                        corrected_line[j] = candidates[max_idx]
            j += 1

        corrected_sentence = " ".join(corrected_line)
        corrected_sentence = re.sub(r"\s*(['])\s*", r"\1", corrected_sentence)  # 去除标点前的空格
        corrected_sentence = re.sub(r"(s')", r"\1 ", corrected_sentence)  # 恢复s'的情况

        corrected_sentence = re.sub(r"\s([.])\s", r"\1", corrected_sentence)  # 去除标点前的空格

        corrected_sentence = re.sub(r"\s([,])", r"\1", corrected_sentence)  # 去除标点前的空格（保留逗号后面的空格）
        corrected_sentence = re.sub(r"(\d)([,])\s+(\d)", r"\1\2\3", corrected_sentence)  # 去除数据中的空格
        # 句点补全
        if corrected_sentence[-1] != ".":
            corrected_sentence += "."
        return corrected_sentence
