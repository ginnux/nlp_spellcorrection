from nltk import download as nltk_download
import re
from numpy import log
from ngram import generate_ngram_models
from ngram import calculate_smoothed_probability
from nltk.corpus import reuters
from tqdm import tqdm
from edit_distance import CandidatesGenerator
from pathlib import Path

# 获取目录路径对象
import os, sys

work_path = os.path.dirname(sys.argv[0])
vocab_path = os.path.join(work_path, "vocab.txt")
count_path = os.path.join(work_path, "count_1edit.txt")


def generate_noisy_channel_model():
    channel_prob = {}
    total_errors = 0

    i = 0
    # 解析错误数据并计算总错误次数
    for line in open(count_path, "r"):
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
        # 原先是correct，mistake = line。split("|")，对比ppt后感觉是写反了
        mistake, correct = line.split("|")

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


def get_key_value_pairs(correct_word, incorrect_word):
    """
    获取正确单词和错误单词之间修改字符的键值对，如remains与remain，输出为dict:{‘s':’ ‘}。
    """
    key_value_pairs = {}

    # 与寻找候选词类似挨个遍历查找两个单词间具体不同

    # 假设使用26个字符
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    # 将单词在不同的位置拆分成2个字符串，然后分别进行insert，delete你replace操作,
    # 拆分形式为：[('', 'apple'), ('a', 'pple'), ('ap', 'ple'), ('app', 'le'), ('appl', 'e'), ('apple', '')]
    splits = [
        (incorrect_word[:i], incorrect_word[i:]) for i in range(len(incorrect_word) + 1)
    ]

    # insert操作
    for L, R in splits:
        for c in letters:
            insert = L + c + R
            if correct_word == insert:
                key_value_pairs[c] = " "
    # delete
    # 判断分割后的字符串R是否为空，不为空，删除R的第一个字符即R[1:]
    for L, R in splits:
        if R:
            delete = L + R[1:]
            if correct_word == delete:
                key_value_pairs[" "] = R[0]

    # transposes
    for L, R in splits:
        if len(R) > 1:
            transpose = L + R[1] + R[0] + R[2:]
            if correct_word == transpose and transpose != incorrect_word:
                key_value_pairs[R[1] + R[0]] = R[0] + R[1]
    # replace
    for L, R in splits:
        if R:
            for c in letters:
                replace = L + c + R[1:]  # 替换R的第一个字符,即c+R[1:]
                if correct_word == replace and replace != incorrect_word:
                    key_value_pairs[c] = R[0]
    # exchange
    for i in range(len(incorrect_word)):
        for j in range(i + 1, len(incorrect_word)):
            exchange = (
                incorrect_word[:i]
                + incorrect_word[j]
                + incorrect_word[i + 1 : j]
                + incorrect_word[i]
                + incorrect_word[j + 1 :]
            )
            if correct_word == exchange and exchange != incorrect_word:
                key_value_pairs[
                    incorrect_word[j] + incorrect_word[i + 1 : j] + incorrect_word[i]
                ] = incorrect_word[i:j]

    return key_value_pairs


def generate_candidates_probs(
    candidates, word, channel_prob, bigram_count, term_count, line, j
):
    # 对于每一个candidate, 计算它的score
    # score = p(correct)*p(mistake|correct)
    #       = log p(correct) + log p(mistake|correct)
    # 返回score最大的candidate
    # 已修改，原先没有传入line和j
    probs = []
    for candi in candidates:
        prob = 0
        # 计算channel probability
        # get_key_value_pairs获取candi与word的字符区别，以便在channel_prob中寻找
        pair = get_key_value_pairs(candi, word)
        for correct_char, incorrect_char in pair.items():
            if (
                correct_char in channel_prob
                and incorrect_char in channel_prob[correct_char]
            ):
                prob += log(channel_prob[correct_char][incorrect_char])
                prob -= log(0.0001)
        prob += log(0.0001)

        # 计算语言模型的概率
        # 以s=I like playing football.为例line=['I','like','playing','football']
        # word为playing时
        if j > 0:
            forward_word = (
                line[j - 1] + " " + candi
            )  # 考虑前一个单词,出现like playing的概率
            prob += calculate_smoothed_probability(
                bigram_count,
                term_count,
                len(term_count.keys()),
                forward_word,
                line[j - 1],
            )
        if j + 1 < len(line):
            backward_word = (
                candi + " " + line[j + 1]
            )  # 考虑后一个单词，出现playing football的概率
            prob += calculate_smoothed_probability(
                bigram_count, term_count, len(term_count.keys()), backward_word, candi
            )
        probs.append(prob)
    return probs


class spell_correction:
    def __init__(self):
        try:
            self.vocab = {line.rstrip() for line in open(vocab_path, "r")}
        except FileNotFoundError:
            print("No vocab file found.")
            exit(1)
        try:
            categories = reuters.categories()  # 路透社语料库的类别
            corpus = reuters.sents(categories=categories)  # sents()指定分类中的句子
        except LookupError:
            print(
                "first time running, download reuters corpus, it will take some time."
            )
            nltk_download("reuters")
            nltk_download("punkt")
            categories = reuters.categories()  # 路透社语料库的类别
            corpus = reuters.sents(categories=categories)  # sents()指定分类中的句子

        # 构建语言模型：bigram
        self.term_count, self.bigram_count = generate_ngram_models(corpus, 2)

        # 构造channel模型
        try:
            self.channel_prob = generate_noisy_channel_model()
        except FileNotFoundError:
            print("No channel noisy model generate dataset.")
            exit(1)
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
        # 开头增加符号<s>
        line = ["<s>"] + line
        corrected_line = line
        # 非词错误标志
        not_word = 0

        # 遍历句子单词
        for j, word in enumerate(line):
            if word == "<s>" or word == "</s>":
                continue
            if word not in self.vocab:
                # 需要替换word成正确的单词
                # Step1: 生成所有的(valid)候选集合
                # 获得编辑距离小于2的候选列表

                # 存在非词错误
                not_word = 1
                candidates = self.CG.generate_candidates(
                    word, max_distance=self.max_distance
                )
                candidates = list(candidates)
                probs = generate_candidates_probs(
                    candidates,
                    word,
                    self.channel_prob,
                    self.bigram_count,
                    self.term_count,
                    line,
                    j,
                )

                if probs:
                    max_idx = probs.index(max(probs))
                    if len(word) == 1:
                        corrected_line[j] = word  # 不替换单个字母
                    else:
                        corrected_line[j] = candidates[max_idx]
        corrected_line = corrected_line[1:]

        # 存在实词错误,相当于判断无非词错误后来看实词错误
        if not_word != 1:
            # 遍历句子单词的候选词，选出候选词在当前句中最合理的
            word_candidates = []
            best_sentence = []
            best_prob = []
            for k, word in enumerate(line):
                # 跳过开头
                if word == "<s>" or word == "</s>":
                    continue
                # 需要替换word成正确的单词
                # Step1: 生成所有的(valid)候选集合
                # 获得编辑距离小于2的候选列表
                best_prob_for_word = []
                best_sentence_for_word = []
                candidates = self.CG.generate_candidates(
                    word, max_distance=self.max_distance
                )
                candidates = list(candidates)

                # 对长度小于等于3的词候选词默认自身
                if len(word) <= 3:
                    word_candidates = [word]  # 不替换单个字母
                else:
                    word_candidates = candidates
                # 开始遍历候选词，存储对应修改过的句子，比较prob
                for word_candidate in word_candidates:
                    sentence_prob = 0
                    sentence = line[:k] + [word_candidate] + line[k + 1 :]  # 替换word
                    for m in range(
                        len(sentence) - 1
                    ):  # 采用链式法则计算概率，同时还有一项将当前词错写成候选词的概率
                        sentence_word = sentence[m]
                        sentence_word_next = sentence[m + 1]
                        sentence_prob += calculate_smoothed_probability(
                            self.bigram_count,
                            self.term_count,
                            self.V,
                            (sentence_word, sentence_word_next),
                            sentence_word,
                        )
                    # 获取候选词与单词的不同，返回一个字典，然后扔进channel_prob里查询
                    pairs = get_key_value_pairs(word_candidate, word)
                    for correct_char, incorrect_char in pairs.items():
                        if (
                            correct_char in self.channel_prob
                            and incorrect_char in self.channel_prob[correct_char]
                        ):
                            sentence_prob += log(
                                self.channel_prob[correct_char][incorrect_char]
                            )
                            sentence_prob -= log(0.0001)
                    sentence_prob += log(0.0001)
                    # 保存结果
                    best_sentence_for_word.append(sentence)
                    best_prob_for_word.append(sentence_prob)
                # 保存单个单词所有候选词概率最小的结果，然后看下一个单词
                best_sentence.append(
                    best_sentence_for_word[
                        best_prob_for_word.index(min(best_prob_for_word))
                    ]
                )
                best_prob.append(min(best_prob_for_word))
            # 去除开头符号<s>
            corrected_line = best_sentence[best_prob.index(min(best_prob))][1:]

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
