import nltk
import re

# nltk.download('reuters')
# nltk.download('punkt')

# 词典库
vocab = set(
    [line.rstrip() for line in open("./resource/vocab.txt")]
)  # 用set效率高一些(时间复杂度)


# 生成单词的所有候选集合
def generate_candidates(word):
    """
    word: 给定的输入（错误的输入）
    返回所有(valid)候选集合
    """
    # 生成编辑距离不大于1的单词
    # 1.insert 2. delete 3. replace
    # appl: replace: bppl, cppl, aapl, abpl...
    #       insert: bappl, cappl, abppl, acppl....
    #       delete: ppl, apl, app

    # 假设使用26个字符
    letters = "abcdefghijklmnopqrstuvwxyz"

    splits = [
        (word[:i], word[i:]) for i in range(len(word) + 1)
    ]  # 将单词在不同的位置拆分成2个字符串，然后分别进行insert，delete你replace操作,拆分形式为：[('', 'apple'), ('a', 'pple'), ('ap', 'ple'), ('app', 'le'), ('appl', 'e'), ('apple', '')]
    # print(splits)

    # insert操作
    inserts = [L + c + R for L, R in splits for c in letters]
    # delete
    deletes = [
        L + R[1:] for L, R in splits if R
    ]  # 判断分割后的字符串R是否为空，不为空，删除R的第一个字符即R[1:]
    # transposes
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    # replace
    replaces = [
        L + c + R[1:] for L, R in splits if R for c in letters
    ]  # 替换R的第一个字符,即c+R[1:]
    candidates = set(inserts + deletes + replaces + transposes)
    # print(candidates)
    # 过来掉不存在于词典库里面的单词
    return [word for word in candidates if word in vocab]


# generate_candidates("apple")#生成Apple的所有候选集合
def generate_edit_two(word):
    """
    给定一个字符串，生成编辑距离不大于2的字符串
    """
    return [e2 for e1 in generate_candidates(word) for e2 in generate_candidates(e1)]


# 读取语料库
# import nltk
# nltk.download('reuters')
# nltk.download('punkt')

from nltk.corpus import reuters  # reuters路透社语料库

# 读取语料库
categories = reuters.categories()  # 路透社语料库的类别
# print(categories)
corpus = reuters.sents(categories=categories)  # sents()指定分类中的句子
# print(corpus)

# 构建语言模型：bigram
term_count = {}
bigram_count = {}
for doc in corpus:
    doc = ["<s>"] + doc  # '<s>'表示开头
    for i in range(0, len(doc) - 1):
        # bigram: [i,i+1]
        term = doc[i]  # term是doc中第i个单词
        bigram = doc[i : i + 2]  # bigram为第i,i+1个单词组成的

        if term in term_count:
            term_count[term] += 1  # 如果term存在term_count中，则加1
        else:
            term_count[term] = 1  # 如果不存在，则添加，置为1

        bigram = " ".join(bigram)
        if bigram in bigram_count:
            bigram_count[bigram] += 1
        else:
            bigram_count[bigram] = 1

# 用户打错的概率统计 - channel probability
# 创建一个字典来存储channel probabilities
channel_prob = {}
total_errors = 0

i = 0
# 解析错误数据并计算总错误次数
for line in open("count_1edit_test2.txt"):
    i += 1
    print(f"{i}\n")
    correct, parts = line.split("|")
    new_parts = parts.split()
    mistake = new_parts[0]
    count = int(new_parts[1])  # 416 678 1226 1309 1404 1586
    """parts = line.split()
    error_pair = parts[0].strip()
    count = int(parts[1].strip())
    correct, mistake = error_pair.split("|")"""

    if correct not in channel_prob:
        channel_prob[correct] = {}

    channel_prob[correct][mistake] = count
    total_errors += count

# 计算每种错误的概率
for correct in channel_prob:
    for mistake in channel_prob[correct]:
        channel_prob[correct][mistake] /= total_errors

import numpy as np

V = len(term_count.keys())
file = open("testdata.txt", "r")
results = []
i = 1
for line in file:
    line = re.sub(r"([,.])", r" \1 ", line)
    items = line.rstrip().split("\t")
    line = items[2].split()
    corrected_line = line
    print(line)
    j = 0
    for word in line:
        if word not in vocab:
            # 需要替换word成正确的单词
            # Step1: 生成所有的(valid)候选集合
            candidates = generate_candidates(word)  # + generate_edit_two(word)

            # 一种方式： if candidate = [], 多生成几个candidates, 比如生成编辑距离不大于2的
            # TODO ： 根据条件生成更多的候选集合
            if len(candidates) < 1:
                continue
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
                """
               比如s=I like playing football.
               line=['I','like','playing','football']
               word为playing时
               """
                forward_word = (
                    line[j - 1] + " " + candi
                )  # 考虑前一个单词,出现like playing的概率

                if forward_word in bigram_count and line[j - 1] in term_count:
                    prob += np.log(
                        (bigram_count[forward_word] + 1.0)
                        / (term_count[line[j - 1]] + V)
                    )

                else:
                    prob += np.log(1.0 / V)

                if j + 1 < len(line):  # 考虑后一个单词，出现playing football的概率
                    backward_word = candi + " " + line[j + 1]
                    if backward_word in bigram_count and candi in term_count:
                        prob += np.log(
                            (bigram_count[backward_word] + 1.0)
                            / (term_count[candi] + V)
                        )
                    else:
                        prob += np.log(1.0 / V)
                probs.append(prob)

            max_idx = probs.index(max(probs))
            print(word, candidates[max_idx])
            corrected_line[j] = candidates[max_idx]
        j += 1
    corrected_sentence = " ".join(corrected_line)
    results.append(f"{i}\t{corrected_sentence}")
    i += 1

with open("result.txt", "w") as file:
    file.write("\n".join(results))
