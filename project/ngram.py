from collections import Counter


def generate_ngram_models(corpus, n):
    """
    生成 n-grams 模型的函数
    input
    corpus: 语料库
    n: n-grams 的 n 值
    output
    term_count: 单词频率计数
    ngram_count: n-grams 的频率计数
    """
    term_count = Counter()
    ngram_count = Counter()

    for sentence in corpus:
        sentence = ['<s>'] + sentence  # '<s>'表示开头
        for i in range(len(sentence) - n + 1):
            term = sentence[i]
            ngram = sentence[i:i + n]

            term_count[term] += 1
            ngram_count[' '.join(ngram)] += 1

    return term_count, ngram_count


def calculate_smoothed_probability(bigram_count, term_count, V, bigram, term):
    """
    计算使用拉普拉斯平滑的 bigram 概率
    input
    bigram_count: bigram 的频率计数
    term_count: 单词频率计数
    V: 词汇表大小
    bigram: 当前 bigram
    term: 当前单词
    output
    平滑后的概率
    """
    bigram_frequency = bigram_count.get(bigram, 0) + 1  # 加1平滑
    term_frequency = term_count.get(term, 0) + V  # 加 V 平滑
    return bigram_frequency / term_frequency
