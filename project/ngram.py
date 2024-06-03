from collections import Counter
import itertools
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

    # 获取所有文件的 ID
    file_ids = corpus.fileids()

    # 从语料库中获取所有句子
    sentences = [corpus.sents(file_id) for file_id in file_ids]
    sentences = list(itertools.chain(*sentences))

    for sentence in sentences:
        sentence = ['<s>'] + sentence  # '<s>'表示开头
        for i in range(len(sentence) - n + 1):
            term = sentence[i]
            ngram = sentence[i:i + n]

            term_count[term] += 1
            ngram_count[' '.join(ngram)] += 1

    return term_count, ngram_count