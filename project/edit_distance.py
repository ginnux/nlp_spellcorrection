def edit_distance_custom(word1, word2):
    """
    计算两个单词之间的编辑距离
    参数:
    word1: 第一个单词
    word2: 第二个单词

    返回:
    编辑距离
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j  # word1为空，插入所有的word2字符
            elif j == 0:
                dp[i][j] = i  # word2为空，删除所有的word1字符
            elif word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # 字符相等，不需要操作
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],  # 插入
                                   dp[i - 1][j],  # 删除
                                   dp[i - 1][j - 1])  # 替换
    return dp[m][n]

def generate_candidates(word, vocab, max_distance):
    """
    生成编辑距离不大于 max_distance 的单词候选集合

    参数:
    word: 给定的输入单词（可能错误的输入）
    max_distance: 最大允许的编辑距离

    返回:
    所有在词汇表中且编辑距离不大于 max_distance 的候选单词集合
    """
    candidates = set()

    for candidate in vocab:
        if edit_distance_custom(word, candidate) <= max_distance:
            candidates.add(candidate)

    return candidates
