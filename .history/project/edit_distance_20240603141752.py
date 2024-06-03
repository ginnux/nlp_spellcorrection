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
                dp[i][j] = 1 + min(
                    dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1]  # 插入  # 删除
                )  # 替换
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


class CandidatesGenerator:
    def __init__(self, vocab):
        self.vocab = vocab

    # 生成单词的所有候选集合
    def candidates(self, word):
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
        return [word for word in candidates if word in self.vocab]

    def generate_candidates(self, word, max_distance=1):
        """
        生成编辑距离不大于 max_distance 的单词候选集合

        参数:
        word: 给定的输入单词（可能错误的输入）
        max_distance: 最大允许的编辑距离

        返回:
        所有在词汇表中且编辑距离不大于 max_distance 的候选单词集合
        """
        candidates = set()

        edit_list = [[]] * (max_distance + 1)
        # 为了统一for循环，编辑距离为0表示只有输入单词本身
        edit_list[0] = [word]
        # 所有编辑距离相同的单词放在一个列表里
        for i in range(1, max_distance + 1):
            # i 为编辑距离，从1开始遍历，将前一个编辑距离的词生成当前编辑距离的词集合
            (edit_list[i]).extend([self.candidates(w) for w in edit_list[i - 1]])

        # 合并所有编辑距离的词
        all_candidates = []
        for i in range(1, max_distance + 1):
            all_candidates += edit_list[i]
        # 去冗余
        all_candidates = list(set(all_candidates))
        return all_candidates


if __name__ == "__main__":
    vocab = {line.rstrip() for line in open("project/vocab.txt")}

    CG = CandidatesGenerator(vocab=vocab)
    print(CG.generate_candidates("appl", max_distance=2))
