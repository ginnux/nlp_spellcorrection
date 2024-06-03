import nltk

anspath = "ans.txt"
resultpath = "result.txt"
ansfile = open(anspath, "r")
resultfile = open(resultpath, "r")
count = 0
for i in range(1000):
    ansline = ansfile.readline().split("\t")[1]
    anslist = nltk.word_tokenize(ansline)
    ansset = set(anslist)
    resultline = resultfile.readline().split("\t")[1]
    resultlist = nltk.word_tokenize(resultline)
    resultset = set(resultlist)

    if ansset == resultset:
        count += 1
    else:
        result = " ".join(resultset)
        print(f"{i}\t ans: {ansline}")
        print(f"{i}\t result: {resultline}")
        for w in range(min(len(anslist), len(resultlist))):
            if anslist[w] != resultlist[w]:
                print("ans: " + anslist[w] + " result: " + resultlist[w])
print("Accuracy is : %.2f%%" % (count * 1.00 / 10))

exit(count)
