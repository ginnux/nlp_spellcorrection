a = r" {2,}"

text = "       12"
import re

if re.match(a, text):
    print("Matched")
    re.replace(a, r" +", " ")

print(text)
