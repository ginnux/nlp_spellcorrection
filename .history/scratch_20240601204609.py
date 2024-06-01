a = r" {2,}"

text = "       12"
import re

if re.match(a, text):
    print("Matched")
    text.replace(r" +", " ")

print(text)
