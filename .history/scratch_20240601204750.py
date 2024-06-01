a = r" {2,}"

text = "       12"
import re

subs = re.findall(a, text)
for sub in subs:
    text.replace(sub, " ")

print(text)
