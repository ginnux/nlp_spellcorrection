a = r" {2,}"

text = "       12"
import re

sub = re.find(a, text)

text.replace(sub, " ")

print(text)
