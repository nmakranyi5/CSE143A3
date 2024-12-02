#!/usr/bin/env python3

import glob
import os
import re

header_pattern = re.compile("^[A-Za-z-.]+: ")
number_fn_pattern = re.compile("^[0-9]+$")


def strip_headers(fn):
    lines = []
    with open(fn, errors="ignore") as infile:
        for line in infile:
            if not re.match(header_pattern, line) and line.strip():
                lines.append(line)
    with open(fn, "w") as outfile:
        for line in lines:
            print(line, file=outfile, end="")


for fn in glob.glob("20_newsgroups/*/*"):
    base = os.path.basename(fn)
    if re.match(number_fn_pattern, base):
        os.rename(fn, f"{fn}.txt")

for fn in glob.glob("20_newsgroups/*/*.txt"):
    strip_headers(fn)

for fn in glob.glob("20_newsgroups/*/*.txt"):
    base = os.path.basename(fn)
    if base.endswith("0.txt"):
        newname = f"test/{fn}"
        os.makedirs(os.path.dirname(newname), exist_ok=True)
        os.rename(fn, newname)
