#!/usr/bin/env ipython3

import pygame
from views import TextView, Background, ChoiceMenu, ConditionalView

ALL_CHOICES = {
    'plant': ChoiceMenu("plant", ["Don't water the plant, it's too late",
                                  "Water the plant"])
}

def read_script(filename):
    all_texts = []
    row = ""
    with open(filename, 'r') as f:
        for line in f:
            if line[:3] == "!bg":
                imgname = line.strip().split(' ')[1]
                all_texts.append(Background(imgname))
            elif line[:7] == "!choice":
                choice = line.strip().split(' ')[1]
                all_texts.append(ALL_CHOICES[choice])
            elif line[:5] == "!cond":
                L = line.strip().split(' ')
                choice_id = L[1]
                choice_select = int(L[2])
                text = ' '.join(L[3:])
                v = ConditionalView(choice_id, choice_select, TextView(text))
                all_texts.append(v)
            elif line.strip() == "":
                t = row.strip()
                if t != '':
                    all_texts.append(TextView(t))
                row = ""
            else:
                row += line
    if len(row.strip()) > 0:
        all_texts.append(TextView(row.strip()))
    return all_texts

