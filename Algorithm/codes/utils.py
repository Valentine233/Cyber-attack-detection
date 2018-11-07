import os
import torch
from torch.autograd import Variable
import numpy as np


def to_var(x, cuda_num, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda(cuda_num)
    return Variable(x, volatile=volatile)

def getSent(anor_test, nor_test, nor_train, request_file, label_file):
    sentences = []
    labels = []
    if os.path.isfile(request_file) and os.path.isfile(label_file):
        with open(request_file,'r') as f:
            for line in f:
                sentences.append(line.strip())
        with open(label_file,'r') as f:
            for line in f:
                labels.append(line.strip())
    else:
        with open(anor_test,'r') as f:
            first_line = True
            for line in f:
                if first_line == True and line.strip() != "":
                    sentences.append(line.strip())
                    labels.append("1")
                    first_line = False
                if line.strip() == "":
                    first_line = True
        with open(nor_test,'r') as f:
            first_line = True
            for line in f:
                if first_line == True and line.strip() != "":
                    sentences.append(line.strip())
                    labels.append("0")
                    first_line = False
                if line.strip() == "":
                    first_line = True
        with open(nor_train,'r') as f:
            first_line = True
            for line in f:
                if first_line == True and line.strip() != "":
                    sentences.append(line.strip())
                    labels.append("0")
                    first_line = False
                if line.strip() == "":
                    first_line = True
        with open(request_file,'w') as f:
            f.write('\n'.join(sentences))
        with open(label_file,'w') as f:
            f.write('\n'.join(labels))
    return sentences, labels