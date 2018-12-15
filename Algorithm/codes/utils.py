import os
import torch
from torch.autograd import Variable
import numpy as np


def to_var(x, cuda_num, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda(cuda_num)
    return Variable(x, volatile=volatile)

def pad(sents, lens, padding_value=0):
    max_len = max(lens)
    new_sents = []
    for s in sents:
        slen = s.size()[0]
        if slen == max_len:
            new_sents.append(s)
        else:
            new_sents.append(torch.cat((s, torch.zeros((max_len-s.size()[0])).long())))
    new_sents = torch.stack(new_sents)
    return new_sents

# def getSent(anor_test, nor_test, nor_train, request_file, label_file):
#     sentences = []
#     labels = []
#     if os.path.isfile(request_file) and os.path.isfile(label_file):
#         with open(request_file,'r') as f:
#             for line in f:
#                 sentences.append(line.strip())
#         with open(label_file,'r') as f:
#             for line in f:
#                 labels.append(line.strip())
#     else:
#         print("Creating request_file and label_file...")
#         with open(anor_test,'r') as f:
#             first_line = True
#             content_flag = False
#             for line in f:
#                 if first_line == True and line.strip() != "":
#                     sentences.append(line.strip())
#                     labels.append("1")
#                     first_line = False
#                 elif first_line == False and line.strip().startswith("Content-Length:"):
#                     content_flag = True
#                 elif first_line == False and line.strip() != "" and content_flag == True:
#                     sent_split = sentences[-1].split()
#                     sent_split[-2] = sent_split[-2] + "?" + line.strip()
#                     sentences[-1] = " ".join(sent_split)
#                     content_flag = False
#                 elif line.strip() == "" and content_flag == False:
#                     first_line = True
#         with open(nor_test,'r') as f:
#             first_line = True
#             content_flag = False
#             for line in f:
#                 if first_line == True and line.strip() != "":
#                     sentences.append(line.strip())
#                     labels.append("0")
#                     first_line = False
#                 elif first_line == False and line.strip().startswith("Content-Length:"):
#                     content_flag = True
#                 elif first_line == False and line.strip() != "" and content_flag == True:
#                     sent_split = sentences[-1].split()
#                     sent_split[-2] = sent_split[-2] + "?" + line.strip()
#                     sentences[-1] = " ".join(sent_split)
#                     content_flag = False
#                 elif line.strip() == "" and content_flag == False:
#                     first_line = True
#         with open(nor_train,'r') as f:
#             first_line = True
#             content_flag = False
#             for line in f:
#                 if first_line == True and line.strip() != "":
#                     sentences.append(line.strip())
#                     labels.append("0")
#                     first_line = False
#                 elif first_line == False and line.strip().startswith("Content-Length:"):
#                     content_flag = True
#                 elif first_line == False and line.strip() != "" and content_flag == True:
#                     sent_split = sentences[-1].split()
#                     sent_split[-2] = sent_split[-2] + "?" + line.strip()
#                     sentences[-1] = " ".join(sent_split)
#                     content_flag = False
#                 elif line.strip() == "" and content_flag == False:
#                     first_line = True
#         with open(request_file,'w') as f:
#             f.write('\n'.join(sentences))
#         with open(label_file,'w') as f:
#             f.write('\n'.join(labels))
#     return sentences, labels

def getS(data_fn):
    data = []
    with open(data_fn) as f:
        for line in f:
            data.append(line.strip())
    return data

def getSent(train_fn, test_fn, train_label_fn, test_label_fn):
    train_sents = getS(train_fn)
    test_sents = getS(test_fn)
    train_labels = getS(train_label_fn)
    train_labels = [int(i) for i in train_labels]
    test_labels = getS(test_label_fn)
    test_labels = [int(i) for i in test_labels]
    return train_sents, test_sents, train_labels, test_labels

if __name__ == "__main__":
    anor_test = '../data/HTTP_2010/anomalousTrafficTest.txt'
    nor_test = '../data/HTTP_2010/normalTrafficTest.txt'
    nor_train = '../data/HTTP_2010/normalTrafficTraining.txt'
    request_file = '../data/HTTP_2010/request.txt'
    label_file = '../data/HTTP_2010/label.txt'
    train_sents, test_sents, train_labels, test_labels = getSent(train_fn, test_fn, train_label_fn, test_label_fn)
