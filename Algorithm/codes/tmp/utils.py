import torch
from torch.autograd import Variable


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def getSent(anor_test, nor_test, nor_train, request_file_train, 
                                request_file_test,
                                label_file_train,
                                label_file_test):
    sentences = []
    labels = []
    with open(anor_test,'r') as f:
        first_line = True
        content_flag = False
        for line in f:
            if first_line == True and line.strip() != "":
                sentences.append(line.strip())
                labels.append("1")
                first_line = False
            elif first_line == False and line.strip().startswith("Content-Length:"):
                content_flag = True
            elif first_line == False and line.strip() != "" and content_flag == True:
                sent_split = sentences[-1].split()
                sent_split[-2] = sent_split[-2] + "?" + line.strip()
                sentences[-1] = " ".join(sent_split)
                content_flag = False
            elif line.strip() == "" and content_flag == False:
                first_line = True
    with open(nor_test,'r') as f:
        first_line = True
        content_flag = False
        for line in f:
            if first_line == True and line.strip() != "":
                sentences.append(line.strip())
                labels.append("0")
                first_line = False
            elif first_line == False and line.strip().startswith("Content-Length:"):
                content_flag = True
            elif first_line == False and line.strip() != "" and content_flag == True:
                sent_split = sentences[-1].split()
                sent_split[-2] = sent_split[-2] + "?" + line.strip()
                sentences[-1] = " ".join(sent_split)
                content_flag = False
            elif line.strip() == "" and content_flag == False:
                first_line = True
    
    with open(request_file_test,'w') as f:
        f.write('\n'.join(sentences))
    with open(label_file_test,'w') as f:
        f.write('\n'.join(labels))
    
    sentences = []
    labels = []
    with open(nor_train,'r') as f:
        first_line = True
        content_flag = False
        for line in f:
            if first_line == True and line.strip() != "":
                sentences.append(line.strip())
                labels.append("0")
                first_line = False
            elif first_line == False and line.strip().startswith("Content-Length:"):
                content_flag = True
            elif first_line == False and line.strip() != "" and content_flag == True:
                sent_split = sentences[-1].split()
                sent_split[-2] = sent_split[-2] + "?" + line.strip()
                sentences[-1] = " ".join(sent_split)
                content_flag = False
            elif line.strip() == "" and content_flag == False:
                first_line = True
    with open(request_file_train,'w') as f:
        f.write('\n'.join(sentences))
    with open(label_file_train,'w') as f:
        f.write('\n'.join(labels))
    return sentences, labels

if __name__ == "__main__":
    anor_test = 'data/CSIC2010/anomalousTrafficTest.txt'
    nor_test = 'data/CSIC2010/normalTrafficTest.txt'
    nor_train = 'data/CSIC2010/normalTrafficTraining.txt'
    request_file_train = 'data/CSIC2010/request_train.txt'
    request_file_test = 'data/CSIC2010/request_test.txt'
    label_file_train = 'data/CSIC2010/label_train.txt'
    label_file_test = 'data/CSIC2010/label_test.txt'
    sentences, labels = getSent(anor_test, nor_test, nor_train, 
                                request_file_train, 
                                request_file_test,
                                label_file_train,
                                label_file_test)