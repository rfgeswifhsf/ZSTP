# !/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import json,os

def build_vocab(corpus_file_list, vocab_file, tag_file):
    words = set()
    tags = set()
    for file in corpus_file_list:
        for line in open(file, "r", encoding='utf-8').readlines():
            line = line.strip()
            if line == "end":
                continue
            try:
                w,t = line.split()
                words.add(w)
                tags.add(t)
            except Exception as e:
                print(line.split())
                # raise e

    if not os.path.exists(vocab_file):
        with open(vocab_file,"w") as f:
            for index,word in enumerate(["<UKN>"]+list(words) ):
                f.write(word+"\n")

    tag_sort = {
        "O": 0,
        "B": 1,
        "I": 2,
        "E": 3,
    }

    tags = sorted(list(tags),
           key=lambda x: (len(x.split("-")), x.split("-")[-1], tag_sort.get(x.split("-")[0], 100))
           )
    if not os.path.exists(tag_file):
        with open(tag_file,"w") as f:
            for index,tag in enumerate(["<UKN>"]+tags):
                f.write(tag+"\n")

def read_vocab(vocab_file):
    vocab2id = {}
    id2vocab = {}
    for index,line in enumerate([line.strip() for line in open(vocab_file,"r").readlines()]):
        vocab2id[line] = index
        id2vocab[index] = line
    return vocab2id, id2vocab

def tokenize(filename,vocab2id,tag2id):
    contents = []
    labels = []
    content = []
    label = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in [elem.strip() for elem in fr.readlines()][:500000]:
            try:
                if line != "end":
                    w,t = line.split()
                    content.append(vocab2id.get(w,0))
                    label.append(tag2id.get(t,0))
                else:
                    if content and label:
                        contents.append(content)
                        labels.append(label)
                    content = []
                    label = []
            except Exception as e:
                content = []
                label = []
    # 将标量序列转换为 np.ndarray
    # padding 有pre,post
    contents = tf.keras.preprocessing.sequence.pad_sequences(contents, padding='post')  #type(contents)---><class 'list'>
    labels = tf.keras.preprocessing.sequence.pad_sequences(labels, padding='post')

    return contents,labels



tag_check = {
    "I":["B","I"],
    "E":["B","I"],
}


def check_label(front_label,follow_label):
    '''

    :param front_label: index=i，前一个tag
    :param follow_label:  index=i时的tag
    :return:
    '''
    if not follow_label:
        '''判断当前是否存在tag'''
        raise Exception("follow label should not both None")

    if not front_label:
        '''判断前一个状态是否存在tag'''
        return True

    if follow_label.startswith("B-"):
        return False

    '''
    当前tag以  I- 或者 E- 开头都符合要求，表示实体还在继续或者结束
    
    前一个tag的结尾是当前tag 的结尾，表示还在一个实体中
    前一个tag的开头部分在tag_check中，表示符合实体规则
    
    '''
    if (follow_label.startswith("I-") or follow_label.startswith("E-")) and \
        front_label.endswith(follow_label.split("-")[1]) and \
        front_label.split("-")[0] in tag_check[follow_label.split("-")[0]]:
        return True
    return False


def format_result(chars, tags):
    entities = []
    entity = []
    # print(chars)['国', '家', '发', '展', '计', '划', '委', '员', '会', '副', '主', '任', '王', '春', '正']
    # print(tags)['B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'E-ORG', 'O', 'O', 'O', 'B-PER', 'I-PER', 'E-PER']
    for index, (char, tag) in enumerate(zip(chars, tags)):
        # if index>0:
        #     print(tag,tags[index - 1]) #  I-ORG I-ORG
        entity_continue = check_label(tags[index - 1] if index > 0 else None, tag) #判断是否符合实体，Flase截断
        # print(index,entity_continue) #8 True

        if not entity_continue and entity:
            entities.append(entity)
            entity = []
        entity.append([index, char, tag, entity_continue])

    if entity:
        entities.append(entity)



    '''
    entities_result 样例
    {
            "begin": 1,
            "end": 9,
            "words": "国家发展计划委员会",
            "type": "ORG"
        },
    '''

    entities_result = []
    for entity in entities:
        print(entity)
        if entity[0][2].startswith("B-"):
            entities_result.append(
                {"begin": entity[0][0] + 1,
                 "end": entity[-1][0] + 1,
                 "words": "".join([char for _, char, _, _ in entity]),
                 "type": entity[0][2].split("-")[1]
                 }
            )

    return entities_result



if __name__ == "__main__":
    text = ['国','家','发','展','计','划','委','员','会','副','主','任','王','春','正']
    tags =  ['B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'E-ORG', 'O', 'O', 'O', 'B-PER', 'I-PER', 'E-PER']
    entities_result= format_result(text,tags)
    print(json.dumps(entities_result, indent=4, ensure_ascii=False))

