#!/usr/bin/env python
# coding=utf-8
"""
文本中事实三元组抽取
python *.py input.txt output.txt begin_line end_line
"""

# Set your own model path
MODELDIR = "/root/tmp/pycharm_project_96/pyltp_test/ltp_data"

import os

from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer

print("正在加载LTP模型... ...")

segmentor = Segmentor()
segmentor.load(os.path.join(MODELDIR, "cws.model"))

postagger = Postagger()
postagger.load(os.path.join(MODELDIR, "pos.model"))

parser = Parser()
parser.load(os.path.join(MODELDIR, "parser.model"))

recognizer = NamedEntityRecognizer()
recognizer.load(os.path.join(MODELDIR, "ner.model"))


print("加载模型完毕。")

in_file_name = "input.txt"
out_file_name = "output.txt"
begin_line = 1
end_line = 0


def extraction_start(in_file_name, out_file_name):
    """
    事实三元组抽取的总控程序
    Args:
        in_file_name: 输入文件的名称
        #out_file_name: 输出文件的名称
    """
    in_file = open(in_file_name, 'r')
    out_file = open(out_file_name, 'a+')

    for sentence in in_file.readlines():
        if sentence == '':
            continue
        else:
            try:
                fact_triple_extract(sentence, out_file)
                out_file.flush()
            except Exception as e:
                print('61--line--',e)
        break
    in_file.close()
    out_file.close()


def fact_triple_extract(sentence, out_file):
    """
    对于给定的句子进行事实三元组抽取
    Args:
        sentence: 要处理的语句
    """
    # 分词
    words = segmentor.segment(sentence)
    print(len(words))
    print('词------','--'.join(words))
    #词性标注
    postags = postagger.postag(words)
    print('词性标注-----','\t'.join(postags))
    # 实体识别
    netags = recognizer.recognize(words, postags)
    print('实体识别-----','--'.join(netags))
    # 句法依存
    arcs = parser.parse(words, postags)
    print ('句法依存----',"\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))


    child_dict_list = build_parse_child_dict(words, postags, arcs)
    print('cccc',child_dict_list)


    for index in range(len(postags)):
        # 抽取以谓词为中心的事实三元组
        if postags[index] == 'v':
            child_dict = child_dict_list[index]

            # 主谓宾
            if 'SBV' in child_dict and 'VOB' in child_dict:
                e1 = complete_e(words, postags, child_dict_list, child_dict['SBV'][0])

                r = words[index]
                e2 = complete_e(words, postags, child_dict_list, child_dict['VOB'][0])

                out_file.write("主语谓语宾语关系\t(%s, %s, %s)\n" % (e1, r, e2))
                out_file.flush()

            # 定语后置，动宾关系
            if arcs[index].relation == 'ATT':
                if 'VOB' in child_dict:
                    e1 = complete_e(words, postags, child_dict_list, arcs[index].head - 1)
                    r = words[index]
                    e2 = complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                    temp_string = r + e2
                    if temp_string == e1[:len(temp_string)]:
                        e1 = e1[len(temp_string):]
                    if temp_string not in e1:
                            out_file.write("定语后置动宾关系\t(%s, %s, %s)\n" % (e1, r, e2))
                            out_file.flush()

            # 含有介宾关系的主谓动补关系
            if 'SBV' in child_dict and 'CMP' in child_dict:
                e1 = complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
                cmp_index = child_dict['CMP'][0]
                r = words[index] + words[cmp_index]
                if 'POB' in child_dict_list[cmp_index]:
                    e2 = complete_e(words, postags, child_dict_list, child_dict_list[cmp_index]['POB'][0])

                    out_file.write("介宾关系主谓动补\t(%s, %s, %s)\n" % (e1, r, e2))
                    out_file.flush()

        # 尝试抽取命名实体有关的三元组
        if netags[index][0] == 'S' or netags[index][0] == 'B':
            ni = index
            if netags[ni][0] == 'B':
                while netags[ni][0] != 'E':
                    ni += 1
                e1 = ''.join(words[index:ni + 1])
            else:
                e1 = words[ni]
            if arcs[ni].relation == 'ATT' and postags[arcs[ni].head - 1] == 'n' and netags[arcs[ni].head - 1] == 'O':
                r = complete_e(words, postags, child_dict_list, arcs[ni].head - 1)
                if e1 in r:
                    r = r[(r.index(e1) + len(e1)):]
                if arcs[arcs[ni].head - 1].relation == 'ATT' and netags[arcs[arcs[ni].head - 1].head - 1] != 'O':
                    e2 = complete_e(words, postags, child_dict_list, arcs[arcs[ni].head - 1].head - 1)
                    mi = arcs[arcs[ni].head - 1].head - 1
                    li = mi
                    if netags[mi][0] == 'B':
                        while netags[mi][0] != 'E':
                            mi += 1
                        e = ''.join(words[li + 1:mi + 1])
                        e2 += e
                    if r in e2:
                        e2 = e2[(e2.index(r) + len(r)):]
                    if r + e2 in sentence:
                            out_file.write("人名//地名//机构\t(%s, %s, %s)\n" % (e1, r, e2))
                            out_file.flush()

    # return  words,postags,arcs
def build_parse_child_dict(words, postags, arcs):
    """
    为句子中的每个词语维护一个保存句法依存儿子节点的字典
    Args:
        words: 分词列表
        postags: 词性列表
        arcs: 句法依存列表
    """
    child_dict_list = []
    for index in range(len(words)):
        child_dict = dict()
        for arc_index in range(len(arcs)):
            if arcs[arc_index].head == index + 1:
                # print(arcs[arc_index].head,arcs[arc_index].relation,arc_index)
                if arcs[arc_index].relation in child_dict:
                    child_dict[arcs[arc_index].relation].append(arc_index)
                else:
                    child_dict[arcs[arc_index].relation] = []
                    child_dict[arcs[arc_index].relation].append(arc_index)

        child_dict_list.append(child_dict)
    return child_dict_list


def complete_e(words, postags, child_dict_list, word_index):
    """
    完善识别的部分实体
    """
    child_dict = child_dict_list[word_index]
    prefix = ''
    if 'ATT' in child_dict:
        for i in range(len(child_dict['ATT'])):
            prefix += complete_e(words, postags, child_dict_list, child_dict['ATT'][i])

    postfix = ''
    if postags[word_index] == 'v':
        if 'VOB' in child_dict:
            postfix += complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
        if 'SBV' in child_dict:
            prefix = complete_e(words, postags, child_dict_list, child_dict['SBV'][0]) + prefix
    return prefix + words[word_index] + postfix




if __name__ == "__main__":
    extraction_start(in_file_name, out_file_name)

