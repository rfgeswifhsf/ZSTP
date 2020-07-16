import os
from pyltp import Segmentor,Postagger,Parser,NamedEntityRecognizer,SementicRoleLabeller
class LtpParser:
    def __init__(self):
        LTP_PATH = '/root/tmp/pycharm_project_96/pyltp_test/ltp_data'

        # 分词
        self.segmentor = Segmentor()
        self.segmentor.load(os.path.join(LTP_PATH,'cws.model'))
        # 词性标注
        self.postagger = Postagger()
        self.postagger.load(os.path.join(LTP_PATH,'pos.model'))
        # 依存句法
        self.parser = Parser()
        self.parser.load(os.path.join(LTP_PATH,'parser.model'))
        # 命名实体识别
        self.recognizer = NamedEntityRecognizer()
        self.recognizer.load(os.path.join(LTP_PATH,'ner.model'))
        # # 语义角色标注
        self.labeller = SementicRoleLabeller()
        self.labeller.label(os.path.join(LTP_PATH,'pisrl.model'))

    def format_labeller(self,words,postags):
        '''语义角色标注'''
        arcs = self.parser.parse(words,postags)
        roles = self.labeller.label(words,postags,arcs)
        roles_dict = {}
        for role in roles:
            #role.index代表谓词的索引
            #role.arguments 代表关于该谓词的若干语义角色
            # arg.name表示语义角色类型
            # arg.range.start表示该语义角色起始词的开始位置索引
            # arg.range.end表示该语义角色起始词的结束位置索引
            roles_dict[role.index] = {arg.name:{arg.name,arg.range.start,arg.range.end} for arg in role.arguments}
        return roles_dict

    def build_parser_child_dict(self,words,postags,arcs):
        '''句法分析，为句子中的每个词语维护一个保存句法依存儿子节点的字典'''
        child_dict_list = []
        format_parser_list = []
        for index in range(len(words)):# 循环每个词
            child_dict = dict() # 存储的格式为{词关系（ATT）: 词}
            for arc_index in range(len(arcs)):
                #arc_index为当前词的索引，arcs[arc_index]为一个元祖（arc.head,arc.relation）
                if arcs[arc_index].head ==index+1:# 去找到父节点为当第一个for里面索引的词，即当前词的子节点
                    if arcs[arc_index].relation in child_dict:
                        child_dict[arcs[arc_index].relation].append(arc_index)
                    else:
                        child_dict[arcs[arc_index].relation] = []
                        child_dict[arcs[arc_index].relation].append(arc_index)
            child_dict_list.append(child_dict)
        rely_id = [arc.head for arc in arcs]# 提取依存父节点id
        relation = [arc.relation for arc in arcs]# 提取关系依存
        heads = ['Root' if id==0 else words[id-1] for id in rely_id]# 匹配依存父节点词语
        for i in range(len(words)):
            a = [relation[i],words[i],i,postags[i],heads[i],rely_id[i]-1,postags[rely_id[i]-1]]
            format_parser_list.append(a)
        return child_dict_list,format_parser_list
    def parser_main(self,sent):
        '''依存关系分析的主函数'''
        words = list(self.segmentor.segment(sent))
        postags = list(self.postagger.postag(words))
        arcs = self.parser.parse(words,postags)
        child_dict_list, format_parser_list = self.build_parser_child_dict(words,postags,arcs)
        roles_dict = self.format_labeller(words,postags)
        return words,postags,child_dict_list,format_parser_list,roles_dict

if __name__ == '__main__':
    sent = '''
    唐纳德·特朗普（Donald Trump），1946年6月14日生于纽约，美国共和党籍政治家、企业家、商人，第45任美国总统。
1968年从宾夕法尼亚大学沃顿商学院毕业后，进入其父的房地产公司工作，并在1971年开始掌管公司运营，正式进军商界。在随后几十年间，特朗普开始建立自己的房地产王国，人称“地产之王”。除房地产外，特朗普将投资范围延伸到其他行业，包括开设赌场、高尔夫球场等。他还涉足娱乐界，是美国真人秀《名人学徒》等电视节目的主持人，并担任“环球小姐”选美大赛主席。美国杂志《福布斯》曾评估特朗普资产净值约为45亿美元，特朗普则称超过100亿美元。
特朗普在过去20年间分别支持过共和党和民主党各主要总统竞选者。2015年6月，特朗普以共和党竞选者身份正式参加2016年美国总统选举。此前，特朗普没有担任过公共职务。特朗普结过3次婚，育有5个子女。 [1]  2016年11月9日，唐纳德·特朗普已获得了276张选举人票，超过270张选举人票的获胜标准，当选美国第45任总统。 [2]  2017年1月20日，特朗普正式成为美国第45任总统。
2018年8月7日，西好莱坞市市议会投票通过“特朗普之星”将永久移除
    '''
    parser = LtpParser()

    words, postags, child_dict_list, format_parser_list,roles_dict = parser.parser_main(sent)
    print(words)
    print(postags)
    print(child_dict_list)
    print(format_parser_list)
    print(roles_dict)



