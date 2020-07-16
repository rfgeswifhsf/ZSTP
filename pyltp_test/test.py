# 分句子
from pyltp import SentenceSplitter
sen = SentenceSplitter.split('元芳你怎么看？我就趴窗口上看呗！')
print('\n'.join(sen))



# 分词
import os
from pyltp import Segmentor
LTP_DATA_DIR='./ltp_data'
cws_model_path=os.path.join(LTP_DATA_DIR,'cws.model')
segmentor=Segmentor()
segmentor.load(cws_model_path)
words=segmentor.segment('熊高雄你吃饭了吗')
print(list(words))
print('\t'.join(words))
segmentor.release()


# 自定义词典
LTP_DATA_DIR='./ltp_data'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
segmentor = Segmentor()  # 初始化实例
segmentor.load_with_lexicon(cws_model_path, 'newwordsdict') # 加载模型，第二个参数是您的外部词典文件路径
words = segmentor.segment('马云是阿里巴巴的创始人')
print('\t'.join(words))
segmentor.release()


# 词性标准
LTP_DATA_DIR='./ltp_data'
# ltp模型目录的路径
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`

from pyltp import Postagger
postagger = Postagger() # 初始化实例
postagger.load(pos_model_path)  # 加载模型

# words = ['元芳', '你', '怎么', '看']  # 分词结果
words = list(words) #李开心	说	亚硝酸盐	是	一	种	化学	物质
postags = postagger.postag(words)  # 词性标注

print('\t'.join(postags))
postagger.release()  # 释放模型


# 实体识别
import os
LTP_DATA_DIR='./ltp_data' # ltp模型目录的路径
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`

from pyltp import NamedEntityRecognizer
recognizer = NamedEntityRecognizer() # 初始化实例
recognizer.load(ner_model_path)  # 加载模型

words = list(words)
postags = list(postags)
netags = recognizer.recognize(words, postags)  # 命名实体识别

print('\t'.join(netags))
recognizer.release()  # 释放模型


# 句法依存
import os
LTP_DATA_DIR='./ltp_data' # ltp模型目录的路径
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`

from pyltp import Parser
parser = Parser() # 初始化实例
parser.load(par_model_path)  # 加载模型

words = list(words)
postags = list(postags)
arcs = parser.parse(words, postags)  # 句法分析

print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
parser.release()  # 释放模型

for i in range(len(words)):
    print(arcs[i].head,arcs[i].relation,words[i])

