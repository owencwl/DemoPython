import  jieba
import  jieba.analyse
from  collections import Counter
import copy
from wordcloud import WordCloud

import matplotlib.pyplot  as plt


testStr="同时如果你希望使用其它分词工具,那么你可以留意我之后的博客,我会在接下来的日子里发布其他有关内容."

test="习近平指出中共十九大规划了中国从现在到本世纪中叶的发展蓝图，宣示了中方愿同各方推动构建人类命运共同体的真诚愿望。政党在国家政治生活中发挥着重要作用，也是推动人类文明进步的重要力量。年终岁末，来自世界各国近300个政党和政治组织的领导人齐聚北京，共商合作大计，充分体现了大家对人类发展和世界前途的关心"

result=jieba.cut(test+testStr)
# seg_list = jieba.cut(text, cut_all=False)  #精确模式（默认是精确模式）
# seg_list = jieba.cut(text)  # 精确模式（默认是精确模式）
# print("[精确模式]: ", "/ ".join(seg_list))

# seg_list2 = jieba.cut(text, cut_all=True)    #全模式
# print("[全模式]: ", "/ ".join(seg_list2))

# seg_list3 = jieba.cut_for_search(text)    #搜索引擎模式
# print("[搜索引擎模式]: ","/ ".join(seg_list3))
# data=dict(Counter(result))

# tags = jieba.analyse.extract_tags(test, topK=5)
# print("关键词:    ", " / ".join(tags))





txt1 = open('word.txt', 'r', encoding='utf8').read()
words_ls = jieba.cut(txt1)

# ss=copy.deepcopy(words_ls)

# data=dict(Counter(words_ls))
# for k,v in data.items():
#     print(k,v)


words_split = "/".join(words_ls)
print(words_split)

wc = WordCloud()    # 字体这里有个坑，一定要设这个参数。否则会显示一堆小方框wc.font_path="simhei.ttf"   # 黑体
wc.font_path="simhei.ttf"
my_wordcloud = wc.generate(words_split)
plt.imshow(my_wordcloud)
plt.axis("off")
plt.show()

# wc.to_file('zzz.png') # 保存图片文件
