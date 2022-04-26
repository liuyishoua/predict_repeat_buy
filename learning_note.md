### 天猫复购率预测

1. 数据探索
2. 特征工程，重点
3. 模型训练
4. 特征选择及优化

#### 数据探索

.head() , .info(), .describe()探索数据基本信息

isna().sum()判断缺失数据。没有意义的数据一并按缺失处理，查看缺失数据与没有意义的数据的数量

```python
user_info['age_range'].isna() | user_info['age_range'] == 0
```

value_counts() 处理列或多列（与下面的sort_values()相似），离散数据（类别数据），比如判断样本预测值的分布。

可视化自变量因素对因变量（正负样本的影响），加深对数据的理解。

自变量有离散和连续两种变量。离散量的可视化分析可以使用sns.countplot画柱状图

连续量可使用sns.distplot画直方图，设置 fit=stats.norm (from scipy import stats)，比较连续量与正态分布的区别。也可使用stat.probplot画QQ（分位数-分位数）图，比较与该变量与正态分布的联系。

#### 特征工程

import gc   gc.collect回收内存。一般先del 变量，之后使用gc.collect回收内存。

from collections import Counter     collections集合中的数据结构，计数器，与dict结构类似。给重复的值计数，可用于找出最常出现的数据值以及其重复的次数。

实际业务场景一般出现多表，这时候就需要进行联表。将所有相关信息联表，就便于处理后续生成有效特征，也方便训练预测。

操作如下，相当于做笛卡尔积，左联（数据库联表查询也有这个性质）。

```python
all_data.merge(user_info,on=['user_id'],how='left') 
```

sort_values指定framedata相应的列进行排序。

```python
user_log = user_log.sort_values(['user_id','time_stamp'])
```

自定义聚合函数很有必要，方便操作。

```python
list_join_func = lambda x: " ".join([str(i) for i in x])
agg_dict = {
            'item_id' : list_join_func, 
            'cat_id' : list_join_func}
rename_dict = {
            'item_id' : 'item_path',
            'cat_id' : 'cat_path'}
user_log_path = user_log.groupby('user_id').agg(agg_dict).reset_index().rename(columns=rename_dict) 
```

lambda函数的使用：

```pythonn
lambda x: 返回的变量
```

除了对数据进行聚合处理，还需要定义一些基函数，用于扩展每一行的基本信息。如下：

```python
# set统计唯一数据，用的好
def nunique_(x):
    try:
        return len(set(x.split(' ')))
    except:
        return -1
def user_nunique(df_data, single_col, name):
    df_data[name] = df_data[single_col].apply(nunique_)
    return df_data
# 用户查看不同店铺个数
all_data_test = user_nunique(all_data_test,  'seller_path', 'seller_nunique')
# 用户查看不同品类个数
all_data_test = user_nunique(all_data_test,  'cat_path', 'cat_nunique')
```

处理行内的杂乱信息，转化成业务中有意义的信息。（就算没意义也没关系，后期可通过特征选择筛掉）

定义如上user_nunique基函数，之后就是将有意义的变量一通使用，一大堆特征就生成了。

由于行内有文本信息，除去如上基函数的一些提取特征的操作，还可以利用TF-IDF，word2vec和stacking提取特征。用这些文本分析提取的特征可就多了。

```python
# TF-IDF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
tfidfVec = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 1), max_features=100)
columns_list = ['seller_path']
for i, col in enumerate(columns_list):
    all_data_test[col] = all_data_test[col].astype(str)
    tfidfVec.fit(all_data_test[col])
    data_ = tfidfVec.transform(all_data_test[col])
    if i == 0:
        data_cat = data_
    else:
        data_cat = sparse.hstack((data_cat, data_))

# 特征合并
df_tfidf = pd.DataFrame(data_cat.toarray())
df_tfidf.columns = ['tfidf_' + str(i) for i in df_tfidf.columns]
all_data_test = pd.concat([all_data_test, df_tfidf],axis=1)
```

```python
# word2vec
import gensim
# Train Word2Vec model
# window表示词向量模型扫瞄前后对的窗口数为5.
# min_count表示最低频率，如果一个词语在文档钟出现次数少于5，就会丢弃
model = gensim.models.Word2Vec(all_data_test['seller_path'].apply(lambda x: x.split(' ')), window=5, min_count=5, workers=4)
def mean_w2v_(x, model, size=100):
    try:
        i = 0
        for word in x.split(' '):
            if word in model.wv.vocab:
                i += 1
                if i == 1:
                    vec = np.zeros(size)
                vec += model.wv[word]
        return vec / i 
    except:
        return  np.zeros(size)
def get_mean_w2v(df_data, columns, model, size):
    data_array = []
    for index, row in df_data.iterrows():
        w2v = mean_w2v_(row[columns], model, size)
        data_array.append(w2v)
    return pd.DataFrame(data_array)
df_embeeding = get_mean_w2v(all_data_test, 'seller_path', model, 100)
df_embeeding.columns = ['embeeding_' + str(i) for i in df_embeeding.columns]
```

TF-IDF仅考虑词语在文中出现的频率，没有考虑句子之间的上下文关系，因此出现了word2vec。

word2vec有两种方式学习上下文信息：1. skim-gram          2. cbow

1. skim-gram：输入一个词语，预测其上下文。希望预测和实际的上下文的差距越小越好。损失函数使用softmax（行向量与列向量乘积）+交叉熵。
2. cbow：输入上下文，预测中心词。

one-hot编码进行文本预处理，并且将模型中的隐层作为word2vec特征化的结果。模型没有使用激活函数，包括输入层，隐层，输出层，均是矩阵相乘操作。



