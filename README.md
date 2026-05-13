# Kaggle Sentiment Analysis - Movie Reviews

基于Bag of Words Meets Bags of Popcorn比赛的情感分析项目，使用多种机器学习模型进行电影评论情感分类。

## 项目介绍

这是一个针对Kaggle比赛"Bag of Words Meets Bags of Popcorn"的实现，目标是对IMDB电影评论进行情感分类（正面/负面）。

## 模型实现

### 1. **Word2Vec + 逻辑回归** (`submission_with_probabilities.py`)
- 使用Word2Vec将文本转换为词向量
- 计算句子的均值embedding作为特征
- 使用逻辑回归进行分类
- 输出概率结果，适合ROC AUC评估

### 2. **NBSVM + Word2Vec集成模型** (`ensemble_with_probabilities.py`)
- 结合TF-IDF（1-2元语法）和Word2Vec特征
- 使用逻辑回归作为分类器
- 输出预测概率，获得更好的性能

### 3. **NBSVM + Word2Vec集成模型** (`nbsvm_word2vec_ensemble.py`)
- 集成NBSVM和Word2Vec的特征
- 使用逻辑回归分类
- 输出二分类结果

## 技术栈

- **Python 3.x**
- **scikit-learn** - 机器学习算法
- **gensim** - Word2Vec模型
- **pandas** - 数据处理
- **BeautifulSoup** - HTML标签处理
- **numpy** - 数值计算

## 项目结构

```
比赛2/
├── ensemble_with_probabilities.py    # 集成模型（输出概率）
├── nbsvm_word2vec_ensemble.py       # NBSVM+Word2Vec集成
├── submission_with_probabilities.py  # 参考实现
├── kaggle-sentiment-popcorn-master/  # 参考项目
└── .gitignore                        # Git忽略规则
```

## 使用方法

1. **准备数据**
   - 下载Kaggle比赛数据
   - 数据文件包括：`labeledTrainData.tsv`, `testData.tsv`, `unlabeledTrainData.tsv`

2. **运行模型**
   ```bash
   # 运行集成模型（推荐）
   python ensemble_with_probabilities.py
   
   # 或运行其他模型
   python nbsvm_word2vec_ensemble.py
   ```

3. **提交结果**
   - 生成的CSV文件可直接提交到Kaggle
   - 包含概率的版本适合ROC AUC评估

## 性能表现

- 验证准确率：约88-89%
- 验证AUC：约95-96%
- 使用全部训练数据训练

## 参考项目

本项目参考了：
- https://github.com/vinhkhuc/kaggle-sentiment-popcorn
- 论文：Ensemble of Generative and Discriminative Techniques for Sentiment Analysis of Movie Reviews

## 数据预处理

- 移除HTML标签
- 保留情感标点符号（! 和 ?）
- 过滤停用词但保留否定词
- 小写化处理
- 移除特殊字符

## 许可证

仅供学习和研究使用。
