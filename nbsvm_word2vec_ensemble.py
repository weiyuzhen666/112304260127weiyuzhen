import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# 文本预处理函数
def preprocess_text(text):
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 转换为小写
    text = text.lower()
    # 保留情感标点
    text = re.sub(r'(!+)', ' EXCLAMATION ', text)
    text = re.sub(r'([?]+)', ' QUESTION ', text)
    # 去除其他标点和数字
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # 分词
    tokens = simple_preprocess(text, deacc=True)
    # 过滤停用词，但保留否定词
    stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
    negation_words = set(['not', 'no', 'never', 'nor', 'none', 'nothing', 'nobody', 'nowhere', 'neither', 'hardly', 'scarcely', 'barely'])
    filtered_tokens = [token for token in tokens if token not in stopwords or token in negation_words]
    # 保留长度大于1的单词
    filtered_tokens = [token for token in filtered_tokens if len(token) > 1]
    return filtered_tokens

# 计算句子的均值embedding
def get_sentence_embedding(tokens, model, embedding_dim):
    embeddings = []
    for token in tokens:
        if token in model.wv:
            embeddings.append(model.wv[token])
    if not embeddings:
        return np.zeros(embedding_dim)
    return np.mean(embeddings, axis=0)

# 主函数
def main():
    # 读取数据
    print("Reading data...")
    train_df = pd.read_csv('labeledTrainData.tsv', sep='\t', quoting=3)
    test_df = pd.read_csv('testData.tsv', sep='\t', quoting=3)
    
    # 预处理训练数据
    print("Preprocessing training data...")
    train_tokens = []
    for review in train_df['review']:
        tokens = preprocess_text(review)
        train_tokens.append(tokens)
    
    # 预处理测试数据
    print("Preprocessing test data...")
    test_tokens = []
    for review in test_df['review']:
        tokens = preprocess_text(review)
        test_tokens.append(tokens)
    
    # 训练Word2Vec模型
    print("Training Word2Vec model...")
    embedding_dim = 150
    word2vec_model = Word2Vec(
        sentences=train_tokens,
        vector_size=embedding_dim,
        window=5,
        min_count=3,
        epochs=10,
        workers=4
    )
    
    # 提取Word2Vec特征
    print("Extracting Word2Vec features...")
    X_word2vec_train = np.array([get_sentence_embedding(tokens, word2vec_model, embedding_dim) for tokens in train_tokens])
    X_word2vec_test = np.array([get_sentence_embedding(tokens, word2vec_model, embedding_dim) for tokens in test_tokens])
    
    # 准备NBSVM特征
    print("Preparing NBSVM features...")
    # 将分词结果转换为字符串
    train_texts = [' '.join(tokens) for tokens in train_tokens]
    test_texts = [' '.join(tokens) for tokens in test_tokens]
    
    # 使用TF-IDF向量器
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=8000)
    X_tfidf_train = tfidf_vectorizer.fit_transform(train_texts)
    X_tfidf_test = tfidf_vectorizer.transform(test_texts)
    
    # 合并特征
    print("Combining features...")
    from scipy.sparse import hstack
    X_combined_train = hstack([X_tfidf_train, X_word2vec_train]).toarray()
    X_combined_test = hstack([X_tfidf_test, X_word2vec_test]).toarray()
    
    # 准备标签
    y_train = train_df['sentiment'].values
    
    # 分割验证集
    X_train, X_val, y_train_subset, y_val = train_test_split(X_combined_train, y_train, test_size=0.2, random_state=42)
    
    # 训练逻辑回归模型
    print("Training logistic regression model...")
    clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    clf.fit(X_train, y_train_subset)
    
    # 验证模型
    print("Evaluating model...")
    val_pred = clf.predict(X_val)
    val_prob = clf.predict_proba(X_val)[:, 1]
    val_accuracy = accuracy_score(y_val, val_pred)
    val_auc = roc_auc_score(y_val, val_prob)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")
    
    # 使用全部数据重新训练
    print("Retraining with full data...")
    clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    clf.fit(X_combined_train, y_train)
    
    # 预测测试集
    print("Predicting test data...")
    test_pred = clf.predict(X_combined_test)
    
    # 生成提交文件
    print("Generating submission file...")
    # 确保id列不包含多余的引号
    submission_df = pd.DataFrame({'id': test_df['id'].astype(str).str.strip('"'), 'sentiment': test_pred})
    # 使用quoting=2（QUOTE_MINIMAL）来只对包含特殊字符的字段加引号
    submission_df.to_csv('submission_ensemble.csv', index=False, quoting=2)
    print("Submission file generated: submission_ensemble.csv")
    print("All steps completed successfully!")

if __name__ == "__main__":
    main()
