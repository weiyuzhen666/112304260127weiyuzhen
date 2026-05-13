import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import os

def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review, features="html.parser").get_text()
    review_text = review_text.replace('!', ' EXCLAMATION ')
    review_text = review_text.replace('?', ' QUESTION ')
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    
    processed_words = []
    for word in words:
        if word == 'exclamation':
            processed_words.append('!')
        elif word == 'question':
            processed_words.append('?')
        else:
            processed_words.append(word)
    
    stops = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
        'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
        'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
        'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'should', 'now'
    }
    
    # 保留否定词
    negation_words = set(['not', 'no', 'never', 'nor', 'none', 'nothing', 'nobody', 'nowhere', 'neither', 'hardly', 'scarcely', 'barely'])
    meaningful_words = [w for w in processed_words if not w in stops or w in negation_words]
    return(" ".join(meaningful_words))

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
    print("=" * 60)
    print("集成模型 - 输出概率结果")
    print("=" * 60)
    
    print("\n1. 读取训练数据...")
    train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    
    print("2. 清理训练数据...")
    num_reviews = train["review"].size
    clean_train_reviews = []
    train_tokens = []
    for i in range(0, num_reviews):
        if (i+1) % 1000 == 0:
            print(f"   已处理 {i+1} / {num_reviews} 条评论")
        clean_review = review_to_words(train["review"][i])
        clean_train_reviews.append(clean_review)
        train_tokens.append(simple_preprocess(clean_review, deacc=True))
    
    print("\n3. 创建TF-IDF特征...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    
    train_data_features = tfidf_vectorizer.fit_transform(clean_train_reviews)
    print(f"   TF-IDF特征形状: {train_data_features.shape}")
    
    print("\n4. 训练Word2Vec模型...")
    embedding_dim = 150
    word2vec_model = Word2Vec(
        sentences=train_tokens,
        vector_size=embedding_dim,
        window=5,
        min_count=3,
        epochs=10,
        workers=4
    )
    
    print("5. 提取Word2Vec特征...")
    X_word2vec_train = np.array([get_sentence_embedding(tokens, word2vec_model, embedding_dim) for tokens in train_tokens])
    print(f"   Word2Vec特征形状: {X_word2vec_train.shape}")
    
    print("\n6. 合并特征...")
    from scipy.sparse import hstack
    X_combined_train = hstack([train_data_features, X_word2vec_train]).toarray()
    print(f"   合并特征形状: {X_combined_train.shape}")
    
    print("\n7. 训练逻辑回归...")
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_combined_train, train["sentiment"])
    
    print("\n8. 读取和处理测试数据...")
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
    
    num_reviews = len(test["review"])
    clean_test_reviews = []
    test_tokens = []
    for i in range(0, num_reviews):
        if (i+1) % 1000 == 0:
            print(f"   已处理 {i+1} / {num_reviews} 条评论")
        clean_review = review_to_words(test["review"][i])
        clean_test_reviews.append(clean_review)
        test_tokens.append(simple_preprocess(clean_review, deacc=True))
    
    test_data_features = tfidf_vectorizer.transform(clean_test_reviews)
    X_word2vec_test = np.array([get_sentence_embedding(tokens, word2vec_model, embedding_dim) for tokens in test_tokens])
    X_combined_test = hstack([test_data_features, X_word2vec_test]).toarray()
    
    print("\n9. 预测（输出概率）...")
    result_proba = lr.predict_proba(X_combined_test)[:, 1]
    
    print("\n10. 保存提交文件（包含概率）...")
    # 确保id列不包含多余的引号
    submission_df = pd.DataFrame({'id': test['id'].astype(str).str.strip('"'), 'sentiment': result_proba})
    submission_df.to_csv("Submission_With_Probabilities.csv", index=False, quoting=3)
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print("\n📁 生成的文件: Submission_With_Probabilities.csv")
    print("💡 这个文件包含预测概率，适用于ROC AUC评估！")

if __name__ == "__main__":
    main()
