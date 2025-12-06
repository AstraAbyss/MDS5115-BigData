import pandas as pd
import numpy as np
from sts.data import tokenize, build_vocabulary, vectorize_corpus, Corpus, Document
from sts.anchor import AnchorInitializer
from sts.em import EMRunner
from sts.viz import plot_top_words, plot_prevalence_sentiment
import os

# -------------------------- 1. 配置参数 --------------------------
CSV_PATH = "data.csv"  # CSV文件路径
TEXT_COL = "text"      # CSV中文本列名
COVARIATE_COL = "rating"  # CSV中协变量列名（如评分）
K = 5  # 预设主题数（可根据数据调整，如3-10）
MIN_WORD_COUNT = 5  # 过滤低频词（出现次数<该值的词会被剔除）
MAX_VOCAB_SIZE = 10000  # 最大词汇表大小
MAX_EM_ITER = 50  # EM迭代上限
CONVERGENCE_THRESH = 1e-6  # ELBO收敛阈值
OUTPUT_DIR = "output_csv"  # 结果保存目录

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------- 2. 读取并预处理CSV数据 --------------------------
print("加载CSV数据...")
df = pd.read_csv(CSV_PATH)

# 过滤空值（关键：避免数据异常）
df = df.dropna(subset=[TEXT_COL, COVARIATE_COL])
df = df[df[TEXT_COL].str.strip() != ""]  # 过滤空文本
print(f"有效数据量：{len(df)} 条")

# 提取文本和协变量
texts = df[TEXT_COL].tolist()  # 文本列表
covariates = df[COVARIATE_COL].values.reshape(-1, 1)  # 协变量矩阵（N×1，N为文档数）

# -------------------------- 3. 文本向量化：构建Corpus对象 --------------------------
print("文本分词...")
# 分词（支持中文/英文，自动过滤标点、停用词）
tokenized_texts = [tokenize(text) for text in texts]

print("构建词汇表...")
# 构建词汇表（过滤低频词）
vocab = build_vocabulary(
    tokenized_texts,
    min_count=MIN_WORD_COUNT,
    max_size=MAX_VOCAB_SIZE
)
print(f"词汇表大小：{len(vocab)}")

print("生成文档-词计数矩阵（DTM）...")
# 生成稀疏DTM（文档-词计数矩阵）
dtm = vectorize_corpus(tokenized_texts, vocab)

# 封装为Corpus对象（模型输入格式）
corpus = Corpus(
    dtm=dtm,
    vocab=vocab,
    covariates=covariates  # 协变量传入模型，用于情感调制
)

# -------------------------- 4. 锚点初始化（加速EM收敛） --------------------------
print("锚点词初始化主题-词分布...")
anchor_initializer = AnchorInitializer(corpus, K=K)
beta0 = anchor_initializer.initialize()  # 初始主题-词分布（K×V矩阵）

# -------------------------- 5. 启动EM训练 --------------------------
print("启动EM迭代训练...")
trainer = EMRunner(
    corpus=corpus,
    K=K,
    beta0=beta0,
    max_iter=MAX_EM_ITER,
    convergence_threshold=CONVERGENCE_THRESH
)

# 执行训练（返回最终模型参数）
model_params = trainer.train()
print("训练完成！")

# -------------------------- 6. 结果可视化与保存 --------------------------
print("生成可视化结果...")
# 1. 绘制每个主题的Top10词（中位情感水平）
plot_top_words(
    model_params["kappa_t"],  # 主题基准词分布
    model_params["kappa_s"],  # 情感调制系数
    vocab,  # 词汇表
    n_top_words=10,
    save_path=os.path.join(OUTPUT_DIR, "top_words_csv.png")
)

# 2. 绘制情感潜变量与协变量（评分）的散点图
# 从模型参数中提取文档潜变量a_d（情感+流行度，此处取情感维度，默认是第2维）
a_d = np.array([doc_params["a"] for doc_params in trainer.document_params])
sentiment_latent = a_d[:, 1]  # 情感潜变量（根据模型定义调整维度）

plot_prevalence_sentiment(
    sentiment_latent=sentiment_latent,
    covariate=covariates.squeeze(),  # 原始协变量（评分）
    save_path=os.path.join(OUTPUT_DIR, "sentiment_vs_rating_csv.png")
)

# 保存模型参数（可用于后续推理）
np.savez(
    os.path.join(OUTPUT_DIR, "model_params_csv.npz"),
    kappa_t=model_params["kappa_t"],
    kappa_s=model_params["kappa_s"],
    Gamma=model_params["Gamma"],
    Sigma=model_params["Sigma"],
    vocab=vocab
)
print(f"所有结果已保存到：{OUTPUT_DIR}")