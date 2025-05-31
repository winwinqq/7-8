from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 根据学号选取的句子
movie_review = "这部电影太精彩了，节奏紧凑毫无冷场，完全沉浸其中！。"
food_review = "分量太少了，照片看着满满的，实际就几口。"

# 加载预训练模型和分词器
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 情感标签映射
label_map = {0: "负面", 1: "正面"}


def predict_sentiment(text):
    # 文本编码
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # 模型预测
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取预测结果
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_label = torch.argmax(probabilities, dim=1).item()

    return label_map[predicted_label]


# 执行预测
movie_sentiment = predict_sentiment(movie_review)
food_sentiment = predict_sentiment(food_review)

# 输出结果
print(f"影评情感倾向：{movie_sentiment}")
print(f"外卖评价情感倾向：{food_sentiment}")