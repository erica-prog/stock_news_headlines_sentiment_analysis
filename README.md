# StockSentimentAI: Financial News Sentiment Analysis for Stock Insights

**StockSentimentAI** is a financial news analytics project designed to predict the sentiment of stock headlines and assess how well different machine learning and deep learning models can classify market sentiment. By combining traditional methods like TF-IDF with SVMs and cutting-edge deep learning models like BiLSTM, BERT, and FinBERT, this project evaluates which approaches best uncover the emotional tone behind stock market news.

The project leverages a dataset of over 5,000 stock news headlines collected via the EODHD API from 2021 to 2022, focusing on major companies like AAPL, TSLA, NVDA, and others. Through advanced feature engineering, hyperparameter tuning, and model experimentation, StockSentimentAI offers a thorough comparison of financial text mining methods for sentiment prediction.

## Motivation

Financial markets are heavily influenced by the tone and sentiment of news coverage. Positive headlines can drive stock prices upward, while negative ones can cause sharp declines.

This project addresses a critical question: **Can we reliably detect sentiment from stock headlines using AI models, and which models are best suited for this task?**

By exploring multiple techniques across classical and deep learning paradigms, StockSentimentAI seeks to contribute to more robust financial forecasting tools and enhance the understanding of news-based sentiment in markets.

## Machine Learning Methods and Results

### TF-IDF + Traditional Machine Learning

TF-IDF vectorization was used to convert news headlines into numerical representations. Several supervised models were applied:

- Support Vector Machine (SVM) achieved the highest performance with a test accuracy of 81.76%, showcasing the strength of simple textual representations combined with robust classifiers.

- Random Forest and Naive Bayes classifiers achieved competitive but lower accuracies (around 77–78%).

These results demonstrate that, in financial text analysis, traditional feature engineering still holds significant predictive power when paired with strong algorithms.

### Deep Learning – BiLSTM Models
Five BiLSTM architectures were explored, varying in depth, width, and regularization (dropout):

- Model A (Small BiLSTM + Dropout): Rapid overfitting, validation plateaued at ~78%.

- Model B (Medium BiLSTM, No Dropout): Severe overfitting, validation accuracy around 75%.

- Model C (Large BiLSTM + Dropout): Slight performance improvement, but still moderate overfitting.

- Model D (Balanced BiLSTM + Dropout): Improved generalization with ~76% validation accuracy.

- Model E (Deep BiLSTM, No Dropout): **Best deep learning model**, achieving a **stable validation accuracy around 80%** without heavy overfitting.

These results highlight that deeper architectures, even without dropout, can outperform shallower ones if model capacity matches task complexity.

### Transformer Models – BERT and FinBERT
Transformer-based models were evaluated using pre-trained BERT and FinBERT embeddings combined with Random Forest and SVM classifiers.

- Both BERT and FinBERT **plateaued at ~63.81% test accuracy**, likely due to:

  - Lack of domain-specific fine-tuning.
  
  - Small dataset size compared to what Transformers require to generalize effectively.

Notably, FinBERT exhibited a heavy bias toward Neutral sentiment classifications, while BERT models leaned more heavily on Positive/Negative labels.

## Challenges and Reflections

### Challenges
- Overfitting in shallow BiLSTM models despite using dropout, requiring careful tuning.

- Small sample size (5,000 headlines) constrained the effectiveness of large Transformer models without extensive fine-tuning.

- Sentiment imbalance: A skew toward Positive sentiments made model calibration challenging.

### Reflections
- Simple models still win: TF-IDF with SVM consistently outperformed even complex deep models in this context.

- Deep capacity matters: Deeper BiLSTM networks provided superior stability and generalization compared to shallow ones.

- Domain adaptation is crucial: Pre-trained Transformers like FinBERT underperformed without task-specific fine-tuning.

## Conclusion

StockSentimentAI demonstrates that **traditional machine learning models** like TF-IDF + SVM can still outperform deep and Transformer-based models for moderate-sized, domain-specific sentiment analysis tasks.

Deep BiLSTM architectures also show strong promise when tuned correctly, achieving near SVM-level performance.

**Key findings:**
- TF-IDF + SVM achieved the highest test accuracy (**81.76%**).

- Deep BiLSTM (Model E) achieved strong validation performance (**~80%**).

- BERT/FinBERT struggled (**~63.8%**), highlighting the importance of fine-tuning in Transformer use cases.

Future work will involve scraping fresh headlines from 2023–2024, fine-tuning BERT/FinBERT using TensorFlow Data Pipelines, and exploring multi-head attention-based models for enhanced generalization.

By tailoring model choice to data size, task domain, and resource availability, businesses and researchers can build smarter, faster, and more reliable financial sentiment engines.

**For more details, please refer to the [final code](new_headlines_with_the_single_headlines_datasets.py).**
