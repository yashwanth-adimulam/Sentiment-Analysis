"""
pip install transformers torch datasets scikit-learn
"""


import pandas as pd
import json
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
with open('amazon_reviews.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(df.head())

# Preprocess the text function (tokenization, cleaning is handled by BERT tokenizer)
def preprocess_text(text):
    return text

df['processed_text'] = df['text'].apply(preprocess_text)

# Split dataset into train and test sets
X = df['processed_text']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the texts (you'll need to do this for both training and test data)
def tokenize_data(texts):
    return tokenizer(texts.tolist(), padding=True, truncation=True, max_length=512, return_tensors='pt')

train_encodings = tokenize_data(X_train)
test_encodings = tokenize_data(X_test)

# Convert labels to tensor format
train_labels = torch.tensor(y_train.tolist())
test_labels = torch.tensor(y_test.tolist())

# Create a custom dataset class for BERT
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Prepare dataset
train_dataset = SentimentDataset(train_encodings, train_labels)
test_dataset = SentimentDataset(test_encodings, test_labels)

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory for model checkpoints
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch",     # evaluate after each epoch
)

# Define Trainer
trainer = Trainer(
    model=model,                         # the model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,           # evaluation dataset
    compute_metrics=lambda p: {
        'accuracy': accuracy_score(p.label_ids, p.predictions.argmax(axis=-1)),
        'report': classification_report(p.label_ids, p.predictions.argmax(axis=-1), output_dict=True)
    }                                  # evaluation metrics
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()

# Print the evaluation results
print(f"Accuracy: {eval_results['eval_accuracy']}")
print(f"Classification Report: {eval_results['eval_report']}")

# Save the model and tokenizer
model.save_pretrained('bert_sentiment_model')
tokenizer.save_pretrained('bert_sentiment_model')

print("Model and tokenizer saved.")


"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
"""