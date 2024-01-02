import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# Load your DataFrame
final_lyrics_df = pd.read_csv('final_lyrics.csv')  # Adjust the path as needed

# Remove the rows with genre_top as 'Experimental'
final_lyrics_df = final_lyrics_df[final_lyrics_df['genre_top'] != 'Experimental']

#convert genre_top to numeric values
final_lyrics_df['genre_top'] = pd.Categorical(final_lyrics_df['genre_top'])

# Splitting the dataset into training and testing sets
train_df, test_df = train_test_split(final_lyrics_df, test_size=0.2, random_state=42)

def preprocess_data(df, tokenizer, max_length=512):
    # Tokenize the lyrics
    return tokenizer(df['trackLyrics'].tolist(), padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=final_lyrics_df['genre_top'].nunique())

class LyricsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Preprocess the data
train_encodings = preprocess_data(train_df, tokenizer)
test_encodings = preprocess_data(test_df, tokenizer)

# Create datasets
train_dataset = LyricsDataset(train_encodings, train_df['genre_top'].tolist())
test_dataset = LyricsDataset(test_encodings, test_df['genre_top'].tolist())

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# Make predictions
predictions = trainer.predict(test_dataset)

# Evaluate
predicted_labels = predictions.predictions.argmax(-1)
report = classification_report(test_df['genre_top'], predicted_labels)
print(report)


