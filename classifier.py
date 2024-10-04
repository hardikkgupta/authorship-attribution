import argparse
import os
import random
from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from collections import defaultdict
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
from transformers import AutoTokenizer
import time
import numpy as np

class AuthorClassifier:
    def __init__(self, authorlist, approach, testfile=None):
        self.authorlist = authorlist
        self.approach = approach
        self.testfile = testfile
        self.models = {}
        self.dev_split_ratio = 0.1

    def load_data(self, filename):
        """Loads text data from the file."""
        with open(filename, 'r') as file:
            return file.read().splitlines()

    def preprocess_data(self, text, n=3):
        """Tokenizes and pads the text to prepare it for n-gram model training."""
        tokenized_text = [list(word_tokenize(sent)) for sent in text]
        train_data, vocab = padded_everygram_pipeline(n, tokenized_text)
        return train_data, vocab

    def train_models(self):
        """Trains language models for each author in the authorlist."""
        for author_file in self.authorlist:
            print(f"Training model for author: {author_file}")
            data = self.load_data(author_file)
            random.shuffle(data)
            unique_data = list(set(data))  # Remove duplicates to prevent data leakage
            random.shuffle(unique_data)
            dev_size = int(len(unique_data) * self.dev_split_ratio)
            dev_set = unique_data[:dev_size]
            train_set = unique_data[dev_size:]

            train_data, vocab = self.preprocess_data(train_set)
            model = Laplace(3)  # Using trigram model with Laplace smoothing for faster processing
            model.fit(train_data, vocab)
            self.models[author_file] = {'model': model, 'dev_set': dev_set}

    def evaluate_models(self):
        """Evaluates each model on the entire development set in batches."""
        print("Results on dev set:")
        batch_size = 10
        for author_file, data in self.models.items():
            correct = 0
            total = len(data['dev_set'])
            if total == 0:
                print(f"Warning: Development set for {author_file} is empty. Skipping evaluation.")
                continue

            start_time = time.time()
            for i in range(0, total, batch_size):
                batch = data['dev_set'][i:i + batch_size]
                print(f"Evaluating batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size} for author {author_file}...")
                predictions = [self.classify(sentence) for sentence in batch]
                correct += sum(1 for prediction in predictions if prediction == author_file)

            accuracy = (correct / total) * 100
            elapsed_time = time.time() - start_time
            print(f"{author_file} {accuracy:.1f}% correct (evaluation time: {elapsed_time:.2f} seconds)")

    def classify(self, text):
        """Classifies the input text to the most probable author."""
        try:
            tokenized_text = list(ngrams(word_tokenize(text), 3))
        except Exception as e:
            print(f"Tokenization error: {e}")
            return None
        if not tokenized_text:
            return None
        min_perplexity = float('inf')
        best_author = None
        for author_file, data in self.models.items():
            model = data['model']
            perplexity = model.perplexity(tokenized_text)
            if perplexity < min_perplexity:
                min_perplexity = perplexity
                best_author = author_file
        return best_author

    def test_file(self, testfile):
        """Classifies each line in the given test file."""
        test_data = self.load_data(testfile)
        for line in test_data:
            predicted_author = self.classify(line)
            if predicted_author:
                print(predicted_author)
            else:
                print("Could not classify the given line.")

    def train_discriminative_model(self):
        """Trains a discriminative model using Huggingface Transformers."""
        # Prepare data
        texts = []
        labels = []
        label2id = {}
        id2label = {}

        for idx, author_file in enumerate(self.authorlist):
            label2id[author_file] = idx
            id2label[idx] = author_file
            data = self.load_data(author_file)
            texts.extend(data)
            labels.extend([idx] * len(data))

        # Split data into training and validation sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=self.dev_split_ratio, random_state=42, stratify=labels)

        # Load tokenizer and tokenize data
        model_name = 'distilbert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True)

        # Create datasets
        train_dataset = Dataset.from_dict({'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask'], 'labels': train_labels})
        val_dataset = Dataset.from_dict({'input_ids': val_encodings['input_ids'], 'attention_mask': val_encodings['attention_mask'], 'labels': val_labels})

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.authorlist),
            id2label=id2label,
            label2id=label2id
        )

        # Define training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            evaluation_strategy='epoch',
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU is available
            report_to='none',  # Disable reporting to avoid unnecessary logging in Colab
            logging_dir='./logs',  # Directory for logging
            logging_steps=10
        )

        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Define trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        # Train the model
        trainer.train()

    def run(self):
        if self.approach == 'generative':
            self.train_models()
            if self.testfile:
                self.test_file(self.testfile)
            else:
                self.evaluate_models()
        elif self.approach == 'discriminative':
            self.train_discriminative_model()
        else:
            raise ValueError("Unknown approach. Use 'generative' or 'discriminative'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Authorship Classifier")
    parser.add_argument('authorlist', type=str, help="File containing a list of training set file names")
    parser.add_argument('-approach', type=str, choices=['generative', 'discriminative'], required=True, help="Approach to use for classification")
    parser.add_argument('-test', type=str, help="Test file with sentences to classify")
    args = parser.parse_args()

    # Load author list
    with open(args.authorlist, 'r') as f:
        author_files = [line.strip() for line in f.readlines()]

    # Initialize and run the classifier
    classifier = AuthorClassifier(author_files, args.approach, args.test)
    classifier.run()
