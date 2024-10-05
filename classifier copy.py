import argparse
import os
import random
from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from collections import defaultdict
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoTokenizer
)
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor


class AuthorClassifier:
    def __init__(self, authorlist, approach, testfile=None):
        self.authorlist = authorlist
        self.approach = approach
        self.testfile = testfile
        self.models = {}
        self.dev_split_ratio = 0.1

    def load_data(self, filename):
        """Loads text data from the file."""
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read().splitlines()

    def preprocess_data(self, text, n=2):
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
            if self.testfile:
                # Use all data for training
                train_set = data
            else:
                dev_size = int(len(data) * self.dev_split_ratio)
                dev_set = data[:dev_size]
                train_set = data[dev_size:]
                self.models[author_file] = {'dev_set': dev_set}

            train_data, vocab = self.preprocess_data(train_set)
            model = Laplace(2)  # Using bigram model with Laplace smoothing
            model.fit(train_data, vocab)
            self.models[author_file]['model'] = model

    def evaluate_models(self):
        """Evaluates each model on the entire development set in batches."""
        print("Results on dev set:")
        batch_size = 10
        for author_file, data in self.models.items():
            dev_set = data.get('dev_set', [])
            correct = 0
            total = len(dev_set)
            if total == 0:
                print(f"Warning: Development set for {author_file} is empty. Skipping evaluation.")
                continue

            start_time = time.time()
            for i in range(0, total, batch_size):
                batch = dev_set[i:i + batch_size]
                predictions = [self.classify(sentence) for sentence in batch]
                correct += sum(1 for prediction in predictions if prediction == author_file)

            accuracy = (correct / total) * 100
            elapsed_time = time.time() - start_time
            print(f"{author_file} {accuracy:.1f}% correct (evaluation time: {elapsed_time:.2f} seconds)")

    def classify(self, text):
        """Classifies the input text to the most probable author."""
        try:
            tokenized_text = list(ngrams(word_tokenize(text), 2))
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
        """Trains a discriminative model using Huggingface Transformers and tests it if testfile is provided."""
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

        # If testfile is provided, use all data for training
        if self.testfile:
            train_texts, train_labels = texts, labels
            val_texts, val_labels = [], []
        else:
            # Split data into training and validation sets
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=self.dev_split_ratio, random_state=42)

        # Load tokenizer and tokenize data
        model_name = 'distilbert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        if val_texts:
            val_encodings = tokenizer(val_texts, truncation=True, padding=True)

        # Create datasets
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': train_labels
        })
        if val_texts:
            val_dataset = Dataset.from_dict({
                'input_ids': val_encodings['input_ids'],
                'attention_mask': val_encodings['attention_mask'],
                'labels': val_labels
            })
        else:
            val_dataset = None

        # Define model directory
        model_dir = './trained_model'

        # Check if the model is already saved
        if os.path.exists(model_dir):
            print("Loading saved model...")
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
        else:
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
                eval_strategy='no' if self.testfile else 'epoch',
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                num_train_epochs=1,  # Reduced to 1 epoch for faster training
                weight_decay=0.01,
                fp16=torch.cuda.is_available(),  # Use mixed precision if GPU is available
                report_to='none',  # Disable reporting
                logging_dir='./logs',
                logging_steps=100  # Log every 100 steps
            )

            # Move model to device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)

            # Define trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset
            )

            model.save_pretrained('./trained_model')
            tokenizer.save_pretrained('./trained_model')

    # Before training, check if model exists
            if os.path.exists('./trained_model'):
                model = AutoModelForSequenceClassification.from_pretrained('./trained_model')
                tokenizer = AutoTokenizer.from_pretrained('./trained_model')
            else:
                trainer.train()
            

        # If testfile is provided, classify the test sentences
        if self.testfile:
            self.test_discriminative_model(model, tokenizer)
        elif val_dataset:
            # Evaluate on validation set
            print("Evaluating on the validation set...")
            if not os.path.exists('./results'):
                os.makedirs('./results')
            training_args = TrainingArguments(
                output_dir='./results',
                eval_strategy='epoch',
                per_device_eval_batch_size=8,
                logging_dir='./logs',
                logging_steps=100,
                report_to='none'
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                eval_dataset=val_dataset
            )
            eval_results = trainer.evaluate()
            print(f"Validation Results: {eval_results}")

    def test_discriminative_model(self, model, tokenizer):
        """Classifies each line in the given test file using the trained discriminative model."""
        test_data = self.load_data(self.testfile)

        # Tokenize test data
        test_encodings = tokenizer(test_data, truncation=True, padding=True)
        test_dataset = Dataset.from_dict({
            'input_ids': test_encodings['input_ids'],
            'attention_mask': test_encodings['attention_mask']
        })

        # Set the format to 'torch' to ensure DataLoader returns tensors
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

        # Create DataLoader
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8)

        # Move model to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        # Perform prediction
        predictions = []
        id2label = {v: k for k, v in model.config.label2id.items()}

        print("Classifying test sentences...")
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                predictions.extend(preds.cpu().numpy())

        # Map predictions back to author names
        predicted_authors = [id2label[pred] for pred in predictions]

        # Print the predictions
        print("Predicted Authors:")
        for author in predicted_authors:
            print(author)

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
    with open(args.authorlist, 'r', encoding='utf-8') as f:
        author_files = [line.strip() for line in f.readlines()]

    # Initialize and run the classifier
    classifier = AuthorClassifier(author_files, args.approach, args.test)
    classifier.run()
