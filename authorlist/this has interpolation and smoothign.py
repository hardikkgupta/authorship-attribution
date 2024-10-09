import argparse
import sys
import os
import random
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from itertools import chain

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# For the generative model
from nltk.lm import MLE, Lidstone, WittenBellInterpolated, KneserNeyInterpolated, StupidBackoff
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends

# For the discriminative model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset, DatasetDict
import evaluate
import math

# Download necessary NLTK data files
nltk.download('punkt', quiet=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Authorship Classifier')
    parser.add_argument('authorlist', help='File containing list of author files')
    parser.add_argument('-approach', choices=['generative', 'discriminative'], required=True)
    parser.add_argument('-test', help='File containing test sentences')
    return parser.parse_args()

def create_language_model(n):
    # Use Witten-Bell interpolation as basic interpolation method
    model = WittenBellInterpolated(order=n)
    return model

def main():
    args = parse_args()
    authorlist_file = args.authorlist
    approach = args.approach
    test_file = args.test

    # Set random seed for reproducibility
    random.seed(42)

    # Read the authorlist file
    with open(authorlist_file, 'r') as f:
        author_files = [line.strip() for line in f if line.strip()]
    # Each line is an author file
    # We can use the filename (without extension) as author name
    authors = []
    for filename in author_files:
        author_name = filename.split('.')[0]
        authors.append((author_name, filename))

    if approach == 'generative':
        # Now, for each author, read the text data
        author_texts = {}
        author_dev_texts = {}
        for author_name, filename in authors:
            with open(filename, 'r', encoding='utf8') as f:
                text = f.read()
            # Tokenize sentences
            sentences = sent_tokenize(text)
            # Limit data size for faster processing
            sentences = sentences[:1000]  # Adjust as needed
            if test_file:
                # Use all data as training
                author_texts[author_name] = sentences
            else:
                # Extract 10% as dev set
                random.shuffle(sentences)
                split_point = int(0.9 * len(sentences))
                train_sentences = sentences[:split_point]
                dev_sentences = sentences[split_point:]
                author_texts[author_name] = train_sentences
                author_dev_texts[author_name] = dev_sentences

        # Now, train language models for each author
        # Using bigrams for simplicity and speed
        n = 2  # Using bigrams

        author_models = {}

        for author_name in author_texts:
            train_sentences = author_texts[author_name]
            # Tokenize words in sentences
            tokenized_sentences = [word_tokenize(sent.lower()) for sent in train_sentences]
            # Ensure there is data
            if not tokenized_sentences:
                print(f"No data for author {author_name}")
                continue
            # Prepare data for language model
            train_data, padded_sents = padded_everygram_pipeline(n, tokenized_sentences)
            # Build the vocabulary
            vocab = set(chain.from_iterable(tokenized_sentences))
            # Create language model with basic interpolation
            model = create_language_model(n)
            model.fit(train_data, vocab)
            author_models[author_name] = model

        if test_file:
            # Read test file
            with open(test_file, 'r', encoding='utf8') as f:
                test_sentences = [line.strip() for line in f if line.strip()]
            # For each test sentence, classify it
            for sent in test_sentences:
                tokenized_sent = word_tokenize(sent.lower())
                perplexities = {}
                for author_name, model in author_models.items():
                    # Map unknown words to <UNK>
                    tokenized_sent_mapped = list(model.vocab.lookup(tokenized_sent))
                    test_data = list(nltk.ngrams(pad_both_ends(tokenized_sent_mapped, n), n))
                    # Ensure test_data is not empty
                    if not test_data:
                        perplexities[author_name] = float('inf')
                        continue
                    try:
                        perplexity = model.perplexity(test_data)
                        if math.isinf(perplexity) or math.isnan(perplexity):
                            perplexity = 1e300  # Use a very large number
                    except (ZeroDivisionError, ValueError):
                        perplexity = 1e300  # Use a very large number
                    perplexities[author_name] = perplexity
                # Choose the author with lowest perplexity
                predicted_author = min(perplexities, key=perplexities.get)
                print(predicted_author)
        else:
            # Evaluate on dev set
            print("Splitting into training and development...")
            print("Training LMs... (this may take a while)")
            print("Results on dev set:")
            for author_name in author_dev_texts:
                dev_sentences = author_dev_texts[author_name]
                correct = 0
                total = 0
                for sent in dev_sentences:
                    tokenized_sent = word_tokenize(sent.lower())
                    perplexities = {}
                    for candidate_author, model in author_models.items():
                        # Map unknown words to <UNK>
                        tokenized_sent_mapped = list(model.vocab.lookup(tokenized_sent))
                        test_data = list(nltk.ngrams(pad_both_ends(tokenized_sent_mapped, n), n))
                        # Ensure test_data is not empty
                        if not test_data:
                            perplexities[candidate_author] = float('inf')
                            continue
                        try:
                            perplexity = model.perplexity(test_data)
                            if math.isinf(perplexity) or math.isnan(perplexity):
                                perplexity = 1e300  # Use a very large number
                        except (ZeroDivisionError, ValueError):
                            perplexity = 1e300  # Use a very large number
                        perplexities[candidate_author] = perplexity
                    # Choose the author with lowest perplexity
                    predicted_author = min(perplexities, key=perplexities.get)
                    if predicted_author == author_name:
                        correct += 1
                    total += 1
                accuracy = 100.0 * correct / total if total > 0 else 0
                print(f"{author_name} {accuracy:.1f}% correct")

    elif approach == 'discriminative':
        if test_file:
            # Use all data for training
            split_data = False
        else:
            split_data = True

        # Read and process data
        texts = []
        labels = []
        label2id = {}
        id2label = {}
        for idx, (author_name, filename) in enumerate(authors):
            label2id[author_name] = idx
            id2label[idx] = author_name
            with open(filename, 'r', encoding='utf8') as f:
                text = f.read()
            # Tokenize into sentences
            sentences = sent_tokenize(text)
            # Limit data size for faster processing
            sentences = sentences[:1000]  # Adjust as needed
            for sent in sentences:
                texts.append(sent)
                labels.append(idx)

        # Create Dataset
        dataset = Dataset.from_dict({'text': texts, 'label': labels})

        if split_data:
            # Split into train and dev
            dataset = dataset.shuffle(seed=42)
            train_testvalid = dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = train_testvalid['train']
            eval_dataset = train_testvalid['test']
        else:
            train_dataset = dataset
            # If test file is provided, load test data
            with open(test_file, 'r', encoding='utf8') as f:
                test_texts = [line.strip() for line in f if line.strip()]
            eval_dataset = Dataset.from_dict({'text': test_texts})

        # Load tokenizer and model
        model_name = 'distilbert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        num_labels = len(label2id)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )

        # Tokenize datasets
        def tokenize_function(examples):
            return tokenizer(examples['text'], truncation=True, padding=True, max_length=128)

        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Metrics
        metric = evaluate.load("accuracy")

        def compute_metrics(p):
            predictions, labels = p
            preds = predictions.argmax(-1)
            if labels is not None:
                accuracy = metric.compute(predictions=preds, references=labels)
                return accuracy
            else:
                return {}

        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=1,  # Reduced epochs to decrease training time
            per_device_train_batch_size=16,  # Increased batch size if memory allows
            per_device_eval_batch_size=16,
            evaluation_strategy='epoch' if split_data else 'no',
            logging_strategy='no',  # Disable logging to speed up training
            save_strategy='no',
            load_best_model_at_end=False,
            seed=42
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval if split_data else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if split_data else None
        )

        # Train the model
        trainer.train()

        if split_data:
            # Evaluate on dev set
            print("Results on dev set:")
            # Get predictions
            predictions = trainer.predict(tokenized_eval)
            preds = predictions.predictions.argmax(-1)
            labels = predictions.label_ids

            # Initialize counts
            correct_counts = {author_name: 0 for author_name in label2id.keys()}
            total_counts = {author_name: 0 for author_name in label2id.keys()}

            # Accumulate counts
            for pred_label_id, true_label_id in zip(preds, labels):
                true_author = id2label[true_label_id]
                total_counts[true_author] += 1
                if pred_label_id == true_label_id:
                    correct_counts[true_author] += 1

            # Compute and print per-author accuracy
            for author_name in correct_counts:
                correct = correct_counts[author_name]
                total = total_counts[author_name]
                accuracy = 100.0 * correct / total if total > 0 else 0
                print(f"{author_name} {accuracy:.1f}% correct")
        else:
            # Predict on test set
            predictions = trainer.predict(tokenized_eval)
            preds = predictions.predictions.argmax(-1)
            for pred_label_id in preds:
                pred_author = id2label[pred_label_id]
                print(pred_author)
    else:
        print("Invalid approach selected.")
        sys.exit(1)

if __name__ == '__main__':
    main()
