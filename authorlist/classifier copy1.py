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
from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends

# For the discriminative model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
# from datasets import load_metric, Dataset, DatasetDict
from datasets import Dataset, DatasetDict
import evaluate


# Download necessary NLTK data files
nltk.download('punkt', quiet=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Authorship Classifier')
    parser.add_argument('authorlist', help='File containing list of author files')
    parser.add_argument('-approach', choices=['generative', 'discriminative'], required=True)
    parser.add_argument('-test', help='File containing test sentences')
    return parser.parse_args()

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
        # You can experiment with different 'n' and smoothing techniques
        n = 3  # Using trigrams

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
            # Create language model with Laplace smoothing
            model = Laplace(n)
            model.fit(train_data, vocab)
            # Verify total counts
            total_counts = model.counts.N()
            print(f"Total counts for {author_name}: {total_counts}")
            if total_counts == 0:
                print(f"Warning: Model for {author_name} has zero total counts.")
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
                    perplexity = model.perplexity(test_data)
                    perplexities[author_name] = perplexity
                # Choose the author with lowest perplexity
                predicted_author = min(perplexities, key=perplexities.get)
                print(predicted_author)
        else:
            # Evaluate on dev set
            print("splitting into training and development...")
            print("training LMs... (this may take a while)")
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
                            print(f"Warning: Test data is empty for sentence: '{sent}'")
                            perplexities[candidate_author] = float('inf')
                            continue
                        try:
                            perplexity = model.perplexity(test_data)
                        except ZeroDivisionError:
                            print(f"Warning: Division by zero for author {candidate_author} on sentence '{sent}'")
                            perplexity = float('inf')
                        perplexities[candidate_author] = perplexity
                    # Choose the author with lowest perplexity
                    predicted_author = min(perplexities, key=perplexities.get)
                    if predicted_author == author_name:
                        correct += 1
                    total += 1
                accuracy = 100.0 * correct / total if total > 0 else 0
                print(f"{author_name} {accuracy:.1f}% correct")

            # Generate samples from each model
            prompt = 'Once upon a time'
            tokenized_prompt = word_tokenize(prompt.lower())

            for author_name, model in author_models.items():
                print(f"\nSamples from {author_name}'s model:")
                for i in range(5):
                    generated_words = model.generate(15, text_seed=tokenized_prompt)
                    generated_sentence = ' '.join(generated_words)
                    print(f"{i+1}: {generated_sentence}")
        # Existing generative code...
        # [Assuming the code from Step 1 is here]
        pass

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
            # test_labels = [-1]*len(test_texts)  # Dummy labels
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
            return tokenizer(examples['text'], truncation=True)

        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Metrics
        # metric = load_metric("accuracy")
        metric = evaluate.load("accuracy")


        def compute_metrics(p):
            predictions, labels = p
            preds = predictions.argmax(-1)
            valid_indices = labels != -1  # Only compute accuracy on valid labels
            if valid_indices.any():
                accuracy = metric.compute(predictions=preds[valid_indices], references=labels[valid_indices])
                return accuracy
            else:
                return {}

        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy='epoch' if split_data else 'no',
            logging_strategy='epoch',
            save_strategy='no',
            load_best_model_at_end=False,
            metric_for_best_model='accuracy',
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
