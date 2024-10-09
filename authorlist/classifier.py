import argparse
import sys
import os
import random
import math
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from itertools import chain
from collections import defaultdict

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# For the generative model
from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Vocabulary

# For the discriminative model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
import evaluate

# Download necessary NLTK data files
nltk.download('punkt', quiet=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Authorship Classifier')
    parser.add_argument('authorlist', help='File containing list of author files')
    parser.add_argument('-approach', choices=['generative', 'discriminative'], required=True, help='Choose the classification approach')
    parser.add_argument('-test', help='File containing test sentences')
    return parser.parse_args()

# Define interpolation weights
INTERPOLATION_WEIGHTS = {
    1: 0.1,  # Unigram weight
    2: 0.1,  # Bigram weight
    3: 0.8   # Trigram weight
}

n_orders = [1, 2, 3]  # Unigram, Bigram, Trigram

def calculate_interpolated_perplexity(models, tokenized_sent):
    """
    Calculate perplexity using interpolated probabilities.
    
    :param models: Dictionary of n-gram models {n: model}
    :param tokenized_sent: List of words (tokenized)
    :return: Perplexity value
    """
    N = len(tokenized_sent)
    log_prob = 0.0
    for i in range(1, N + 1):
        word = tokenized_sent[i-1]
        prob = 0.0
        for n in n_orders:
            model = models.get(n)
            if not model:
                continue
            if n == 1:
                context = []
            else:
                if i - n + 1 < 0:
                    context = tokenized_sent[0:i-1]
                else:
                    context = tokenized_sent[i-n+1:i-1]
            prob += INTERPOLATION_WEIGHTS.get(n, 0) * model.score(word, context)
        if prob > 0:
            log_prob += math.log(prob, 2)
        else:
            # Assign a large negative log probability for unseen events
            log_prob += float('-inf')
    
    if log_prob == float('-inf'):
        return float('inf')
    
    perplexity = math.pow(2, -log_prob / N)
    return perplexity

def generate_sample(models, prompt, vocab, max_length=15):
    """
    Generate a sample text based on the given prompt and author's language models.
    
    :param models: Dictionary of n-gram models {n: model}
    :param prompt: String prompt to seed the generation
    :param vocab: Set of vocabulary words
    :param max_length: Number of words to generate
    :return: Generated sentence as a string
    """
    tokenized_prompt = word_tokenize(prompt.lower())
    # Map unknown words to <UNK>
    tokenized_prompt_mapped = [word if word in vocab else '<UNK>' for word in tokenized_prompt]
    generated_words = list(tokenized_prompt_mapped)
    
    for _ in range(max_length):
        probs = defaultdict(float)
        for n in n_orders:
            model = models.get(n)
            if not model:
                continue
            if n == 1:
                context = []
            else:
                context = generated_words[-(n-1):] if len(generated_words) >= (n-1) else generated_words
            for word in model.vocab:
                probs[word] += INTERPOLATION_WEIGHTS.get(n, 0) * model.score(word, context)
        # Normalize probabilities
        total_prob = sum(probs.values())
        if total_prob == 0:
            next_word = '<UNK>'
        else:
            words, probabilities = zip(*probs.items())
            probabilities = [p / total_prob for p in probabilities]
            next_word = random.choices(words, weights=probabilities, k=1)[0]
        generated_words.append(next_word)
    
    # Combine words into a sentence
    # Optionally, capitalize the first word and add a period at the end
    generated_sentence = ' '.join(['<UNK>' if word == '<UNK>' else word for word in generated_words[len(tokenized_prompt_mapped):]])
    if generated_sentence:
        generated_sentence = generated_sentence[0].upper() + generated_sentence[1:]
        if generated_sentence[-1] not in ['.', '!', '?']:
            generated_sentence += '.'
    return generated_sentence

def main():
    args = parse_args()
    authorlist_file = args.authorlist
    approach = args.approach
    test_file = args.test

    # Set random seed for reproducibility
    random.seed(42)

    # Read the authorlist file
    if not os.path.isfile(authorlist_file):
        print(f"Error: Author list file '{authorlist_file}' not found.")
        sys.exit(1)

    with open(authorlist_file, 'r') as f:
        author_files = [line.strip() for line in f if line.strip()]
    # Each line is an author file
    # We can use the filename (without extension) as author name
    authors = []
    for filename in author_files:
        if not os.path.isfile(filename):
            print(f"Error: Author file '{filename}' not found.")
            sys.exit(1)
        author_name = os.path.splitext(os.path.basename(filename))[0]
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
                if len(sentences) < 10:
                    print(f"Warning: Not enough sentences for author '{author_name}' to split into train/dev.")
                    author_texts[author_name] = sentences
                    continue
                random.shuffle(sentences)
                split_point = max(int(0.9 * len(sentences)), 1)
                train_sentences = sentences[:split_point]
                dev_sentences = sentences[split_point:]
                author_texts[author_name] = train_sentences
                author_dev_texts[author_name] = dev_sentences

        # Now, train language models for each author
        author_models = {}  # Structure: {author: {1: model_unigram, 2: model_bigram, 3: model_trigram}}

        for author_name in author_texts:
            train_sentences = author_texts[author_name]
            # Tokenize words in sentences
            tokenized_sentences = [word_tokenize(sent.lower()) for sent in train_sentences]
            # Ensure there is data
            if not tokenized_sentences:
                print(f"No data for author {author_name}")
                continue
            # Build a combined vocabulary for all n-gram models
            vocab = set(chain.from_iterable(tokenized_sentences))
            vocab.add('<UNK>')  # Add unknown token

            author_models[author_name] = {}

            for n in n_orders:
                # Prepare n-gram data
                train_data, _ = padded_everygram_pipeline(n, tokenized_sentences)
                
                # Initialize Laplace model
                model = Laplace(n)
                model.fit(train_data, Vocabulary(vocab))
                
                # Verify total counts
                total_counts = model.counts.N()
                print(f"Total counts for '{author_name}' (n={n}): {total_counts}")
                if total_counts == 0:
                    print(f"Warning: Model for author '{author_name}' with n={n} has zero total counts.")
                
                author_models[author_name][n] = model

        if test_file:
            if not os.path.isfile(test_file):
                print(f"Error: Test file '{test_file}' not found.")
                sys.exit(1)
            # Read test file
            with open(test_file, 'r', encoding='utf8') as f:
                test_sentences = [line.strip() for line in f if line.strip()]
            if not test_sentences:
                print("Error: Test file is empty.")
                sys.exit(1)
            # Define five prompts
            prompts = [
                "In the beginning,",
                "The sun set over the horizon,",
                "He thought to himself,",
                "It was a dark and stormy night,",
                "As the story unfolds,"
            ]
            # Check if there are at least five test sentences
            if len(test_sentences) < 5:
                print("Warning: Less than five test sentences provided. Some prompts may be reused.")
            
            # Select five test sentences as prompts
            selected_prompts = test_sentences[:5] if len(test_sentences) >=5 else test_sentences + [test_sentences[-1]]*(5 - len(test_sentences))
            
            # Initialize a list to store results
            results = []
            
            # Iterate over each prompt
            for idx, prompt in enumerate(selected_prompts, 1):
                print(f"\n--- Prompt {idx}: \"{prompt}\" ---")
                # For each author, generate one sample
                for author_name, models in author_models.items():
                    sample_vocab = models[1].vocab  # Using unigram vocab
                    generated_sentence = generate_sample(models, prompt, sample_vocab, max_length=15)
                    
                    # Tokenize the generated sentence for perplexity calculation
                    tokenized_generated = word_tokenize(generated_sentence.lower())
                    # Map unknown words to <UNK>
                    tokenized_generated_mapped = [word if word in sample_vocab else '<UNK>' for word in tokenized_generated]
                    
                    # Calculate perplexity scores across all authors
                    perplexity_scores = {}
                    for other_author, other_models in author_models.items():
                        perplexity = calculate_interpolated_perplexity(other_models, tokenized_generated_mapped)
                        perplexity_scores[other_author] = perplexity
                        
                    # Store the result
                    results.append({
                        'Prompt': prompt,
                        'Author': author_name,
                        'Generated Sample': generated_sentence,
                        'Perplexity Scores': perplexity_scores
                    })
                    
                    # Print the result
                    print(f"\nAuthor: {author_name}")
                    print(f"Generated Sample: {generated_sentence}")
                    print("Perplexity Scores:")
                    for other_author, perp in perplexity_scores.items():
                        perp_display = f"{perp:.2f}" if perp != float('inf') else "Infinity"
                        print(f"  {other_author}: {perp_display}")
            
            # Optionally, you can save the results to a file or further process them
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
                    # Map unknown words to <UNK>
                    sample_author = next(iter(author_models))
                    sample_vocab = author_models[sample_author][1].vocab
                    tokenized_sent_mapped = [word if word in sample_vocab else '<UNK>' for word in tokenized_sent]
                    perplexities = {}
                    for candidate_author, models in author_models.items():
                        perplexity = calculate_interpolated_perplexity(models, tokenized_sent_mapped)
                        perplexities[candidate_author] = perplexity
                    # Choose the author with lowest perplexity
                    predicted_author = min(perplexities, key=perplexities.get)
                    if predicted_author == author_name:
                        correct += 1
                    total += 1
                accuracy = 100.0 * correct / total if total > 0 else 0
                print(f"{author_name}: {accuracy:.1f}% correct")
    
            # Generate samples from each model
            # Define five prompts
            prompts = [
                "In the beginning,",
                "The sun set over the horizon,",
                "He thought to himself,",
                "It was a dark and stormy night,",
                "As the story unfolds,"
            ]
            # Select five dev sentences as prompts or reuse if insufficient
            dev_sentences_flat = list(chain.from_iterable(author_dev_texts.values()))
            selected_prompts = dev_sentences_flat[:5] if len(dev_sentences_flat) >=5 else dev_sentences_flat + [dev_sentences_flat[-1]]*(5 - len(dev_sentences_flat))
            
            # Initialize a list to store results
            results = []
            
            # Iterate over each prompt
            for idx, prompt in enumerate(selected_prompts, 1):
                print(f"\n--- Prompt {idx}: \"{prompt}\" ---")
                # For each author, generate one sample
                for author_name, models in author_models.items():
                    sample_vocab = models[1].vocab  # Using unigram vocab
                    generated_sentence = generate_sample(models, prompt, sample_vocab, max_length=15)
                    
                    # Tokenize the generated sentence for perplexity calculation
                    tokenized_generated = word_tokenize(generated_sentence.lower())
                    # Map unknown words to <UNK>
                    tokenized_generated_mapped = [word if word in sample_vocab else '<UNK>' for word in tokenized_generated]
                    
                    # Calculate perplexity scores across all authors
                    perplexity_scores = {}
                    for other_author, other_models in author_models.items():
                        perplexity = calculate_interpolated_perplexity(other_models, tokenized_generated_mapped)
                        perplexity_scores[other_author] = perplexity
                        
                    # Store the result
                    results.append({
                        'Prompt': prompt,
                        'Author': author_name,
                        'Generated Sample': generated_sentence,
                        'Perplexity Scores': perplexity_scores
                    })
                    
                    # Print the result
                    print(f"\nAuthor: {author_name}")
                    print(f"Generated Sample: {generated_sentence}")
                    print("Perplexity Scores:")
                    for other_author, perp in perplexity_scores.items():
                        perp_display = f"{perp:.2f}" if perp != float('inf') else "Infinity"
                        print(f"  {other_author}: {perp_display}")
    
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

        if not texts:
            print("Error: No data found for discriminative approach.")
            sys.exit(1)

        # Create Dataset
        dataset = Dataset.from_dict({'text': texts, 'label': labels})

        if split_data:
            # Split into train and dev
            dataset = dataset.shuffle(seed=42)
            train_test = dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = train_test['train']
            eval_dataset = train_test['test']
        else:
            train_dataset = dataset
            # If test file is provided, load test data
            if test_file:
                with open(test_file, 'r', encoding='utf8') as f:
                    test_texts = [line.strip() for line in f if line.strip()]
                if not test_texts:
                    print("Error: Test file is empty.")
                    sys.exit(1)
                eval_dataset = Dataset.from_dict({'text': test_texts})
            else:
                eval_dataset = None

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
            return tokenizer(examples['text'], truncation=True, padding=False)

        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        if eval_dataset:
            tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
        else:
            tokenized_eval = None

        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Metrics
        metric = evaluate.load("accuracy")


        def compute_metrics(p):
            predictions, labels = p
            preds = predictions.argmax(-1)
            if labels is None:
                return {}
            return metric.compute(predictions=preds, references=labels)


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

        if split_data and tokenized_eval:
            # Evaluate on dev set
            print("Results on dev set:")
            results = trainer.evaluate()
            accuracy = results.get('eval_accuracy', 0) * 100
            print(f"Accuracy: {accuracy:.1f}%")
        else:
            if test_file and tokenized_eval:
                # Predict on test set
                predictions = trainer.predict(tokenized_eval)
                preds = predictions.predictions.argmax(-1)
                for pred_label_id in preds:
                    pred_author = id2label[pred_label_id]
                    print(pred_author)
            else:
                print("No evaluation performed for discriminative approach.")
    else:
        print("Invalid approach selected.")
        sys.exit(1)

if __name__ == '__main__':
    main()
