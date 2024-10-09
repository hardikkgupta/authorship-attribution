import argparse
import sys
import os
import random
import math
from collections import defaultdict, Counter
from itertools import chain
from nltk.tokenize import word_tokenize, sent_tokenize

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")


# Download necessary NLTK data files
import nltk
nltk.download('punkt', quiet=True)

class NGramModel:
    def __init__(self, n):
        self.n = n
        self.ngram_counts = defaultdict(Counter)  # Maps (n-1)-gram to Counter of next words
        self.context_counts = Counter()           # Counts of (n-1)-grams
        self.vocabulary = set()
    
    def train(self, sentences):
        for sentence in sentences:
            tokens = ['<s>'] * (self.n - 1) + word_tokenize(sentence.lower()) + ['</s>']
            self.vocabulary.update(tokens)
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i + self.n -1])
                word = tokens[i + self.n -1]
                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1
        self.vocab_size = len(self.vocabulary)
    
    def probability(self, context, word):
        # Apply Laplace smoothing
        context = tuple(context)
        word_count = self.ngram_counts[context][word] + 1  # Add-one smoothing
        context_count = self.context_counts[context] + self.vocab_size  # Add-one for each possible word
        return word_count / context_count
    
    def perplexity(self, sentences):
        log_prob = 0
        word_count = 0
        for sentence in sentences:
            tokens = ['<s>'] * (self.n -1) + word_tokenize(sentence.lower()) + ['</s>']
            word_count += len(tokens) - (self.n -1)
            for i in range(len(tokens) - self.n +1):
                context = tuple(tokens[i:i + self.n -1])
                word = tokens[i + self.n -1]
                prob = self.probability(context, word)
                log_prob += math.log(prob)
        perplexity = math.exp(-log_prob / word_count) if word_count > 0 else float('inf')
        return perplexity

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

        # Now, train n-gram models for each author
        n = 2  # Using bigrams for simplicity and speed

        author_models = {}

        for author_name in author_texts:
            train_sentences = author_texts[author_name]
            # Initialize and train the n-gram model
            model = NGramModel(n)
            model.train(train_sentences)
            author_models[author_name] = model

        if test_file:
            # Read test file
            with open(test_file, 'r', encoding='utf8') as f:
                test_sentences = [line.strip() for line in f if line.strip()]
            # For each test sentence, classify it
            for sent in test_sentences:
                perplexities = {}
                for author_name, model in author_models.items():
                    pp = model.perplexity([sent])
                    perplexities[author_name] = pp
                # Choose the author with lowest perplexity
                predicted_author = min(perplexities, key=perplexities.get)
                print(predicted_author)
        else:
            # Evaluate on dev set
            print("Splitting into training and development...")
            print("Training n-gram LMs with Laplace smoothing... (this may take a while)")
            print("Results on dev set:")
            for author_name in author_dev_texts:
                dev_sentences = author_dev_texts[author_name]
                correct = 0
                total = 0
                for sent in dev_sentences:
                    perplexities = {}
                    for candidate_author, model in author_models.items():
                        pp = model.perplexity([sent])
                        perplexities[candidate_author] = pp
                    # Choose the author with lowest perplexity
                    predicted_author = min(perplexities, key=perplexities.get)
                    if predicted_author == author_name:
                        correct += 1
                    total += 1
                accuracy = 100.0 * correct / total if total > 0 else 0
                print(f"{author_name} {accuracy:.4f}% correct")

    elif approach == 'discriminative':
        print("Wrong Choice")
        ...
    else:
        print("Invalid approach selected.")
        sys.exit(1)

if __name__ == '__main__':
    main()
