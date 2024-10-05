# authorship-attribution
![Description](cover.png)
This is a Python script that classifies sentences based on their authorship using two different approaches: **generative** and **discriminative**. It can either predict the author of given sentences or evaluate the classification accuracy on a development set.

## Installation

### Dependencies

The script requires the following Python packages:

- Python 3.6 or higher
- `argparse`
- `nltk`
- `transformers`
- `datasets`
- `evaluate`

### Installation Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/hardikkgupta/authorship-attribution.git
   cd authorship-attribution
   ```
2. **Create a Virtual Environment (Optional but Recommended)**
    ```bash
    python3 -m venv
    source venv/bin/activate
    ```
3. **Install Required Packages**
    ```bash
    pip install nltk transformers datasets evaluate
    ```
4. **Download NLTK Data**
    ```bash
    import nltk
    nltk.download('punkt')
    ```

## Usage

### Command-Line Arguments
```bash
python3 classifier.py [-h] -approach {generative,discriminative} [-test TEST] authorlist
```

The script requires the following Python packages:

- Python 3.6 or higher
- `authorlist` : Path to the file containing a list of author files.
- `-approach` : The approach to use for classification (`generative` or `discriminative`).
- `-test`: (Optional) Path to the file containing test sentences.

### Examples
Example 1: Generative Approach with Development Set
```bash
python authorship_classifier.py -approach generative authorlist.txt
```

Example 2: Generative Approach with Test File
```bash
python authorship_classifier.py -approach generative -test test_sentences.txt authorlist.txt
```
Example 3: Discriminative Approach with Development Set
```bash
python authorship_classifier.py -approach discriminative authorlist.txt
```
Example 4: Discriminative Approach with Test File
```bash
python authorship_classifier.py -approach discriminative -test test_sentences.txt authorlist.txt
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.