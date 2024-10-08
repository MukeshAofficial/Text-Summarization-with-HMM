from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
import requests
from bs4 import BeautifulSoup
import nltk
from gtts import gTTS
from youtube_transcript_api import YouTubeTranscriptApi
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from hmmlearn import hmm
from rouge_score import rouge_scorer
from collections import defaultdict
import random

app = Flask(__name__)

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')

def generate_hmm_summary(text, num_sentences=10, reference_summary=None):
    """
    Generate summary using a Hidden Markov Model-based text summarization.

    Args:
    text (str): Input text to summarize.
    num_sentences (int): Number of sentences in the summary.
    reference_summary (str): Optional reference summary for evaluation.

    Returns:
    str: Summarized text.
    """

    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Tokenize each sentence into words
    tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

    # Create a vocabulary of unique words
    vocab = list(set(word for sentence in tokenized_sentences for word in sentence))
    vocab_size = len(vocab)

    # Create a mapping from words to indices
    word_to_index = {word: i for i, word in enumerate(vocab)}

    # Create a feature matrix for HMM
    def create_feature_matrix(tokenized_sentences):
        feature_matrix = np.zeros((len(tokenized_sentences), vocab_size))
        for i, sentence in enumerate(tokenized_sentences):
            for word in sentence:
                if word in word_to_index:
                    feature_matrix[i, word_to_index[word]] += 1  # Count word occurrences
        return feature_matrix

    # Create the feature matrix
    X = create_feature_matrix(tokenized_sentences)

    # Train HMM
    n_states = 2  # Number of hidden states (e.g., summary and non-summary)
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)
    model.fit(X)

    # Predict the hidden states for each sentence
    hidden_states = model.predict(X)

    # Extract summary sentences based on predicted states
    summary_sentences = [sentences[i] for i in range(len(sentences)) if hidden_states[i] == hidden_states.argmax()]

    # Combine summary sentences and limit to the required number of sentences
    summary = ' '.join(summary_sentences[:num_sentences])

    # Evaluate the summary using ROUGE if reference_summary is provided
    if reference_summary:
        scores = evaluate_summary(reference_summary, summary)
        print("\nROUGE Evaluation Scores:")
        print(f"ROUGE-1: {scores['rouge1']}")
        print(f"ROUGE-2: {scores['rouge2']}")
        print(f"ROUGE-L: {scores['rougeL']}")

    return summary

# Function to evaluate summary using ROUGE
def evaluate_summary(reference_summary, generated_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    return scores

@app.route('/note', methods=['GET', 'POST'])
def note():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'Note_file' not in request.files:
            return render_template('error.html', message='No file part')

        file = request.files['Note_file']

        # Check if the file is empty
        if file.filename == '':
            return render_template('error.html', message='No selected file')

        # Check if the file is of allowed type
        if file and file.filename.endswith('.txt'):
            # Read the file content
            note_text = file.read().decode('utf-8')

            # Generate summary
            summary_text = generate_hmm_summary(note_text)
            
            # Render the result template with summary
            return render_template('note-result.html', summary_text=summary_text)

        else:
            return render_template('error.html', message='Invalid file type. Please upload a text file')

    return render_template('note.html')


def scrape_website(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the HTML content of the webpage
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract all text from the webpage
        text = soup.get_text()

        return text
    except requests.exceptions.RequestException as e:
        print("Error fetching the webpage:", e)
        return None


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/web')
def web():
    return render_template('web-summary.html')


@app.route('/web_summary', methods=['POST'])
def web_summary():
    url = request.form['url']
    scraped_text = scrape_website(url)

    if scraped_text:
        websummary = generate_hmm_summary(scraped_text)
        print(websummary)
        tts = gTTS(text=websummary, lang='en')
        output_file_audio = "static/outputsummary.mp3"
        tts.save(output_file_audio)
        return render_template('web-result.html', websummary=websummary, audio_file=output_file_audio)
    else:
        return "No text scraped from the website."


@app.route('/youtube-summary')
def yt():
    return render_template('youtube.html')


@app.route('/summarize', methods=['POST'])
def summarize():
    video_link = request.form['videoLink']
    video_id = video_link.split('=')[1]

    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    result = ""
    for i in transcript:
        result += ' ' + i['text']

    # Reference summary (you may want to replace this with a real reference)
    reference_summary = "AI is transforming education with innovative solutions..."

    text_summary = generate_hmm_summary(result, reference_summary=reference_summary)
    tts = gTTS(text=text_summary, lang='en')
    tts.save('ytoutput.mp3')

    return render_template('youtube-result.html', summary=text_summary)


@app.route('/play_audio')
def play_audio():
    return send_file('ytoutput.mp3', mimetype='audio/mpeg')


if __name__ == '__main__':
    app.run(debug=True)
