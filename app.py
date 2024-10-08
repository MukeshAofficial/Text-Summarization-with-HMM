from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
import requests
from bs4 import BeautifulSoup
import nltk
from gtts import gTTS
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai

# Configure Google Generative AI API key
genai.configure(api_key="AIzaSyAbmCYsZsjfCPf-uakFksDglYasW4EsehE")

app = Flask(__name__)

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
            summary_text = generate_notesummary(note_text)
            
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

@app.route('/web')
def web():
    return render_template('web-summary.html')

@app.route('/web_summary', methods=['POST'])
def web_summary():
    url = request.form['url']
    scraped_text = scrape_website(url)

    if scraped_text:
        websummary = generate_summary(scraped_text)
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

    text_summary = generate_summary(result)
    tts = gTTS(text=text_summary, lang='en')
    tts.save('ytoutput.mp3')

    return render_template('youtube-result.html', summary=text_summary)

@app.route('/play_audio')
def play_audio():
    return send_file('ytoutput.mp3', mimetype='audio/mpeg')

def generate_notesummary(note_text):
    model = genai.GenerativeModel('gemini-pro')
    rply = model.generate_content("summarize my notes: " + note_text)
    return rply.text

def generate_summary(note_text):
    model = genai.GenerativeModel('gemini-pro')
    rply = model.generate_content("summarize the given text without adding new words. If needed, add one or two lines on your own to keep it short: " + note_text)
    return rply.text

if __name__ == '__main__':
    app.run(debug=True)
