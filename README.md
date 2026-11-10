# MeetMind – AI-Powered Meeting Assistant

MeetMind is an AI-powered meeting assistant that converts recorded speech into text and uses Natural Language Processing (NLP) to extract meaningful information from meetings.

## Features

- **Audio Recording:** Record meeting audio using a microphone.
- **Audio Upload:** Upload existing audio recordings.
- **Speech-to-Text:** Convert speech into text using Faster-Whisper.
- **Meeting Summary:** Generate a concise extractive summary.
- **Key Point Extraction:** Identify important sentences.
- **Topic Identification:** Group discussion into different topics.
- **Action Item Extraction:** Identify tasks and responsible persons.
- **Deadline Detection:** Detect deadlines mentioned during meetings.
- **PDF Report:** Generate downloadable meeting reports.

## Technologies Used

- Python
- Streamlit
- Faster-Whisper
- Scikit-learn
- Regular Expressions
- ReportLab

## Project Structure

```text
MeetMind/
│
├── app.py
├── nlp_processor.py
├── transcribe.py
├── test_audio.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/srishti-gururaj/MeetMind-AI-Meeting-Assistant.git
```

### 2. Open the project folder

```bash
cd MeetMind-AI-Meeting-Assistant
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Run the application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## How It Works

1. The user records or uploads meeting audio.
2. Faster-Whisper converts the audio into text.
3. The NLP processor analyzes the transcript.
4. Important points and summaries are generated.
5. Discussion topics are identified.
6. Action items and deadlines are extracted.
7. A downloadable PDF meeting report is generated.

## Future Enhancements

- Improved abstractive summarization.
- More accurate topic classification.
- Enhanced action item and deadline extraction.
- Speaker identification.
- Multilingual speech recognition.

## Project Information

**Project Name:** MeetMind – AI-Powered Meeting Assistant

**Domain:** Speech Processing and Natural Language Processing

**Repository:** [MeetMind on GitHub](https://github.com/srishti-gururaj/MeetMind-AI-Meeting-Assistant)
