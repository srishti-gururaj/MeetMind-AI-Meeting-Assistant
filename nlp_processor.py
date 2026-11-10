import re
from collections import Counter, defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer


STOP_WORDS = {
    "the", "is", "are", "was", "were", "a", "an", "and",
    "or", "but", "to", "of", "in", "on", "for", "with",
    "this", "that", "it", "we", "they", "he", "she",
    "i", "you", "our", "their", "my", "your", "be",
    "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "should", "must", "can",
    "could", "by", "before", "after", "from", "as",
    "at", "about", "into", "everyone", "need", "hi",
    "hello", "today", "meeting", "discuss", "discussing",
    "going", "like", "just", "also", "first", "finally",
    "next", "let", "begin", "good", "morning"
}


DEADLINE_PATTERN = (
    r'\b(?:by|before|on)\s+(?:next\s+)?'
    r'(?:monday|tuesday|wednesday|thursday|friday|'
    r'saturday|sunday|week|month|'
    r'\d{1,2}\s+'
    r'(?:january|february|march|april|may|june|'
    r'july|august|september|october|november|december))\b'
    r'|\b(?:today|tomorrow|tonight|next week|next month|'
    r'in \d+ days?)\b'
)


# 1. READ TRANSCRIPT

def read_transcript():

    try:
        with open("transcript.txt", "r", encoding="utf-8") as file:
            return file.read()

    except FileNotFoundError:
        print("Transcript file not found!")
        return ""


# 2. SPLIT SENTENCES

def split_sentences(text):

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# 3. CALCULATE SENTENCE IMPORTANCE

def calculate_sentence_scores(sentences):

    if not sentences:
        return []

    if len(sentences) == 1:
        return [1.0]

    vectorizer = TfidfVectorizer(stop_words="english")

    try:
        matrix = vectorizer.fit_transform(sentences)
        return matrix.sum(axis=1).A1

    except ValueError:
        return [1.0] * len(sentences)


# 4. EXTRACT KEY POINTS

def extract_key_points(sentences):

    if not sentences:
        return []

    scores = calculate_sentence_scores(sentences)

    count = min(5, len(sentences))

    ranked_indices = sorted(
        range(len(sentences)),
        key=lambda i: scores[i],
        reverse=True
    )[:count]

    ranked_indices.sort()

    return [sentences[i] for i in ranked_indices]


# 5. GENERATE SUMMARY

def generate_summary(sentences):

    if not sentences:
        return "No meeting content available."

    scores = calculate_sentence_scores(sentences)

    count = min(6, len(sentences))

    ranked_indices = sorted(
        range(len(sentences)),
        key=lambda i: scores[i],
        reverse=True
    )[:count]

    ranked_indices.sort()

    selected_sentences = [
        sentences[i] for i in ranked_indices
    ]

    return " ".join(selected_sentences)


# 6. IDENTIFY TOPICS

def classify_topic(sentence):

    text = sentence.lower()

    topic_keywords = {

        "Project Overview": [
            "project", "objective", "purpose",
            "college project", "meeting assistant"
        ],

        "Application Development": [
            "development", "speech-to-text",
            "whisper", "microphone", "audio",
            "recording", "upload", "transcription"
        ],

        "NLP Features": [
            "nlp", "summary", "summarization",
            "key points", "topic identification",
            "action items", "deadline", "accuracy",
            "extracts", "identifies"
        ],

        "Team Responsibilities": [
            "srishti", "sinchana", "rahul",
            "everyone", "assigned", "responsibilities",
            "complete", "prepare", "review"
        ],

        "Project Demonstration": [
            "demonstration", "demo", "presentation",
            "pdf report", "powerpoint"
        ],

        "Schedule and Next Steps": [
            "next week", "friday", "monday",
            "wednesday", "tomorrow", "pending work",
            "review meeting", "final submission"
        ]
    }

    scores = {}

    for topic, keywords in topic_keywords.items():

        scores[topic] = sum(
            1 for keyword in keywords
            if keyword in text
        )

    best_topic = max(scores, key=scores.get)

    if scores[best_topic] == 0:
        return "General Discussion"

    return best_topic


def identify_topics(sentences):

    if not sentences:
        return []

    grouped_topics = defaultdict(list)
    topic_order = []

    for sentence in sentences:

        topic = classify_topic(sentence)

        if topic not in grouped_topics:
            topic_order.append(topic)

        grouped_topics[topic].append(sentence)

    return [
        grouped_topics[topic]
        for topic in topic_order
    ]


# 7. GENERATE TOPIC NAME

def generate_topic_name(topic):

    if not topic:
        return "General Discussion"

    return classify_topic(" ".join(topic))


# 8. EXTRACT DEADLINE

def extract_deadline(sentence):

    match = re.search(
        DEADLINE_PATTERN,
        sentence,
        re.IGNORECASE
    )

    if match:
        return match.group(0).strip().title()

    return "Not specified"


# 9. CLEAN TASK DESCRIPTION

def clean_task(task):

    task = re.sub(
        DEADLINE_PATTERN,
        "",
        task,
        flags=re.IGNORECASE
    )

    task = re.sub(
        r'\s+so right now.*$',
        '',
        task,
        flags=re.IGNORECASE
    )

    task = re.sub(
        r'\b(?:by|before|on)\s*$',
        '',
        task,
        flags=re.IGNORECASE
    )

    task = task.strip(" .,!;:")

    return task.capitalize()


# 10. EXTRACT TASK FROM SENTENCE

def extract_task_from_sentence(sentence):

    sentence = sentence.strip()

    # First-person and group responsibilities

    group_patterns = [
        (r'^(I)\s+(?:am\s+)?supposed\s+to\s+(.+)$', "Speaker"),
        (r'^(I)\s+need\s+to\s+(.+)$', "Speaker"),
        (r'^(I)\s+have\s+to\s+(.+)$', "Speaker"),
        (r'^(I)\s+must\s+(.+)$', "Speaker"),
        (r'^(I)\s+should\s+(.+)$', "Speaker"),
        (r'^(I)\s+will\s+(.+)$', "Speaker"),

        (r'^(we)\s+(?:are\s+)?supposed\s+to\s+(.+)$', "Team"),
        (r'^(we)\s+need\s+to\s+(.+)$', "Team"),
        (r'^(we)\s+have\s+to\s+(.+)$', "Team"),
        (r'^(we)\s+must\s+(.+)$', "Team"),
        (r'^(we)\s+should\s+(.+)$', "Team"),
        (r'^(we)\s+will\s+(.+)$', "Team"),

        (r'^(everyone)\s+must\s+(.+)$', "Everyone"),
        (r'^(everyone)\s+should\s+(.+)$', "Everyone"),
        (r'^(everyone)\s+need\s+to\s+(.+)$', "Everyone"),
        (r'^(everyone)\s+needs\s+to\s+(.+)$', "Everyone"),
    ]

    for pattern, person in group_patterns:

        match = re.match(pattern, sentence, re.IGNORECASE)

        if match:

            task = clean_task(match.group(2))

            if task:
                return {
                    "Person": person,
                    "Task": task
                }

    # Named-person responsibilities.
    # This pattern is deliberately case-sensitive so that
    # ordinary phrases are not mistaken for people's names.

    person_patterns = [
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+'
        r'(?:should|must|will|needs to|has to)\s+(.+)$',

        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+'
        r'(?:is supposed to|has to)\s+(.+)$'
    ]

    for pattern in person_patterns:

        match = re.match(pattern, sentence)

        if match:

            person = match.group(1)
            task = clean_task(match.group(2))

            if task:
                return {
                    "Person": person,
                    "Task": task
                }

    return None


# 11. EXTRACT STRUCTURED TASKS

def extract_structured_tasks(sentences):

    structured_tasks = []

    for sentence in sentences:

        deadline = extract_deadline(sentence)

        task_data = extract_task_from_sentence(sentence)

        if task_data:

            task_data["Deadline"] = deadline

            structured_tasks.append(task_data)

        elif deadline != "Not specified" and structured_tasks:

            if structured_tasks[-1]["Deadline"] == "Not specified":

                structured_tasks[-1]["Deadline"] = deadline

    return structured_tasks


# 12. DETECT ACTION ITEMS

def detect_action_items(sentences):

    action_items = []

    structured_tasks = extract_structured_tasks(sentences)

    for task in structured_tasks:

        description = task["Task"]
        deadline = task["Deadline"]

        if deadline != "Not specified":

            action_items.append(
                "[DEADLINE] " + description + " - " + deadline
            )

        else:

            action_items.append(description)

    return action_items


# 13. MAIN FUNCTION

def main():

    text = read_transcript()

    if not text:
        return

    sentences = split_sentences(text)

    key_points = extract_key_points(sentences)

    summary = generate_summary(sentences)

    topics = identify_topics(sentences)

    structured_tasks = extract_structured_tasks(sentences)

    action_items = detect_action_items(sentences)

    print("\n========== MEETING TRANSCRIPT ==========")
    print(text)

    print("\n========== SUMMARY ==========")
    print(summary)

    print("\n========== KEY POINTS ==========")

    for point in key_points:
        print("-", point)

    print("\n========== TOPICS ==========")

    for i, topic in enumerate(topics, 1):

        print(f"\nTopic {i}: {generate_topic_name(topic)}")

        for sentence in topic:
            print("-", sentence)

    print("\n========== STRUCTURED ACTION ITEMS ==========")

    for task in structured_tasks:
        print(task)

    print("\n========== ACTION ITEMS ==========")

    for item in action_items:
        print("-", item)

    print("\nPROCESSING COMPLETED!")


if __name__ == "__main__":
    main()