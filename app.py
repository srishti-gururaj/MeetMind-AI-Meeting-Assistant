import streamlit as st
import tempfile
import os
from io import BytesIO
from faster_whisper import WhisperModel

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from xml.sax.saxutils import escape

from nlp_processor import (
    split_sentences,
    extract_key_points,
    generate_summary,
    identify_topics,
    generate_topic_name,
    extract_structured_tasks
)


# PAGE CONFIGURATION

st.set_page_config(
    page_title="MeetMind",
    page_icon="M",
    layout="wide"
)


# CUSTOM CSS

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: #4A90E2;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
}

.section-title {
    font-size: 24px;
    font-weight: bold;
    color: #4A90E2;
}
</style>
""", unsafe_allow_html=True)


# TITLE

st.markdown(
    '<div class="main-title">MeetMind</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Meeting Assistant</div>',
    unsafe_allow_html=True
)

st.write("")

st.info(
    "Record or upload a meeting audio file to generate "
    "transcripts, summaries, important points and action items."
)


# LOAD WHISPER MODEL

@st.cache_resource
def load_model():

    model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"
    )

    return model


# TRANSCRIBE AUDIO

def transcribe_audio(audio_bytes, file_extension):

    model = load_model()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=file_extension
    ) as temp_audio:

        temp_audio.write(audio_bytes)
        audio_path = temp_audio.name

    try:

        segments, info = model.transcribe(
            audio_path,
            beam_size=5
        )

        transcript = ""

        for segment in segments:
            transcript += segment.text.strip() + " "

        return transcript.strip()

    finally:

        if os.path.exists(audio_path):
            os.remove(audio_path)


# GENERATE PDF REPORT

def generate_pdf(
    transcript,
    summary,
    key_points,
    topics,
    structured_tasks
):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontSize=24,
        textColor=colors.HexColor("#24476B"),
        spaceAfter=10
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontSize=15,
        textColor=colors.HexColor("#24476B"),
        spaceBefore=15,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "NormalStyle",
        parent=styles["BodyText"],
        fontSize=10,
        leading=15,
        spaceAfter=6
    )

    story = []

    story.append(
        Paragraph("MeetMind", title_style)
    )

    story.append(
        Paragraph(
            "AI-Powered Meeting Assistant",
            normal_style
        )
    )

    story.append(Spacer(1, 15))

    # SUMMARY

    story.append(
        Paragraph("1. Meeting Summary", heading_style)
    )

    story.append(
        Paragraph(
            escape(summary) if summary else "No summary available.",
            normal_style
        )
    )

    # KEY POINTS

    story.append(
        Paragraph("2. Important Key Points", heading_style)
    )

    if key_points:

        for point in key_points:

            story.append(
                Paragraph(
                    "- " + escape(point),
                    normal_style
                )
            )

    else:

        story.append(
            Paragraph("No key points available.", normal_style)
        )

    # TOPICS

    story.append(
        Paragraph("3. Topic-wise Discussion", heading_style)
    )

    if topics:

        for index, topic in enumerate(topics, 1):

            topic_name = generate_topic_name(topic)

            story.append(
                Paragraph(
                    f"Topic {index}: {escape(topic_name)}",
                    normal_style
                )
            )

            for sentence in topic:

                story.append(
                    Paragraph(
                        "- " + escape(sentence),
                        normal_style
                    )
                )

    else:

        story.append(
            Paragraph("No topics detected.", normal_style)
        )

    # ACTION ITEMS

    story.append(
        Paragraph("4. Action Items and Deadlines", heading_style)
    )

    if structured_tasks:

        task_data = [
            [
                Paragraph("<b>Person</b>", normal_style),
                Paragraph("<b>Task</b>", normal_style),
                Paragraph("<b>Deadline</b>", normal_style)
            ]
        ]

        for task in structured_tasks:

            task_data.append([
                Paragraph(
                    escape(str(task["Person"])),
                    normal_style
                ),
                Paragraph(
                    escape(str(task["Task"])),
                    normal_style
                ),
                Paragraph(
                    escape(str(task["Deadline"])),
                    normal_style
                )
            ])

        task_table = Table(
            task_data,
            colWidths=[100, 260, 110],
            repeatRows=1
        )

        task_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0),
                 colors.HexColor("#24476B")),

                ("TEXTCOLOR", (0, 0), (-1, 0),
                 colors.white),

                ("GRID", (0, 0), (-1, -1), 0.5,
                 colors.grey),

                ("VALIGN", (0, 0), (-1, -1), "TOP"),

                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),

                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8)
            ])
        )

        story.append(task_table)

    else:

        story.append(
            Paragraph("No action items detected.", normal_style)
        )

    # FULL TRANSCRIPT

    story.append(
        Paragraph("5. Full Meeting Transcript", heading_style)
    )

    story.append(
        Paragraph(
            escape(transcript).replace("\n", "<br/>"),
            normal_style
        )
    )

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Generated by MeetMind - AI-Powered Meeting Assistant",
            normal_style
        )
    )

    document.build(story)

    buffer.seek(0)

    return buffer


# AUDIO INPUT

st.markdown(
    '<div class="section-title">1. Audio Input</div>',
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs([
    "Record Meeting",
    "Upload Audio"
])

audio_file = None

with tab1:

    st.write("Record your meeting using your microphone.")

    recorded_audio = st.audio_input(
        "Click to record audio"
    )

    if recorded_audio is not None:
        audio_file = recorded_audio

with tab2:

    st.write("Upload an existing meeting recording.")

    uploaded_audio = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a"]
    )

    if uploaded_audio is not None:
        audio_file = uploaded_audio


# ANALYZE MEETING

if audio_file is not None:

    st.audio(audio_file)

    if st.button(
        "Analyze Meeting",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner("Transcribing audio... Please wait."):

                file_bytes = audio_file.getvalue()

                file_extension = os.path.splitext(
                    audio_file.name
                )[1]

                if not file_extension:
                    file_extension = ".wav"

                transcript = transcribe_audio(
                    file_bytes,
                    file_extension
                )

            if not transcript:

                st.error("No speech detected. Try another recording.")

            else:

                with st.spinner("Analyzing meeting using NLP..."):

                    sentences = split_sentences(transcript)

                    summary = generate_summary(sentences)

                    key_points = extract_key_points(sentences)

                    topics = identify_topics(sentences)

                    structured_tasks = extract_structured_tasks(
                        sentences
                    )

                    st.session_state["meeting_results"] = {
                        "transcript": transcript,
                        "summary": summary,
                        "key_points": key_points,
                        "topics": topics,
                        "structured_tasks": structured_tasks
                    }

                st.success("Meeting analyzed successfully!")

        except Exception as error:

            st.error(f"An error occurred: {error}")


# DISPLAY RESULTS

if "meeting_results" in st.session_state:

    results = st.session_state["meeting_results"]

    transcript = results["transcript"]
    summary = results["summary"]
    key_points = results["key_points"]
    topics = results["topics"]
    structured_tasks = results["structured_tasks"]

    st.markdown("---")

    # TRANSCRIPT

    st.markdown(
        '<div class="section-title">2. Full Transcript</div>',
        unsafe_allow_html=True
    )

    st.text_area(
        "Transcribed Meeting",
        transcript,
        height=180
    )

    st.download_button(
        "Download Transcript",
        data=transcript,
        file_name="meeting_transcript.txt",
        mime="text/plain"
    )

    # SUMMARY

    st.markdown("---")

    st.markdown(
        '<div class="section-title">3. Meeting Summary</div>',
        unsafe_allow_html=True
    )

    st.write(summary)

    # KEY POINTS

    st.markdown("---")

    st.markdown(
        '<div class="section-title">4. Important Key Points</div>',
        unsafe_allow_html=True
    )

    if key_points:

        for point in key_points:
            st.markdown("- " + point)

    else:

        st.info("No key points found.")

    # TOPICS

    st.markdown("---")

    st.markdown(
        '<div class="section-title">5. Topic-wise Discussion</div>',
        unsafe_allow_html=True
    )

    if topics:

        for index, topic in enumerate(topics, 1):

            topic_name = generate_topic_name(topic)

            with st.expander(
                f"Topic {index}: {topic_name}",
                expanded=True
            ):

                for sentence in topic:
                    st.write("- " + sentence)

    else:

        st.info("No topics detected.")

    # ACTION ITEMS AND DEADLINES

    st.markdown("---")

    st.markdown(
        '<div class="section-title">6. Action Items and Deadlines</div>',
        unsafe_allow_html=True
    )

    if structured_tasks:

        # TABLE HEADER

        header1, header2, header3 = st.columns([1, 3, 1.5])

        with header1:
            st.markdown("**Person**")

        with header2:
            st.markdown("**Task**")

        with header3:
            st.markdown("**Deadline**")

        st.divider()

        # TABLE ROWS

        for task in structured_tasks:

            col1, col2, col3 = st.columns([1, 3, 1.5])

            with col1:
                st.write(task["Person"])

            with col2:
                st.write(task["Task"])

            with col3:
                st.write(task["Deadline"])

            st.divider()

        # CHECKLIST

        st.write("### Task Checklist")

        for index, task in enumerate(structured_tasks):

            st.checkbox(
                f'{task["Person"]}: {task["Task"]} | Deadline: {task["Deadline"]}',
                key=f"task_checkbox_{index}"
            )

    else:

        st.info("No action items detected in this meeting.")

    # PDF REPORT

    st.markdown("---")

    st.markdown(
        '<div class="section-title">7. Download Meeting Report</div>',
        unsafe_allow_html=True
    )

    pdf_file = generate_pdf(
        transcript,
        summary,
        key_points,
        topics,
        structured_tasks
    )

    st.download_button(
        label="Download Complete Meeting Report (PDF)",
        data=pdf_file,
        file_name="MeetMind_Meeting_Report.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    st.success("Your meeting report is ready!")


# FOOTER

st.markdown("---")

st.markdown(
    "<center>MeetMind | AI-Powered Meeting Assistant</center>",
    unsafe_allow_html=True
)