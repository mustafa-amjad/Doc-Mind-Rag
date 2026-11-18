from types import SimpleNamespace

from docmind import voice


class FakeTranscriptions:
    def __init__(self, text):
        self.text, self.calls = text, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.text)


def fake_client(text):
    transcriptions = FakeTranscriptions(text)
    return SimpleNamespace(audio=SimpleNamespace(transcriptions=transcriptions)), transcriptions


def test_transcribe_sends_the_audio_to_whisper():
    client, calls = fake_client(" How long do I have to return a jacket? ")
    assert voice.transcribe(client, b"RIFF...") == "How long do I have to return a jacket?"
    assert calls.calls[0]["model"] == voice.STT_MODEL and calls.calls[0]["file"][1] == b"RIFF..."


def test_silence_and_empty_audio_give_no_question():
    client, calls = fake_client(" . ")
    assert voice.transcribe(client, b"RIFF...") == ""
    assert voice.transcribe(client, b"") == "" and len(calls.calls) == 1  # no API call without audio


def test_speech_text_drops_citations_and_markdown():
    answer = "You have **30 days** to return a jacket [1]. Gift cards can't be returned [2].\n- Tags must be attached"
    assert voice.speech_text(answer) == ("You have 30 days to return a jacket. Gift cards can't be returned. "
                                         "Tags must be attached")


def test_long_answers_are_cut_at_a_sentence():
    text = voice.speech_text("This is one sentence. " * 200)
    assert len(text) <= voice.MAX_SPOKEN_CHARS and text.endswith(".")


def wav(samples) -> bytes:
    import io
    import wave

    import numpy as np

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(np.asarray(samples, dtype=np.int16).tobytes())
    return buffer.getvalue()


def speechlike(seconds=1.5):
    """A loud tone between quiet gaps, like words between pauses."""
    import numpy as np

    t = np.arange(int(16000 * seconds)) / 16000
    tone = 4000 * np.sin(2 * np.pi * 220 * t)
    gap = np.zeros(8000)
    return np.concatenate([gap, tone, gap])


def test_silence_is_not_sent_to_whisper():
    client, calls = fake_client("Thank you.")
    assert voice.transcribe(client, wav([0] * 48000)) == "" and calls.calls == []


def test_room_noise_is_not_sent_to_whisper():
    import numpy as np

    client, calls = fake_client("Ungri eruð þvís.")
    noise = np.random.default_rng(1).normal(0, 120, 48000)
    assert voice.transcribe(client, wav(noise)) == "" and calls.calls == []


def test_speech_is_sent_in_english_with_a_hint():
    client, calls = fake_client("How long do I have to return a jacket?")
    assert voice.transcribe(client, wav(speechlike())) == "How long do I have to return a jacket?"
    sent = calls.calls[0]
    assert sent["language"] == voice.STT_LANGUAGE == "en" and sent["prompt"] == voice.STT_PROMPT


def test_whisper_filler_is_dropped():
    for filler in ("Thank you.", "Thanks for watching!", "you"):
        client, _ = fake_client(filler)
        assert voice.transcribe(client, wav(speechlike())) == "", filler
