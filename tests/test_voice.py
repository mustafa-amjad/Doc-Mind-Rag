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
