"""Azure Speech Service - Text-to-Speech synthesis."""

import os
import re
import logging


def synthesize_speech(text: str) -> bytes | None:
    """Convert text to speech using Azure Speech Service. Returns MP3 bytes."""
    key = os.environ.get("AZURE_SPEECH_KEY")
    region = os.environ.get("AZURE_SPEECH_REGION", "eastus2")
    if not key:
        logging.warning("AZURE_SPEECH_KEY not set, TTS disabled")
        return None

    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError:
        logging.warning("azure-cognitiveservices-speech not installed, TTS disabled")
        return None

    config = speechsdk.SpeechConfig(subscription=key, region=region)
    config.speech_synthesis_voice_name = "it-IT-IsabellaNeural"
    config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
    )

    synthesizer = speechsdk.SpeechSynthesizer(speech_config=config, audio_config=None)
    # Strip markdown for cleaner speech
    clean = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    clean = re.sub(r'[#*`|]', '', clean)
    clean = clean.replace("---", "").strip()
    if len(clean) > 3000:
        clean = clean[:3000] + "... testo troncato."

    result = synthesizer.speak_text_async(clean).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return result.audio_data
    logging.error(f"TTS failed: {result.reason}")
    return None
