"""Azure Speech Service - Text-to-Speech synthesis."""

import os
import logging
import azure.cognitiveservices.speech as speechsdk


def synthesize_speech(text: str) -> bytes | None:
    """Convert text to speech using Azure Speech Service. Returns WAV bytes."""
    key = os.environ.get("AZURE_SPEECH_KEY")
    region = os.environ.get("AZURE_SPEECH_REGION", "eastus2")
    if not key:
        logging.warning("AZURE_SPEECH_KEY not set, TTS disabled")
        return None

    config = speechsdk.SpeechConfig(subscription=key, region=region)
    config.speech_synthesis_voice_name = "it-IT-IsabellaNeural"
    config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
    )

    synthesizer = speechsdk.SpeechSynthesizer(speech_config=config, audio_config=None)
    # Strip markdown for cleaner speech
    clean = text.replace("**", "").replace("```mermaid", "").replace("```", "")
    clean = clean.replace("#", "").replace("---", "").strip()
    # Truncate to avoid long synthesis
    if len(clean) > 3000:
        clean = clean[:3000] + "... testo troncato."

    result = synthesizer.speak_text_async(clean).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return result.audio_data
    logging.error(f"TTS failed: {result.reason}")
    return None
