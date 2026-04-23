import os
import uuid
from google.cloud import storage, translate_v2 as translate
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

# Prerequisites (one-time setup):
#   gcloud auth application-default login
#   export GOOGLE_CLOUD_PROJECT=your-project-id

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "project-e264ab5e-738a-4066-88b")
AUDIO_FILE = os.environ.get("AUDIO_FILE", "output_trimmed.flac")
BUCKET_NAME = os.environ.get("GCS_BUCKET", f"{PROJECT_ID}-stt-tmp")


def get_or_create_bucket(bucket_name: str) -> str:
    storage_client = storage.Client(project=PROJECT_ID)
    try:
        bucket = storage_client.get_bucket(bucket_name)
        print(f"Using existing bucket: {bucket_name}")
    except Exception:
        print(f"Creating bucket: {bucket_name}")
        bucket = storage_client.create_bucket(bucket_name, location="us-central1")
    return bucket_name


def upload_to_gcs(local_path: str, bucket_name: str) -> str:
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blob_name = f"audio/{uuid.uuid4().hex}_{os.path.basename(local_path)}"
    blob = bucket.blob(blob_name)
    print(f"Uploading {local_path} to gs://{bucket_name}/{blob_name} ...")
    blob.upload_from_filename(local_path)
    print("Upload complete.")
    return f"gs://{bucket_name}/{blob_name}"


def transcribe_hindi(audio_file: str):
    # Upload to GCS
    bucket_name = get_or_create_bucket(BUCKET_NAME)
    gcs_uri = upload_to_gcs(audio_file, bucket_name)

    client = SpeechClient()

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=["hi-IN", "en-IN"],
        model="long",
    )

    request = cloud_speech.BatchRecognizeRequest(
        recognizer=f"projects/{PROJECT_ID}/locations/global/recognizers/_",
        config=config,
        files=[cloud_speech.BatchRecognizeFileMetadata(uri=gcs_uri)],
        recognition_output_config=cloud_speech.RecognitionOutputConfig(
            inline_response_config=cloud_speech.InlineOutputConfig(),
        ),
    )

    print("Transcribing (this may take a minute)...")
    operation = client.batch_recognize(request=request)
    response = operation.result(timeout=300)

    print("\n=== Transcript ===")
    found = False
    for uri, result in response.results.items():
        for res in result.transcript.results:
            if res.alternatives:
                alt = res.alternatives[0]
                print(alt.transcript)
                if alt.confidence:
                    print(f"Confidence: {alt.confidence:.2f}")
                found = True

    if not found:
        print("(no results returned — check audio file or language code)")
        return

    # Translate full transcript to English
    full_text = " ".join(
        res.alternatives[0].transcript
        for _, result in response.results.items()
        for res in result.transcript.results
        if res.alternatives
    )

    translate_client = translate.Client()
    translation = translate_client.translate(full_text, target_language="en", source_language="hi")

    print("\n=== English Translation ===")
    print(translation["translatedText"])


if __name__ == "__main__":
    transcribe_hindi(AUDIO_FILE)
