import boto3
import time
import os
from google.cloud import texttospeech
import logging
import subprocess

# Set up logging to file and console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
    logging.StreamHandler(),  # Print to the console
    logging.FileHandler('nohup.out', mode='a')  # Append logs to nohup.out
])

# S3 클라이언트 초기화
s3 = boto3.client('s3')

# 처리한 파일 목록을 저장할 집합
processed_files = set()


def read_text_from_s3(bucket_name, file_key):
    # S3에서 텍스트 파일 읽기
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    return response['Body'].read().decode('utf-8')


def synthesize_text(text):
    # Google TTS 클라이언트 초기화
    client = texttospeech.TextToSpeechClient()

    # SSML 태그를 추가하여 발음을 교정
    ssml_text = "<speak>{}</speak>".format(
        text.replace("문의", '<phoneme alphabet="ipa" ph="muni">문의</phoneme>')  # 예시: 문의 발음 수정
    )

    # 요청 구성
    input_text = texttospeech.SynthesisInput(ssml=ssml_text)
    voice = texttospeech.VoiceSelectionParams(
        language_code='ko-KR',
        name='ko-KR-Wavenet-A',  # WaveNet 음성 모델
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    # TTS 요청
    response = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )

    return response.audio_content


def convert_to_aws_connect_format(input_path, output_path):
    """FFmpeg를 사용하여 AWS Connect 요구사항에 맞게 파일 변환"""
    command = [
        "ffmpeg",
        "-i", input_path,
        "-ar", "8000",  # 샘플링 레이트 8kHz
        "-ac", "1",  # Mono
        "-c:a", "pcm_mulaw",  # PCM μ-law 코덱
        "-f", "wav",  # 파일 형식 명시
        output_path
    ]
    subprocess.run(command, check=True)


def get_next_tts_file_name(bucket_name, prefix='tts/'):
    # S3에서 tts/ 경로의 파일 리스트 가져오기
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    max_index = 0

    if 'Contents' in response:
        for obj in response['Contents']:
            file_key = obj['Key']
            # 파일 이름에서 번호를 추출 (예: tts_1.wav -> 1)
            base_name = os.path.basename(file_key)
            if base_name.startswith('tts_') and base_name.endswith('.wav'):
                try:
                    index = int(base_name[4:-4])  # tts_ 이후 숫자만 추출
                    max_index = max(max_index, index)
                except ValueError:
                    continue

    # 다음 번호 생성
    return f"tts/tts_{max_index + 1}.wav"


def upload_to_s3(audio_content, bucket_name, object_name):
    temp_file = "/tmp/temp_audio_file.wav"
    converted_file = "/tmp/converted_audio_file.wav"

    with open(temp_file, "wb") as f:
        f.write(audio_content)

    # 파일 변환
    convert_to_aws_connect_format(temp_file, converted_file)

    # S3에 변환된 파일 업로드
    s3.upload_file(converted_file, bucket_name, object_name)
    logging.info(f"파일 {converted_file}이 {bucket_name}/{object_name}에 업로드되었습니다.")

    # 임시 파일 삭제
    os.remove(temp_file)
    os.remove(converted_file)


def process_files(bucket_name):
    # S3에서 'nlp/' 폴더 내 텍스트 파일 가져오기
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix='nlp/')
    if 'Contents' in response:
        for obj in response['Contents']:
            file_key = obj['Key']

            # 텍스트 파일만 처리하고, 아직 처리되지 않은 파일인지 확인
            if file_key.endswith('.txt') and file_key not in processed_files:
                logging.info(f"처리 중인 파일: {file_key}")

                # 텍스트 파일 읽기
                text = read_text_from_s3(bucket_name, file_key)

                # 텍스트에 "문의하신 내용이" 추가
                text = f"  문의하신 내용이 {text}"

                # TTS 시작 로그
                logging.info(f"TTS 시작 - Processing text file: {file_key}")

                # 텍스트를 음성으로 변환
                audio_content = synthesize_text(text)

                # 오름차순 파일 이름 생성
                wav_file_key = get_next_tts_file_name(bucket_name)

                # WAV 파일을 S3에 업로드
                upload_to_s3(audio_content, bucket_name, wav_file_key)

                # TTS 종료 및 파일 저장 로그
                logging.info(f"TTS 종료 및 파일 저장 - Audio file uploaded to S3: {wav_file_key}")

                # 처리한 파일 목록에 추가
                processed_files.add(file_key)


def main():
    bucket_name = 'aicc-alll'

    while True:
        process_files(bucket_name)

        # 주기적으로 S3를 확인 (예: 1초마다 확인)
        time.sleep(1)


if __name__ == '__main__':
    main()
