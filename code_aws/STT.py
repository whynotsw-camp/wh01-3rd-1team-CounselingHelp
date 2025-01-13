import os
import json
import requests
import boto3
import time
import logging

# S3 클라이언트 초기화
s3_client = boto3.client('s3')

# S3 버킷 및 폴더 설정
BUCKET_NAME = "aicc-alll"  # S3 버킷 이름
INPUT_FOLDER = "connect/aicc-alll/CallRecordings/ivr/"   # 입력 폴더 경로
OUTPUT_FOLDER = "stt/"  # 출력 폴더 경로

# STT API 설정
STT_CONFIG = {
    "use_diarization": True,
    "diarization": {"spk_count": 2},
    "use_itn": True,
    "use_disfluency_filter": False,
    "use_profanity_filter": False,
    "use_paragraph_splitter": True,
    "paragraph_splitter": {"max": 50}
}

CLIENT_ID = "클라이언트ID"  # 클라이언트 ID
CLIENT_SECRET = "시크릿"  # 클라이언트 시크릿

# Set up logging to file and console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
    logging.StreamHandler(),  # Print to the console
    logging.FileHandler('nohup.out', mode='a')  # Append logs to nohup.out
])

def authenticate_stt_api():
    """STT API 인증 토큰 발급"""
    logging.info("STT 인증 중...")
    auth_resp = requests.post(
        'https://openapi.vito.ai/v1/authenticate',
        data={
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET
        }
    )
    auth_resp.raise_for_status()
    logging.info("STT 인증 완료")
    return auth_resp.json().get('access_token')

def get_next_file_number(bucket_name, output_folder):
    """S3에 저장된 stt 파일 중 다음 파일 번호를 계산"""
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=output_folder)
        if 'Contents' in response:
            file_numbers = []
            for obj in response['Contents']:
                filename = os.path.basename(obj['Key'])
                if filename.startswith("stt_") and filename.endswith(".txt"):
                    try:
                        number = int(filename[4:-4])  # stt_와 .txt 사이의 숫자 추출
                        file_numbers.append(number)
                    except ValueError:
                        continue
            next_file_number = max(file_numbers) + 1 if file_numbers else 1
            logging.info(f"다음 파일 번호: {next_file_number}")
            return next_file_number
        else:
            logging.info("이전에 저장된 STT 파일이 없습니다. 첫 번째 번호로 시작합니다.")
            return 1
    except Exception as e:
        logging.error(f"다음 파일 번호 계산 중 오류 발생: {e}")
        return 1

def process_audio_file(bucket_name, object_key, token):
    """S3에서 파일 처리 후 변환 결과 저장"""
    local_audio_file = f"/tmp/{os.path.basename(object_key)}"
    local_text_file = f"/tmp/{os.path.basename(object_key).replace('.wav', '.txt')}"

    try:
        # S3에서 음성 파일 다운로드
        logging.info(f"처리 중인 파일: {object_key}")
        s3_client.download_file(bucket_name, object_key, local_audio_file)

        # STT API 호출
        logging.info(f"STT 시작 - 파일 처리 중: {object_key}")
        with open(local_audio_file, 'rb') as audio_file:
            stt_resp = requests.post(
                'https://openapi.vito.ai/v1/transcribe',
                headers={'Authorization': f'bearer {token}'},
                data={'config': json.dumps(STT_CONFIG)},
                files={'file': audio_file}
            )
        stt_resp.raise_for_status()
        transcription_id = stt_resp.json().get('id')

        # STT 변환 결과 확인
        elapsed_time = 0
        while elapsed_time < 60:
            logging.info(f"STT 결과 확인 중 - ID: {transcription_id}")
            result_resp = requests.get(
                f'https://openapi.vito.ai/v1/transcribe/{transcription_id}',
                headers={'Authorization': f'bearer {token}'}
            )
            result_resp.raise_for_status()
            result = result_resp.json()

            if 'results' in result and 'utterances' in result['results']:
                utterances = result['results']['utterances']
                text_output = '\n'.join([utterance['msg'] for utterance in utterances])

                # S3의 다음 파일 번호 계산
                next_file_number = get_next_file_number(bucket_name, OUTPUT_FOLDER)
                output_key = f"{OUTPUT_FOLDER}stt_{next_file_number}.txt"

                # 텍스트 파일로 저장
                with open(local_text_file, 'w', encoding='utf-8') as file:
                    file.write(text_output)

                # S3에 업로드
                s3_client.upload_file(local_text_file, bucket_name, output_key)
                logging.info(f"STT 종료 및 파일 저장 - 텍스트 파일이 S3에 업로드되었습니다: {bucket_name}/{output_key}")
                break
            elif result.get('status') == 'transcribing':
                logging.info("STT 결과가 아직 처리 중입니다...")
                elapsed_time += 5
                time.sleep(5)
            else:
                logging.error(f"STT 실패 또는 문제 발생: {result}")
                break
    except Exception as e:
        logging.error(f"파일 처리 중 오류 발생 - {object_key}: {e}")
    finally:
        # 임시 파일 삭제
        if os.path.exists(local_audio_file):
            os.remove(local_audio_file)
        if os.path.exists(local_text_file):
            os.remove(local_text_file)

def monitor_s3_bucket():
    """S3 버킷 모니터링"""
    processed_files = set()
    token = authenticate_stt_api()

    while True:
        try:
            # S3에서 객체 목록 가져오기
            response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=INPUT_FOLDER)
            if 'Contents' in response:
                for obj in response['Contents']:
                    object_key = obj['Key']

                    if object_key.endswith(".wav") and object_key not in processed_files:
                        logging.info(f"새 파일 감지됨: {object_key}")
                        process_audio_file(BUCKET_NAME, object_key, token)
                        processed_files.add(object_key)
        except Exception as e:
            logging.error(f"S3 버킷 모니터링 중 오류 발생: {e}")

        # 1초마다 S3 버킷 확인
        time.sleep(1)

if __name__ == "__main__":
    logging.info("S3 버킷 모니터링 시작...")
    monitor_s3_bucket()
