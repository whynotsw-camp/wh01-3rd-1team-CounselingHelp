from flask import Flask, jsonify
import boto3
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
import threading
import time
import os
# 1. input_folder에 txt파일이 생성
# 2. "nlp_final"을 사용(summary)하여
# 3. 결과를 summary_folder에 저장하는 코드
app = Flask(__name__)

# S3 설정
s3 = boto3.client('s3')
bucket_name = "aicc-alll" # 버킷이름
input_folder = "stt/" # STT 파일 저장 경로
summary_folder = "nlp/" # 요약한 파일 저장 경로

# 모델 로드
model_dir = "nlp_final"
model = BartForConditionalGeneration.from_pretrained(model_dir)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)

# 이미 처리된 파일 리스트를 저장할 Set
processed_files = set()

def process_file(file_key):
    """
    S3에서 파일 다운로드, 요약 처리, 결과 업로드
    """
    try:
        # 파일 다운로드
        original_filename = file_key.split('/')[-1]  # 원래 파일 이름
        input_path = f"/tmp/{original_filename}"
        s3.download_file(bucket_name, file_key, input_path)

        # 모델 처리
        with open(input_path, 'r', encoding='utf-8') as f:
            input_text = f.read()

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=1024)
        summary_ids = model.generate(inputs["input_ids"], max_length=128, num_beams=4, early_stopping=True)
        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # 결과를 summary 폴더에 저장할 새로운 파일명 생성
        new_filename = original_filename.replace("stt_", "nlp_")  # 파일명 변경
        output_key = f"{summary_folder}{new_filename}"  # 지정된 파일명을 summary_folder에 저장
        output_path = f"/tmp/{new_filename}"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)

        # S3에 업로드
        s3.upload_file(output_path, bucket_name, output_key)
        print(f"NLP 종료 및 파일 저장: {file_key} as {new_filename}")
    except Exception as e:
        print(f"Error processing file {file_key}: {e}")

def monitor_s3():
    """
    S3 폴더를 주기적으로 확인하여 새로운 파일을 처리
    """
    while True:
        try:
            response = s3.list_objects_v2(Bucket=bucket_name, Prefix=input_folder)
            if 'Contents' in response:
                for obj in response['Contents']:
                    file_key = obj['Key']
                    if file_key not in processed_files and file_key.endswith('.txt'):
                        print(f"NLP 시작: {file_key}")
                        process_file(file_key)
                        processed_files.add(file_key)
        except Exception as e:
            print(f"Error monitoring S3: {e}")

        time.sleep(1)  # 1초마다 폴더 확인

@app.route('/status', methods=['GET'])
def status():
    """
    현재 처리된 파일 목록을 반환
    """
    return jsonify({"processed_files": list(processed_files)})

if __name__ == "__main__":
    # 백그라운드 스레드로 S3 모니터링 시작
    monitor_thread = threading.Thread(target=monitor_s3, daemon=True)
    monitor_thread.start()

    # Flask 앱 실행
    app.run(host='0.0.0.0', port=5000)