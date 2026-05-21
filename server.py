"""
VAEP 분석 서버
Flask API: 비디오 업로드 → VAEP 분석 → 오버레이 영상 저장 → 다운로드
"""

import os
import json
import uuid
import threading
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static')
CORS(app)

# 경로 설정 (Windows/Linux 공통)
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024  # 2GB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

jobs: dict = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_job(job_id: str, video_path: str):
    """백그라운드 처리: VAEP 분석 → 오버레이 영상 생성"""
    jobs[job_id]['status'] = 'processing'
    jobs[job_id]['progress'] = 0
    jobs[job_id]['step'] = 'VAEP 분석 중...'

    try:
        from video_processor import VideoProcessor

        def progress_cb(current, total):
            jobs[job_id]['progress'] = int(current / total * 50)  # 0~50%

        # 1단계: VAEP 분석
        show_names = jobs[job_id].get("show_names", False)
        player_names = jobs[job_id].get("player_names", {})
        processor = VideoProcessor(use_yolo=True)
        result = processor.process_video(
            video_path,
            progress_callback=progress_cb,
            sample_rate=3
        )

        # 결과 JSON 저장
        json_path = os.path.join(OUTPUT_FOLDER, f"{job_id}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        jobs[job_id]['step'] = '오버레이 영상 렌더링 중...'
        jobs[job_id]['progress'] = 50

        # 2단계: 오버레이 영상 렌더링
        from overlay_renderer import render_vaep_overlay

        overlay_path = os.path.join(OUTPUT_FOLDER, f"{job_id}_vaep.mp4")

        def overlay_progress(current, total):
            jobs[job_id]['progress'] = 50 + int(current / total * 50)  # 50~100%

        render_vaep_overlay(video_path, result, overlay_path, overlay_progress,
            show_names=jobs[job_id].get("show_names", False),
            player_names=jobs[job_id].get("player_names", {}))

        jobs[job_id]['status']       = 'done'
        jobs[job_id]['progress']     = 100
        jobs[job_id]['step']         = '완료'
        jobs[job_id]['result_path']  = json_path
        jobs[job_id]['overlay_path'] = overlay_path
        jobs[job_id]['summary'] = {
            'total_frames':     result['total_frames'],
            'processed_frames': result['processed_frames'],
            'duration':         round(result['duration'], 1),
            'total_events':     result['total_events'],
        }
        print(f"[INFO] Job {job_id} 완료 → {overlay_path}")

    except Exception as e:
        import traceback
        jobs[job_id]['status']    = 'error'
        jobs[job_id]['error']     = str(e)
        jobs[job_id]['traceback'] = traceback.format_exc()
        print(f"[ERROR] Job {job_id}: {e}")
        traceback.print_exc()


# ──────────────────────────────────────────────
# 라우트
# ──────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': '비디오 파일이 없습니다'}), 400

    file = request.files['video']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': '지원하지 않는 형식입니다'}), 400

    job_id   = str(uuid.uuid4())[:8]
    filename = secure_filename(f"{job_id}_{file.filename}")
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    # 대용량 파일 청크 단위 저장
    file.save(video_path)

    jobs[job_id] = {
        'status': 'queued', 'progress': 0,
        'step': '대기 중', 'video_path': video_path,
        'show_names': request.form.get('show_names', '0') == '1',
        'player_names': json.loads(request.form.get('player_names', '{}')),
        'color_a': request.form.get('color_a', '#dc2626'),
        'color_b': request.form.get('color_b', '#3b82f6'),
    }

    t = threading.Thread(target=process_job, args=(job_id, video_path), daemon=True)
    t.start()

    return jsonify({'job_id': job_id, 'status': 'queued'})


@app.route('/api/status/<job_id>')
def job_status(job_id):
    if job_id not in jobs:
        return jsonify({'error': '작업을 찾을 수 없습니다'}), 404
    return jsonify(jobs[job_id])


@app.route('/api/result/<job_id>')
def job_result(job_id):
    if job_id not in jobs:
        return jsonify({'error': '작업을 찾을 수 없습니다'}), 404
    job = jobs[job_id]
    if job['status'] != 'done':
        return jsonify({'error': '처리 중', 'status': job['status']}), 202

    with open(job['result_path'], 'r', encoding='utf-8') as f:
        return jsonify(json.load(f))


@app.route('/api/download/<job_id>')
def download_overlay(job_id):
    """VAEP 오버레이 mp4 다운로드"""
    if job_id not in jobs:
        return jsonify({'error': '작업을 찾을 수 없습니다'}), 404
    job = jobs[job_id]
    if job['status'] != 'done':
        return jsonify({'error': '아직 처리 중입니다'}), 202

    overlay_path = job.get('overlay_path')
    if not overlay_path or not os.path.exists(overlay_path):
        return jsonify({'error': '오버레이 파일이 없습니다'}), 500

    return send_file(
        overlay_path,
        mimetype='video/mp4',
        as_attachment=True,
        download_name=f"vaep_analysis_{job_id}.mp4"
    )


if __name__ == '__main__':
    print("=" * 50)
    print("  VAEP 분석 서버 시작")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
