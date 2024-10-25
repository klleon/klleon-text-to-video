import os
import cv2

def convert_videos_to_pngs(video_directory, output_directory):
    # 비디오 파일을 포함한 하위 디렉토리 찾기
    for root, _, files in os.walk(video_directory):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):  # 지원하는 비디오 파일 형식
                video_path = os.path.join(root, file)
                #print(f"Processing video: {video_path}")
                filename = file.split('.mp4')[0]
                
                # 비디오 캡처 객체 생성
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Error opening video file: {video_path}")
                    continue
                
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break  # 비디오의 끝에 도달했거나 오류 발생

                    # 출력 경로 설정
                    output_folder = output_directory
                    os.makedirs(output_folder, exist_ok=True)

                    # if frame_count > 2:
                    # 프레임을 PNG 파일로 저장
                    output_path = os.path.join(output_folder, f'{filename}_{frame_count:04d}.png')
                    cv2.imwrite(output_path, frame)
                    frame_count += 1
                    break
                
                cap.release()
                print(f"Extracted {frame_count} frames from {file}")

if __name__ == "__main__":
    video_directory = "results/"  # 비디오 파일이 있는 디렉토리 경로
    output_directory = "results_img/"  # PNG 파일을 저장할 디렉토리 경로
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    convert_videos_to_pngs(video_directory, output_directory)

