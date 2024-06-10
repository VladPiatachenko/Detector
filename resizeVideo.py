import cv2
import argparse


def resize_video(input_video, output_video, target_width=640, target_height=640):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, (target_width, target_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (target_width, target_height))
        out.write(resized_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the resized output video file')
    parser.add_argument('--width', type=int, default=352, help='Target width for resizing (default: 640)')
    parser.add_argument('--height', type=int, default=352, help='Target height for resizing (default: 640)')
    args = parser.parse_args()

    resize_video(args.input, args.output, args.width, args.height)
