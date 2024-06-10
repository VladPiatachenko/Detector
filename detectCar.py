import os
import cv2
import time
import argparse
import torch
import model.detector
import utils.utils


def load_label_names(file_path):
    labels = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            labels.append(line.strip())
    return labels

#To run: python detectCar.py --data coco.data --weights coco-best-model.pth --names dataset.names --video test.mp4 --output result.avi


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='', help='Specify training profile *.data')
    parser.add_argument('--weights', type=str, default='', help='The path of the .pth model to be transformed')
    parser.add_argument('--names', type=str, default='', help='The path of the file containing label names')
    parser.add_argument('--video', type=str, default='', help='Path to the input video file')
    parser.add_argument('--output', type=str, default='output.avi', help='Path to save the output video file')
    opt = parser.parse_args()

    assert os.path.exists(opt.data), "Please specify the correct data file path"
    assert os.path.exists(opt.weights), "Please specify the correct model path"
    assert os.path.exists(opt.names), "Please specify the correct label names file path"
    assert os.path.exists(opt.video), "Please specify the correct input video file path"

    cfg = utils.utils.load_datafile(opt.data)
    label_names = load_label_names(opt.names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)

    state_dict = torch.load(opt.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    cap = cv2.VideoCapture(opt.video)  # Use the specified video file
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(opt.output, fourcc, 20.0, (width, height))  # Output video with detections

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        res_img = cv2.resize(frame, (cfg["width"], cfg["height"]), interpolation=cv2.INTER_LINEAR)
        img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
        img = torch.from_numpy(img.transpose(0, 3, 1, 2))
        img = img.to(device).float() / 255.0

        # Model inference
        preds = model(img)

        # Post-process the feature maps
        output = utils.utils.handel_preds(preds, cfg, device)
        output_boxes = utils.utils.non_max_suppression(output, conf_thres=0.75,
                                                       iou_thres=0.75)

        # Draw predicted bounding boxes
        for box in output_boxes[0]:
            box = box.tolist()
            obj_score = box[4]
            category = label_names[int(box[5])]

            x1, y1 = int(box[0]), int(box[1])
            x2, y2 = int(box[2]), int(box[3])

            # Draw a tiny red rectangle border
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

            # Calculate the center of the bounding box
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            # Draw a line pointing to the center of the object
            cv2.line(frame, (x1, y1), (center_x, center_y), (0, 0, 255), 1)
            cv2.line(frame, (x2, y1), (center_x, center_y), (0, 0, 255), 1)
            cv2.line(frame, (x1, y2), (center_x, center_y), (0, 0, 255), 1)
            cv2.line(frame, (x2, y2), (center_x, center_y), (0, 0, 255), 1)

            # Draw labels
            cv2.putText(frame, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

        # Write frame with detections to output video
        out.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
