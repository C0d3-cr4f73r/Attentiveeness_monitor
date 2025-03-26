import cv2
from flask import Flask, render_template, Response, redirect, url_for
import mediapipe as mp
import datetime
import csv
from utils import mediapipe_detection, predict_attentiveness, draw_styled_landmarks
import io
import os
import pandas as pd
import matplotlib.pyplot as plt



# Customize these based on your setup
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

# Initialize MediaPipe Holistic model (replace with your loading method if using pre-trained)
mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

# Flask app setup
app = Flask(__name__)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize global variables
recording = False
timestamps = []
attentiveness_levels = []
detect = False

# Route for the main web page
@app.route('/')
def index():
    return render_template('index.html', recording=recording, timestamps=timestamps, attentiveness_levels=attentiveness_levels)

# Route for video streaming
def generate_frames():
    global recording, timestamps, attentiveness_levels, detect

    # Initialize MediaPipe Face Detection for multi-face support
    mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue  # ✅ Retry reading instead of breaking

        if detect:
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_face.process(image_rgb)

            if results.detections:
                for detection in results.detections:
                    # Extract bounding box per face
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                 int(bboxC.width * iw), int(bboxC.height * ih)

                    # ✅ Ensure bounding box is within frame
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, iw - x)
                    h = min(h, ih - y)

                    face_roi = frame[y:y+h, x:x+w]

                    # ✅ Skip if face_roi is invalid or empty
                    if face_roi is None or face_roi.size == 0:
                        continue

                    # Call attentiveness logic
                    attention_status = predict_attentiveness(face_roi)

                    # Draw rectangle and attention status
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, attention_status, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Record data if recording is ON
                    if recording:
                        timestamps.append(datetime.datetime.now().strftime("%H:%M:%S"))
                        attentiveness_levels.append(attention_status)

        # ✅ Always stream the frame
        encoded_frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame + b'\r\n')



# Start recording button handler
@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording, detect
    recording = True
    detect = True
    return redirect(url_for('index'))

# Stop recording button handler
@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording, timestamps, attentiveness_levels, detect

    # Stop recording and detection
    recording = False
    detect = False

    # Save data to CSV if available
    if timestamps and attentiveness_levels:
        with open("attentiveness_report.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Timestamp", "Attentiveness"])
            for timestamp, attentiveness in zip(timestamps, attentiveness_levels):
                writer.writerow([timestamp, attentiveness])

    # Clear the lists for the next session
    timestamps = []
    attentiveness_levels = []

    # ✅ Keep the camera alive, release only when app shuts down
    # cap.release()  # Do NOT release camera here

    return redirect(url_for('index'))

@app.route('/download_csv')
def download_csv():
    try:
        return Response(
            open("attentiveness_report.csv", "rb"),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment;filename=attentiveness_report.csv"}
        )
    except FileNotFoundError:
        return "No CSV report found. Record first!"

# @app.route('/download_report_graph')
# def download_report_graph():
#     try:
#         df = pd.read_csv('attentiveness_report.csv')
#         timestamps = df['Timestamp'].tolist()
#         attentiveness_levels = df['Attentiveness'].tolist()
#     except Exception as e:
#         print("Error reading CSV:", e)
#         return "No data available for graph generation.", 404

#     image_bytes = plot_attentiveness_csv(timestamps, attentiveness_levels)

#     response = Response(image_bytes, mimetype='image/png')
#     response.headers['Content-Disposition'] = 'attachment; filename=attentiveness_report.png'
#     return response


@app.route('/view_report')
def view_report():
    try:
        with open("attentiveness_report.csv", "r") as file:
            data = file.readlines()
    except FileNotFoundError:
        return "No report found. Record first!"

    # Simple HTML table for visualization
    html = "<h2>Attentiveness Report</h2><table border='1'><tr><th>Timestamp</th><th>Status</th></tr>"
    for line in data[1:]:  # Skip the header
        timestamp, status = line.strip().split(",")
        html += f"<tr><td>{timestamp}</td><td>{status}</td></tr>"
    html += "</table>"

    return html

@app.route('/download_line_graph')
def download_line_graph():
    from utils import generate_attentiveness_graph  # Import here if utils is modular
    graph_buffer = generate_attentiveness_graph()
    if graph_buffer:
        return Response(
            graph_buffer,
            mimetype='image/png',
            headers={'Content-Disposition': 'attachment; filename=attentiveness_graph.png'}
        )
    else:
        return "Error generating the graph. Please record some data first.", 500


# Route for video streaming
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if os.path.exists('attentiveness_report.csv'):
    os.remove('attentiveness_report.csv')


if __name__ == "__main__":
    app.run(debug=True)
