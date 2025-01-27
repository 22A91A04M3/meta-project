import cv2
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

# Initialize camera with specified API
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def detect_potholes(frame):
    """
    Detect potholes in the frame.
    
    Parameters:
    frame (numpy array): Frame from the webcam.
    
    Returns:
    output (numpy array): Frame with potholes highlighted.
    """
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges using Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output = frame.copy()
    
    pothole_detected = False
    
    # Iterate through contours
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter contours based on area and aspect ratio
        if area > 100 and w/h > 1.5:
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
            pothole_detected = True
    
    # Display "Pothole Detected" text
    if pothole_detected:
        cv2.putText(output, "Pothole Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return output

def gen_frames():
    """Generate frame by frame from camera"""
    while True:
        try:
            success, frame = camera.read()  
            if not success:
                break
            else:
                output = detect_potholes(frame)
                ret, buffer = cv2.imencode('.jpg', output)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error: {str(e)}")

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

if __name__ == '__main__':
        app.run(debug=True)