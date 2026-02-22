# AI-Based Face Recognition Attendance System

## Features
- Real-time face recognition using webcam
- Roll No + Name identification
- Confidence % display
- Attendance logging in CSV
- Prevents duplicate attendance for same day
- Saves unknown face snapshots
- GUI Start/Stop buttons

## Folder Structure
- ImagesAttendance/  -> Known faces (ROLLNO_NAME.jpg)
- Unknown/           -> Unknown snapshots
- main.py            -> Main file
- Attendance.csv     -> Output attendance

## Run
pip install -r requirements.txt
python main.py