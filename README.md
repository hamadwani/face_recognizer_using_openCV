Face Recognition System using OpenCV
A real-time face detection and recognition system built with OpenCV and LBPH. Supports custom datasets, multiple users, and live webcam recognition. Ideal for beginners learning computer vision and AI-based face identification.



face_recognizer/
├─ dataset/                 # images saved per user (created by capture_faces.py)
│  ├─ User.Name.1/          # e.g. "Hammad.1"
│  │   ├─ 1.jpg
│  │   └─ ...
├─ trainer/                 # model output
│  └─ trainer.yml
├─ capture_faces.py
├─ train_model.py
├─ recognize.py
├─ utils.py
├─ requirements.txt
├─ .gitignore
└─ README.md
