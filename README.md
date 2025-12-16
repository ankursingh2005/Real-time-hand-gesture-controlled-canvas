---------------------------------Real-Time Hand Gesture–Controlled Canvas---------------------------------


--Overview--

Real-Time Hand Gesture–Controlled Canvas is a computer vision–based interactive drawing application that enables users to draw, erase, and select colors in mid-air using hand gestures captured through a webcam. The system eliminates the need for traditional input devices such as a mouse or stylus and demonstrates the practical application of gesture recognition and human–computer interaction (HCI).

The project leverages MediaPipe Hands for real-time hand landmark detection and OpenCV for image processing and rendering, delivering a smooth and responsive drawing experience.

------------------------------------Project Objectives--------------------------------------------------

To design a touch-free drawing system using hand gestures

To understand and implement real-time hand tracking

To demonstrate the integration of computer vision with interactive applications

To build an intuitive and modern UI for gesture-based interaction


---------------------------------------Key Features------------------------------------------------------

Real-time hand tracking using a webcam

Draw in the air using the index finger

Color selection using index + middle finger gesture

Transparent modern color palette UI

Eraser functionality using gesture-based selection

Dynamic brush and eraser thickness

FPS (Frames Per Second) counter for performance monitoring

Save canvas as an image

Clear canvas and exit controls via keyboard





------------------------------------------- Hand Gesture Controls---------------------------------------------
Gesture	Function
Index finger up	Draw on canvas
Index + Middle finger up	Color selection mode
Select black color	Eraser
No fingers up	Idle mode




⌨️ Keyboard Controls
Key	Action
C	Clear canvas
S	Save drawing
Q	Quit application






🧰 Technologies Used

Python 3

OpenCV

MediaPipe

NumPy

Webcam (Real-Time Video Input)






🗂️ Project Structure
Real-Time-Hand-Gesture-Controlled-Canvas/
│── main.py
│── README.md
│── requirements.txt



📦 requirements.txt
opencv-python
mediapipe
numpy






🖼️ Output

Live webcam feed with hand landmarks

Transparent color palette

Real-time drawing on virtual canvas

Option to save artwork as PNG image







🚀 Applications

Touchless drawing systems

Interactive whiteboards

Smart classrooms

AR/VR interfaces

Assistive technology for differently-abled users






🔮 Future Enhancements

Virtual transparent keyboard for text and sentence writing

Gesture-based undo/redo

Multi-hand support

Voice command integration

Shape recognition (circle, rectangle, line)

Cloud-based canvas saving







👨‍💻 Author

Ankur Singh
B.Tech (CSE – Artificial Intelligence)
Bansal Institute of Engineering and Technology, Lucknow




📜 License

This project is developed for academic and educational purposes.
