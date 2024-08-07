# Car and Person Detection and Counting

This project aims to detect and count cars and people in a video using a trained YOLOv8 model. It leverages the use of YOLOv8 for object detection and provides functionalities to count cars and people, detect car colors, and determine the gender of detected people. The project includes a command-line interface for video input and mode selection (view or save).

## Features

- **Car and Person Detection**: Utilizes a pre-trained YOLOv8 model to detect cars and people in a video.
- **Car Color Detection**: Detects the dominant color of cars using HSV color space.
- **Person Gender Detection**: Determines the gender of detected people using a trained gender classification model. The model was trained in the python notebook get_person_gender.ipynb
- **Counting**: Counts the number of cars and people crossing a defined line in the video.
- **Command-Line Interface**: Allows users to specify the video file path and mode (view or save) via command-line arguments.

## Installation

To run this code, you need to have Python installed on your system. The project has been tested on Python 3.12. Follow these steps to set up the project:

1. **Clone the repository**:
   ```sh
   git clone https://github.com/JoyBis48/Car_Person_Detection.git
   ```
2. **Navigate to the project directory**:
   ```sh
   cd Car_Person_Detection
   ```
3. **Install the required dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

To start the application, run the `main.py` script from the terminal. You can specify the video file path and mode (view or save) using command-line arguments.

### Command-Line Arguments

- `--video`: Path to the video file (default: `video.mp4`).
- `--mode`: Mode to run the script in `view`or `save`.

### Example

To view the video with detections:
```sh
python main.py --video path/to/video.mp4 --mode view
```

To save the processed video:
```sh
python main.py --video path/to/video.mp4 --mode save
```

## How It Works

### Model Initialization

The application uses a YOLOv8 model for object detection and a separate model for gender classification. The models are initialized in the **main.py** script.

### Frame Processing

Each frame of the video is processed to detect cars and people. The detected cars are further analyzed to determine their color, and the detected people are analyzed to determine their gender.

### Counting

The application counts the number of cars and people crossing a defined line in the video. The counts are displayed on the video frames.

### Saving or Viewing

Depending on the mode specified 'view' or 'save', the application either displays the video with detections in a window or saves the processed video to a file. 

## Dependencies

- OpenCV
- Numpy
- PIL
- Ultralytics YOLO
- argparse

## File Structure

- **main.py**: Main script to run the application.
- **utilities.py**: Contains utility functions for car color detection and person gender detection.
- **main.ipynb**: Jupyter notebook for initial testing and development.
- **get_person_gender.ipynb**: Jupyter notebook for training the gender classification model.
- **requirements.txt**: List of required dependencies.