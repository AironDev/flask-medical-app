# Flask Medical App

This Flask application allows users to upload MRI scans, perform inference using a deep learning model, and visualize the results.

## Getting Started

### Prerequisites

- Docker
- Python 3.x

### Setup

1. Build the Docker image:

    ```bash
    docker build -t flask-medical-app .
    ```

2. Run the Docker container:

    ```bash
    docker run -p 5000:5000 flask-medical-app
    ```

3. Open your web browser and go to `http://localhost:5000` to access the application.
# flask-medical-app
