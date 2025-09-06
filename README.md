# Hazard Spotter AI

A full-stack AI-powered application for detecting and classifying hazards in images using deep learning. The application consists of a React frontend and a Python FastAPI backend that uses YOLOv8 for object detection.

## Project Structure

```
hazard-spotter-ai/
├── frontend/          # React + TypeScript + Tailwind CSS frontend
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── ...
├── backend/           # Python FastAPI backend with YOLO model
│   ├── main.py
│   ├── model_manager.py
│   ├── requirements.txt
│   └── ...
└── README.md
```

## Features

### Frontend
- **Modern React UI**: Built with React, TypeScript, and Tailwind CSS
- **Responsive Design**: Works on desktop and mobile devices
- **Image Upload**: Drag-and-drop or click-to-upload interface
- **Real-time Results**: Live display of detection results
- **Visual Feedback**: Color-coded hazard classifications

### Backend
- **YOLO Object Detection**: Uses YOLOv8 for accurate object detection
- **Hazard Classification**: Categorizes objects as "Healthy" or "Hazardous"
- **REST API**: FastAPI-based API with automatic documentation
- **File Upload Support**: Handles both file uploads and base64 images
- **Model Management**: Automatic dataset download and model training

## Quick Start

### Prerequisites
- Node.js 16+ and npm
- Python 3.8+
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Venkatyarramsetti/hazard-spotter-ai.git
cd hazard-spotter-ai
```

### 2. Setup Backend
```bash
cd backend

# Windows
start_server.bat

# Linux/Mac
chmod +x start_server.sh
./start_server.sh
```

The backend will:
- Create a virtual environment
- Install Python dependencies
- Download the TACO dataset (first run only)
- Train the YOLO model (first run only)
- Start the API server at http://localhost:8000

### 3. Setup Frontend
```bash
cd frontend
npm install
npm run dev
```

The frontend will start at http://localhost:8081 (or another port if 8081 is busy).

### 4. Access the Application
- Open your browser and go to http://localhost:8081
- Click "Start Detection" to upload and analyze images
- The frontend will communicate with the backend API for hazard detection

## API Documentation

Once the backend is running, visit http://localhost:8000/docs for interactive API documentation.

### Key Endpoints:
- `GET /health` - Check API health
- `POST /detect` - Upload image for hazard detection
- `GET /model-info` - Get model information

## How It Works

1. **Image Upload**: User uploads an image through the frontend interface
2. **API Call**: Frontend sends the image to the backend API
3. **Object Detection**: Backend uses YOLOv8 to detect objects in the image
4. **Classification**: Detected objects are classified as "Healthy" or "Hazardous"
5. **Results Display**: Frontend displays the results with visual indicators

## Hazard Categories

The system classifies detected objects into:

**Hazardous Items:**
- Broken glass
- Cigarettes
- Plastic waste
- Aluminum foil
- Bottle caps
- And other potentially harmful items

**Healthy Items:**
- Safe, non-hazardous objects

## Development

### Frontend Development
```bash
cd frontend
npm run dev          # Start development server
npm run build        # Build for production
npm run preview      # Preview production build
```

### Backend Development
```bash
cd backend
# Activate virtual environment
source hazard_env/bin/activate  # Linux/Mac
# or
hazard_env\Scripts\activate     # Windows

# Start with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Troubleshooting

### Common Issues:

1. **Port Already in Use**: If ports 8000 or 8081 are busy, the applications will automatically try different ports.

2. **Model Download Issues**: The first run downloads a large dataset. Ensure stable internet connection.

3. **CORS Issues**: The backend is configured to allow requests from common development ports.

4. **Memory Issues**: Model training requires significant RAM. Consider reducing epochs or using a smaller model variant.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the object detection model
- [TACO Dataset](https://github.com/pedropro/TACO) for garbage detection training data
- [Shadcn/ui](https://ui.shadcn.com/) for the UI components
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
