# Hazard Spotter AI 🚨

A full-stack AI-powered application for detecting and classifying hazards in images using deep learning. The application uses YOLOv8 for real-time object detection and hazard assessment.

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/yourusername/hazard-spotter-ai)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

## 🌟 Features

- **AI-Powered Detection**: YOLOv8 model for accurate object detection
- **Smart Classification**: Automatically categorizes objects as Safe or Hazardous
- **Real-time Analysis**: Instant hazard assessment with confidence scores
- **User Authentication**: Secure JWT-based authentication system
- **Modern UI**: Responsive React frontend with Tailwind CSS
- **RESTful API**: FastAPI backend with automatic documentation
- **MongoDB Integration**: Persistent data storage for user accounts

## 🏗️ Project Structure

```
hazard-spotter-ai/
├── frontend/          # React + TypeScript + Vite
│   ├── src/
│   │   ├── components/    # Reusable UI components
│   │   ├── pages/         # Page components
│   │   └── lib/           # Utilities
│   └── package.json
├── backend/           # Python FastAPI backend
│   ├── main.py            # API routes
│   ├── model_manager.py   # YOLO model handling
│   ├── auth.py            # Authentication logic
│   ├── database.py        # MongoDB connection
│   └── requirements.txt
└── DEPLOYMENT.md      # Detailed deployment guide
```

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Python 3.12+
- MongoDB (local or Atlas)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/hazard-spotter-ai.git
cd hazard-spotter-ai
```

### 2. Setup Backend
```bash
cd backend

# Create virtual environment
python -m venv hazard_env

# Activate virtual environment
# Windows:
hazard_env\Scripts\activate
# Linux/Mac:
source hazard_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env and add your MongoDB URL and SECRET_KEY

# Start server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend runs at: `http://localhost:8000`

### 3. Setup Frontend
```bash
cd frontend

# Install dependencies
npm install

# Create environment file
cp .env.development .env

# Start development server
npm run dev
```

Frontend runs at: `http://localhost:8080`

## 🔧 Configuration

### Backend (.env)
```env
MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/hazard_db
SECRET_KEY=your-secret-key-min-32-characters
CORS_ORIGINS=http://localhost:8080
```

### Frontend (.env)
```env
VITE_BACKEND_URL=http://localhost:8000
```

## 📚 API Documentation

Interactive API docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Key Endpoints

#### Authentication
- `POST /auth/register` - Register new user
- `POST /auth/login` - Login user

#### Detection
- `POST /detect` - Upload image for hazard detection
- `GET /health` - Health check
- `GET /model-info` - Get model information

## 🎯 How It Works

1. **Upload Image**: User uploads an image via the web interface
2. **Preprocessing**: Image is preprocessed and enhanced
3. **Detection**: YOLOv8 model detects objects in the image
4. **Classification**: Objects are classified as Safe or Hazardous based on:
   - Object type (food, vehicles, hazards, etc.)
   - Confidence scores
   - Context analysis
5. **Results**: Annotated image with bounding boxes and risk assessment

## 🏷️ Classification Categories

### Safe Items ✅
- Foods (fruits, vegetables, meals)
- People and animals
- Vehicles
- Furniture and household items
- Nature elements

### Hazardous Items ⚠️
- **Critical**: Fire, smoke, gas leaks
- **High**: Chemicals, sharp objects, broken glass
- **Medium**: General waste, trash
- **Low**: Minor debris

## 🚀 Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

### Quick Deploy

**Frontend (Vercel)**
```bash
cd frontend
vercel --prod
```

**Backend (Render)**
1. Push to GitHub
2. Connect repository to Render
3. Set environment variables
4. Deploy

## 🛠️ Development

### Frontend Stack
- React 18 + TypeScript
- Vite (build tool)
- Tailwind CSS + shadcn/ui
- React Router

### Backend Stack
- FastAPI (Python web framework)
- YOLOv8 (object detection)
- PyTorch (ML framework)
- MongoDB (database)
- JWT authentication

### Run Tests
```bash
# Backend
cd backend
pytest

# Frontend
cd frontend
npm test
```

## 📊 Performance

- **Detection Speed**: ~2-5 seconds per image
- **Accuracy**: 70-90% depending on image quality
- **Supported Formats**: JPG, PNG, WEBP
- **Max Image Size**: 10MB

## 🐛 Troubleshooting

### Backend Issues
- **Port in use**: Change port in command or kill process
- **Model download slow**: First run downloads ~6MB model
- **MongoDB connection**: Check connection string and network

### Frontend Issues
- **API connection**: Verify VITE_BACKEND_URL matches backend
- **CORS errors**: Ensure backend CORS_ORIGINS includes frontend URL
- **Build errors**: Delete node_modules and reinstall

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- shadcn/ui component library
- FastAPI framework
