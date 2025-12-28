# Instagram Campaign Analyzer

A FastAPI application that analyzes Instagram campaign screenshots and videos to extract engagement metrics using AI-powered OCR.

## Features

- **Multi-format Upload**: Supports videos (MP4, AVI, MOV, MKV), images (JPG, PNG), and ZIP archives
- **Frame Extraction**: Automatically extracts frames from videos for analysis
- **AI Frame Classification**: Uses TensorFlow model to filter good quality frames from bad ones
- **OCR Metrics Extraction**: Extracts Instagram Insights metrics using Google Gemini AI via OpenRouter
- **Multi-language Support**: Handles both English and Persian Instagram interfaces
- **User Authentication**: Session-based login system with admin user support
- **S3 Storage**: Optional AWS S3 integration for storing results
- **Excel Export**: Export campaign results to Excel format
- **Job Tracking**: Track processing status with custom job IDs

## Extracted Metrics

The system can extract the following Instagram Insights metrics:
- Views, Accounts Reached
- Followers / Non-followers ratio
- Interactions, Likes, Replies, Shares
- Link Clicks, Sticker Taps
- Navigation (Forward, Back, Next Story, Exited)
- Profile Activity, Profile Visits, Follows

## Tech Stack

- **Backend**: FastAPI, Python 3.11
- **AI/ML**: TensorFlow, Google Gemini (via OpenRouter)
- **Database**: PostgreSQL
- **Storage**: AWS S3 (optional)
- **Image Processing**: OpenCV, Pillow

## Installation

### Prerequisites

- Python 3.11+
- PostgreSQL database
- OpenRouter API key (for Gemini AI)
- AWS S3 bucket (optional)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/hamed-aghasi/influ-OCR.git
cd influ-OCR
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your configuration:
```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/instagram_analyzer

# API Keys
OPENROUTER_API_KEY=your_openrouter_api_key

# Authentication
SECRET_KEY=your-secret-key
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your-admin-password

# AWS S3 (optional)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_S3_BUCKET=your-bucket-name
AWS_REGION=us-east-1

# Directories
UPLOAD_DIR=/tmp/uploads
PROCESSING_DIR=/tmp/processing
```

4. Place your trained TensorFlow model in:
```
instagram_analyzer_app/models/frame_classifier_savedmodel/
```

5. Run the application:
```bash
cd instagram_analyzer_app
python main.py
```

Or with uvicorn:
```bash
uvicorn instagram_analyzer_app.main:app --host 0.0.0.0 --port 8000
```

## Docker

Build and run with Docker:

```bash
docker build -t instagram-analyzer .
docker run -p 8000:8000 --env-file .env instagram-analyzer
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Upload form (requires auth) |
| `/upload` | POST | Upload campaign files |
| `/status/{job_id}` | GET | View job status |
| `/jobs` | GET | List all jobs |
| `/export` | GET | Export results to Excel |
| `/login` | GET/POST | User login |
| `/logout` | GET | User logout |
| `/health` | GET | Health check |
| `/api/job/{job_id}` | GET | Get job details (JSON) |
| `/api/job/{job_id}/metrics` | GET | Get extracted metrics |

## Usage

1. Log in with your credentials (default: admin/admin123)
2. Fill in campaign details (date, name, product, company)
3. Upload a video, image, or ZIP file containing Instagram screenshots
4. Wait for processing to complete
5. View extracted metrics on the status page
6. Export results to Excel if needed

## Processing Pipeline

1. **Upload**: User uploads campaign media files
2. **Frame Extraction**: Videos are split into individual frames
3. **Classification**: TensorFlow model filters good/bad quality frames
4. **OCR**: Gemini AI extracts metrics from good frames
5. **Aggregation**: Results are aggregated and stored
6. **Storage**: Metrics saved to database and optionally S3

## License

MIT License
