# SmartRecon GST

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v2.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A sophisticated Flask-based web application that automates Goods and Services Tax (GST) related tasks using AI and Machine Learning. The system provides intelligent invoice OCR, automated reconciliation between invoices, and efficient GST rate lookup functionality.

## 🎯 Overview

This application streamlines GST compliance workflows by:
- Automatically extracting invoice data using advanced OCR
- Comparing and reconciling invoices with semantic matching
- Providing accurate GST rate lookups from official databases
- Generating comprehensive compliance reports

## ⚙️ Tech Stack:

### Frontend:
![alt text](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![alt text](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)


### Backend:
![alt text](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![alt text](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)

### AI & Machine Learning:
![alt text](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![alt text](https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![alt text](https://img.shields.io/badge/FuzzyWuzzy-787878?style=for-the-badge&logoColor=white)<br />

### Data Processing & Document Handling:
![alt text](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![alt text](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![alt text](https://img.shields.io/badge/Pillow-007396?style=for-the-badge&logoColor=white)
![alt text](https://img.shields.io/badge/PDFPlumber-CC0000?style=for-the-badge&logoColor=white)
![alt text](https://img.shields.io/badge/PDF2Image-D2B48C?style=for-the-badge&logoColor=black)

### OCR Engine Interface:

![alt text](https://img.shields.io/badge/TesseractOCR-5F2D8F?style=for-the-badge&logo=tesseract&logoColor=white)<br />

## ✨ Key Features

### 🔍 Automated Invoice OCR
- Extracts key information from invoice images (PNG, JPG, JPEG) and PDFs
- Utilizes Document Question Answering (DocVQA) models with Tesseract OCR fallback
- Captures: invoice numbers, dates, supplier/buyer details, GSTINs, line items, totals

### 🤝 Intelligent Invoice Reconciliation
- **Header Comparison**: Validates invoice numbers, dates, names, and GSTINs
- **Amount Verification**: Cross-checks subtotals, GST amounts, and grand totals
- **Line Item Matching**: Uses SBERT (Sentence-BERT) for semantic description matching
- **Detailed Reporting**: Provides confidence scores and mismatch highlights

### 📊 GST Rate Finder
- Searches comprehensive database built from official HSN codes and notifications
- Employs SBERT and FuzzyWuzzy for robust product matching
- Uses Question-Answering models for contextual rate extraction
- Supports both product descriptions and HSN code queries

### ✅ GST Authenticity Verification
- Cross-references line items with official GST database
- Validates applied GST rates against expected rates
- Aids in compliance checking and audit preparation

### 🌐 User-Friendly Interface
- Clean, responsive web interface built with Flask
- Intuitive file upload and query workflows
- Comprehensive HTML reporting with visual indicators

## 🛠️ Technology Stack

### Backend & Core
- **Python 3.8+** - Core programming language
- **Flask 2.0+** - Web framework
- **Werkzeug** - WSGI utility library

### AI & Machine Learning
- **PyTorch** - Deep learning framework
- **Transformers (Hugging Face)** - DocVQA and QA pipelines
- **Sentence-Transformers** - SBERT models for semantic matching
- **FuzzyWuzzy** - Text similarity matching

### Data Processing
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **PDFPlumber** - PDF text extraction
- **Pillow (PIL)** - Image processing

### OCR & Document Processing
- **PyTesseract** - Tesseract OCR wrapper
- **PDF2Image** - PDF to image conversion

## 🔧 Prerequisites

### System Requirements
- **Python**: Version 3.8 or higher
- **pip**: Python package installer
- **RAM**: Minimum 4GB (8GB+ recommended for ML models)
- **Storage**: At least 2GB free space for model downloads

### External Dependencies

#### 1. Tesseract OCR Engine
Essential for OCR fallback and PDF processing.

**Installation:**
- **Windows**: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
- **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
- **macOS**: `brew install tesseract`

**Configuration:(Important)**
```python
# If tesseract is not in PATH, configure in app.py:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
```

#### 2. Poppler Utilities
Required for PDF to image conversion.

**Installation:**
- **Ubuntu/Debian**: `sudo apt-get install poppler-utils`
- **macOS**: `brew install poppler`
- **Windows**: Download Poppler from the [GitHub repository](https://github.com/oschwartz10612/poppler-windows), and add it to your PATH.


#### 3. ML Models (Auto-downloaded)
The following models will be downloaded automatically on first run:
- `impira/layoutlm-document-qa` (~1.2GB)
- `all-mpnet-base-v2` (~420MB)
- `all-MiniLM-L6-v2` (~90MB)
- `deepset/roberta-base-squad2` (~1.4GB)

## 📥 Installation

### 1. Clone Repository
```bash
git clone https://github.com/wardayX/SmartRecon-GST
cd SmartRecon-GST
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Data Files
Place these files in the project root directory:

#### Required Files:
- **`HSN-Codes-for-GST-Enrolment.pdf`**: Official HSN codes document
- **`cbic_gst_goods_rates_exact.csv`**: GST rates database (see generation guide below)

## 📊 Data Setup Guide

### Generating `cbic_gst_goods_rates_exact.csv`

This file contains the official GST rates database and is crucial for accurate rate lookups.

#### Option 1: Direct Download (Recommended)
1. Visit [CBIC GST Portal](https://cbic-gst.gov.in/gst-goods-services-rates.html)
2. Parse the HTML with the help of BeautifulSoup4.
3. Note the CGST,SGST/UTGST,IGST,Description/Name,HS Code.
4. Save as CSV with filename: `cbic_gst_goods_rates_exact.csv`

#### Option 2: Web Scraping (Advanced)
If data is only available in HTML format:

```python
import requests
import pandas as pd
from bs4 import BeautifulSoup

# Example scraping script (customize based on site structure)
def scrape_gst_rates():
    url = "https://cbic-gst.gov.in/gst-goods-services-rates.html"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract table data (customize selectors)
    tables = soup.find_all('table')
    df = pd.read_html(str(tables[0]))[0]  # Adjust index as needed
    df.to_csv('cbic_gst_goods_rates_exact.csv', index=False)

scrape_gst_rates()
```

#### Expected CSV Format:
```csv
Chapter/ Heading/ Sub-heading/ Tariff item,Description of Goods,CGST Rate (%),SGST Rate (%),IGST Rate (%)
0101,Live horses, asses, mules and hinnies,0,0,0
0102,Live bovine animals,0,0,0
...
```

### HSN Codes PDF
Source the official "HSN-Codes-for-GST-Enrolment.pdf" from:
- CBIC official website
- GST portal downloads section
- Tax practitioner resources

## 🚀 Running the Application

### Development Mode
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run the application
python app.py
```

### Production Deployment
```bash
# Using Gunicorn (recommended for production)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app

# Using uWSGI
pip install uwsgi
uwsgi --http :8000 --module app:app --processes 4
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

## 📁 Project Structure

```
gst-invoice-reconciliation/
├── app.py                                  # Main Flask application
├── requirements.txt                        # Python dependencies
├── HSN-Codes-for-GST-Enrolment.pdf         # HSN codes database
├── cbic_gst_goods_rates_exact.csv          # GST rates database
├── templates/                              # HTML templates
│   ├── index.html                          # Home page
│   ├── reconcile.html                      # Invoice upload page
│   ├── find_gst_rate.html                  # GST rate finder
│   └── reconciliation_report_display.html  # Report display
├── notebooks/                              # Models
│   ├── GST_goods_rates_scrapper.ipynb      # Scrapping GST Rates
│   └── productname_to_gst_model.ipynb      # GST Rate Finder Model
├── tests/                                  # Some Tests
│   ├── TheRealG.ipynb                      # Parsing Model
│   ├── theunrealg (1).py                   # Parsing model testing in flask
│   └── theunrealg.py                       # Earlier Testing of parsing model in flask
├── uploads/                                # Temporary file storage
└── README.md                               # This file
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root:

```env
# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here

# File Paths
UPLOAD_FOLDER=uploads
HSN_PDF_PATH=HSN-Codes-for-GST-Enrolment.pdf
GST_CSV_PATH=cbic_gst_goods_rates_exact.csv

# Model Configuration
DOC_QA_MODEL_NAME=impira/layoutlm-document-qa
SENTENCE_MODEL_GST_FINDER=all-mpnet-base-v2
SENTENCE_MODEL_ITEM_MATCHER=all-MiniLM-L6-v2
QA_MODEL_GST_FINDER=deepset/roberta-base-squad2

# OCR Configuration
TESSERACT_PATH=/usr/bin/tesseract
```

### Application Settings
Key configurations in `app.py`:

```python
# File Upload Settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Processing Settings
CONFIDENCE_THRESHOLD = 0.7
MAX_PROCESSING_TIME = 300  # 5 minutes timeout
```

## 📊 Usage Guide

### 1. Invoice Reconciliation

#### Step 1: Upload Invoices
- Navigate to the "Reconcile Invoices" page
- Upload two invoice files (images or PDFs)
- Click "Reconcile Invoices"

#### Step 2: Review Results
The system will provide:
- **Header Comparison**: Invoice numbers, dates, parties, GSTINs
- **Amount Reconciliation**: Line-by-line amount matching
- **Line Item Analysis**: Semantic matching of product descriptions
- **GST Authenticity Check**: Verification of applied GST rates

#### Step 3: Download Report
- View detailed HTML report
- Export results for record-keeping
- Share findings with stakeholders

### 2. GST Rate Lookup

#### Query Methods:
- **Product Description**: Enter product name or description
- **HSN Code**: Enter specific HSN/SAC code
- **Mixed Query**: Combine description with HSN code

#### Example Queries:
```
"Mobile phone accessories"
"HSN 8517"
"Bluetooth headphones HSN 8518"
"Construction materials cement"
```

### 3. API Integration (Future)

```python
import requests

# GST Rate Lookup API
response = requests.post('http://localhost:5000/api/gst-rate', 
                        json={'query': 'laptop computers'})
gst_rates = response.json()

# Invoice Reconciliation API
files = {
    'invoice1': open('invoice1.pdf', 'rb'),
    'invoice2': open('invoice2.pdf', 'rb')
}
response = requests.post('http://localhost:5000/api/reconcile', files=files)
reconciliation_result = response.json()
```

## 🔍 Core Components

### GSTReconciliationEngine Class

#### Key Methods:
- **`extract_invoice_data()`**: DocVQA-based OCR processing
- **`compare_invoices()`**: Main reconciliation orchestrator
- **`_match_line_items_sbert()`**: Semantic line item matching
- **`verify_item_gst_rates()`**: GST rate authenticity checking
- **`generate_reconciliation_report()`**: HTML report generation

#### Processing Flow:
1. **OCR Extraction**: Extract structured data from invoice images
2. **Data Standardization**: Normalize extracted information
3. **Header Comparison**: Validate basic invoice details
4. **Amount Reconciliation**: Compare financial totals
5. **Line Item Matching**: Semantic matching of products/services
6. **GST Verification**: Check rate authenticity against database
7. **Report Generation**: Create comprehensive comparison report

### Data Processing Pipeline

#### HSN Data Processing:
```python
def parse_hsn_pdf_gst_finder(pdf_path):
    """Extract HSN codes and descriptions from official PDF"""
    # PDF text extraction
    # Data cleaning and structuring
    # Return structured DataFrame
```

#### GST Rate Processing:
```python
def parse_gst_csv_gst_finder(csv_path):
    """Process official GST rates CSV"""
    # CSV parsing with error handling
    # Column standardization
    # Rate calculation and validation
```

## 🧪 Testing

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test categories
pytest tests/test_ocr.py
pytest tests/test_reconciliation.py
pytest tests/test_gst_finder.py
```

### Test Structure
```
tests/
├── test_ocr.py                 # OCR functionality tests
├── test_reconciliation.py      # Invoice comparison tests
├── test_gst_finder.py          # GST rate lookup tests
├── test_data_processing.py     # Data pipeline tests
├── fixtures/                   # Test data files
│   ├── sample_invoices/
│   └── test_gst_data.csv
└── conftest.py                 # Test configuration
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Tesseract Not Found
**Error**: `TesseractNotFoundError`
**Solution**:
```python
# Add to app.py
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

#### 2. PDF2Image Issues
**Error**: `pdf2image.exceptions.PDFInfoNotInstalledError`
**Solution**: Install Poppler utilities (see Prerequisites section)

#### 3. Model Download Timeouts
**Error**: `HTTPSConnectionPool` timeout
**Solution**:
```bash
# Pre-download models
python -c "from transformers import pipeline; pipeline('document-question-answering', model='impira/layoutlm-document-qa')"
```

#### 4. Memory Issues
**Error**: `RuntimeError: CUDA out of memory`
**Solution**:
```python
# Use CPU instead of GPU
import torch
device = torch.device('cpu')
```

### Performance Optimization

#### 1. Model Caching
```python
# Enable model caching
import os
os.environ['TRANSFORMERS_CACHE'] = './model_cache'
```

#### 2. Batch Processing
```python
# Process multiple invoices efficiently
def batch_process_invoices(invoice_list, batch_size=4):
    # Implementation for batch OCR processing
    pass
```

#### 3. Database Optimization
```python
# Index GST database for faster lookups
df_gst.set_index(['HSN_Code', 'Description'], inplace=True)
```

## 🔒 Security Considerations

### File Upload Security
- Validate file types and sizes
- Sanitize uploaded filenames
- Use temporary storage with cleanup
- Implement virus scanning for production

### Data Privacy
- Encrypt sensitive invoice data
- Implement secure file deletion
- Add user authentication for production
- Comply with data protection regulations

### Production Security
```python
# Security headers
from flask_talisman import Talisman
Talisman(app, force_https=True)

# Input validation
from werkzeug.utils import secure_filename
filename = secure_filename(uploaded_file.filename)
```

## 🔮 Future Enhancements

### Phase 1: Core Improvements
- [ ] Enhanced OCR for handwritten invoices
- [ ] Multi-language support (Hindi, regional languages)
- [ ] Batch processing capabilities
- [ ] Advanced error handling and recovery

### Phase 2: Integration Features
- [ ] GSTR-2A/2B integration
- [ ] E-way bill validation
- [ ] ERP system connectors
- [ ] Email invoice processing

### Phase 3: Advanced Analytics
- [ ] Machine learning for anomaly detection
- [ ] Predictive GST compliance scoring
- [ ] Historical trend analysis
- [ ] Custom reporting dashboards

### Phase 4: Enterprise Features
- [ ] Multi-tenant architecture
- [ ] Role-based access control
- [ ] API rate limiting
- [ ] Audit trail logging

## 🤝 Contributing

### Development Setup
```bash
# Fork the repository
git clone https://github.com/wardayX/SmartRecon-GST

# Install development dependencies
pip install -r requirements.txt

# Run pre-commit hooks
pre-commit install
```

### Code Standards
- Follow PEP 8 style guide
- Add docstrings to all functions
- Write unit tests for new features
- Update documentation for changes

### Submission Process
1. Create feature branch
2. Write tests for new functionality
3. Ensure all tests pass
4. Update documentation
5. Submit pull request with detailed description

## 📄 Requirements File

```txt
Flask>=2.0
Werkzeug>=2.0
Pillow>=9.0
pandas>=1.3
pdfplumber>=0.7
fuzzywuzzy>=0.18
python-Levenshtein>=0.12 # Often improves fuzzywuzzy speed
torch>=1.10 # Or a version compatible with your CUDA if using GPU
numpy>=1.20
sentence-transformers>=2.2
transformers>=4.15
pytesseract>=0.3.8
pdf2image>=1.16
accelerate>=0.12 # Often helpful for transformers
```

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Contact

Hrishikesh Nath - [@wardayX](https://github.com/wardayX) - nathh722@gmail.com<br />
Rituraj Adhikary - [@Riceguy007](https://github.com/Riceguy007) - riturajadhikay99@gmail.com<br />
Subhadeep Deb - [@coderSubhadeepdeb](https://github.com/coderSubhadeepdeb) - sbhdpdeb@gmail.com<br />
Nishanta Kamal Baruah - [@Nishanta-13](https://github.com/Nishanta-13) - nishantapro@gmail.com<br />


### Project Link: [https://github.com/wardayX/SmartRecon-GST](https://github.com/wardayX/SmartRecon-GST)
