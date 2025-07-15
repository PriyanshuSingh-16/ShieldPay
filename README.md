
# Safe Pay - QR Code Security Analysis

A comprehensive web application that uses machine learning to detect potentially malicious QR codes and protect users from security threats. Shield Pay combines advanced ML analysis with an intuitive interface to provide real-time QR code security assessment.

## Features

- **AI-Powered QR Analysis**: Machine learning model to classify QR codes as safe, suspicious, or malicious
- **User Dashboard**: Track your QR code scan history and analysis results
- **Admin Panel**: Administrative interface for reviewing reported QR codes
- **Real-time Analysis**: Instant security assessment of uploaded QR codes
- **Secure Authentication**: JWT-based user authentication system
- **File Management**: Persistent storage of QR code images for analysis

## Technology Stack

- **Backend**: Node.js, Express.js
- **Database**: MongoDB with Mongoose ODM
- **Frontend**: React (built version included)
- **Machine Learning**: Python-based ML inference engine
- **Authentication**: JWT with bcrypt password hashing
- **File Upload**: Multer for handling QR code image uploads

## Installation

1. **Clone the repository** 
2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Environment Setup**:
   Create a `.env` file with the following variables:
   ```
   MONGO_URI=your_mongodb_connection_string
   JWT_SECRET=your_jwt_secret_key
   NODE_ENV=development
   PORT=3000
   ```

4. **Python Dependencies** (for ML inference):
   The Python ML model requires specific packages for image processing and prediction.

## Usage

### Starting the Application

```bash
npm run dev
```

The application will be available at `http://localhost:3000`

### Default Admin Account

- **Username**: `admin`
- **Password**: `admin123`

⚠️ **Important**: Change the default admin credentials in production!

### API Endpoints

#### Authentication
- `POST /api/register` - Register new user
- `POST /api/login` - User login
- `GET /api/logout` - User logout

#### QR Code Analysis
- `POST /api/upload-qr` - Upload and analyze QR code (authenticated users)
- `GET /api/dashboard` - Get user's scan history
- `PUT /api/report-qr/:id` - Report a QR code as malicious

#### Admin Routes
- `GET /api/admin/dashboard` - View all reported QR codes
- `POST /api/admin/verify-qr/:id` - Admin verification of reported QR codes
- `DELETE /api/admin/remove-qr/:id` - Remove QR code record

### File Structure

```
├── client/build/          # React frontend (production build)
├── ml_inference/          # Python ML model and inference scripts
├── models/               # MongoDB/Mongoose data models
├── public/              # Static HTML pages
├── uploaded_qr_codes/   # Persistent storage for QR images
├── server.js           # Main Express server
├── authMiddleware.js   # JWT authentication middleware
└── package.json       # Node.js dependencies
```

### Machine Learning Model

The ML inference engine is located in the `ml_inference/` directory:
- `predict_model.py` - Main prediction script
- `ml_model.py` - ML model definition
- `generate.py` - Data generation utilities

## Security Features

- **JWT Authentication**: Secure token-based authentication
- **Password Hashing**: bcrypt for secure password storage
- **File Validation**: Image upload validation and security
- **Admin Protection**: Separate admin authentication middleware
- **CORS Configuration**: Cross-origin resource sharing setup

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Development Notes

- The application uses MongoDB for data persistence
- QR code images are stored permanently in `uploaded_qr_codes/`
- Python ML model is called via child process spawning
- Frontend is a pre-built React application
- Admin users have elevated privileges for QR code verification

## License

MIT License - see LICENSE file for details

## Author

Priyanshu Singh

## Support

For issues and questions, please open an issue in the repository or contact the development team.
