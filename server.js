require('dotenv').config(); // Load environment variables from .env file
const express = require('express');
const mongoose = require('mongoose');
const path = require('path');
const cookieParser = require('cookie-parser');
const multer = require('multer'); // For handling file uploads
const { spawn } = require('child_process'); // To run Python script
const fs = require('fs'); // For file system operations
const { protect, adminProtect } = require('./authMiddleware'); // Our JWT protection middleware
const jwt = require('jsonwebtoken'); // Import JWT for generateToken
const bcrypt = require('bcryptjs'); // For initial admin password hash

// Mongoose Models
const User = require('./models/User');
const QRCodeScan = require('./models/QRCodeScan'); // Assuming models/ is the correct path

const app = express();
const PORT = process.env.PORT || 3000;

// --- Permanent Storage Directory for Uploaded QR Codes ---
const UPLOAD_DIR = path.join(__dirname, 'uploaded_qr_codes');
// Ensure this directory exists
fs.mkdirSync(UPLOAD_DIR, { recursive: true });


// --- Database Connection ---
mongoose.connect(process.env.MONGO_URI)
    .then(() => {
        console.log('MongoDB Connected...');
        // --- Create Admin User on Server Startup if not exists ---
        createAdminUser();
    })
    .catch(err => console.error(err));

// Function to create admin user
async function createAdminUser() {
    const adminUsername = 'admin';
    const adminPasswordPlain = 'admin123'; // Plain text password for initial setup

    try {
        const adminUser = await User.findOne({ username: adminUsername, isAdmin: true });
        if (!adminUser) {
            // Pass the plain text password directly. The UserSchema.pre('save') hook will hash it.
            await User.create({
                username: adminUsername,
                password: adminPasswordPlain, // Corrected: pass plain text password here
                isAdmin: true
            });
            console.log('Admin user "admin" created successfully.');
        } else {
            console.log('Admin user "admin" already exists.');
        }
    } catch (error) {
        console.error('Error creating admin user:', error);
    }
}


// --- Middleware ---
app.use(express.json()); // Body parser for JSON requests
app.use(express.urlencoded({ extended: true })); // Body parser for URL-encoded requests
app.use(cookieParser()); // Cookie parser for JWT in cookies

// Serve static files from the 'public' directory
app.use(express.static(path.join(__dirname, 'public')));
// Serve uploaded QR codes permanently (making them accessible for display later)
app.use('/uploaded_qr_codes', express.static(UPLOAD_DIR));


// Set up Multer for file uploads
const upload = multer({
    dest: 'uploads/', // Temporary directory for incoming uploads
    limits: { fileSize: 5 * 1024 * 1024 } // 5MB file size limit
});

// --- JWT Helper Function (for generating token) ---
const generateToken = (id) => {
    return jwt.sign({ id }, process.env.JWT_SECRET, {
        expiresIn: '1h', // Token expires in 1 hour
    });
};

// --- Routes ---

// @route   POST /api/register
// @desc    Register a new user (admin cannot register via this route)
// @access  Public
app.post('/api/register', async (req, res) => {
    const { username, password } = req.body;

    if (!username || !password) {
        return res.status(400).json({ message: 'Please enter all fields' });
    }

    try {
        const userExists = await User.findOne({ username });
        if (userExists) {
            return res.status(400).json({ message: 'User already exists' });
        }
        // Ensure new users are NOT admins by default
        const user = await User.create({ username, password, isAdmin: false }); 

        // Set JWT as a httpOnly cookie
        const token = generateToken(user._id);
        res.cookie('jwt', token, {
            httpOnly: true,
            secure: process.env.NODE_ENV === 'production', // Use secure cookies in production
            maxAge: 3600000 // 1 hour
        });

        res.status(201).json({
            message: 'User registered successfully',
            user: {
                id: user._id,
                username: user.username,
                isAdmin: user.isAdmin // Include isAdmin in response
            }
        });

    } catch (error) {
        console.error(error);
        res.status(500).json({ message: 'Server error during registration' });
    }
});

// @route   POST /api/login
// @desc    Authenticate user & get token
// @access  Public
app.post('/api/login', async (req, res) => {
    const { username, password } = req.body;

    if (!username || !password) {
        return res.status(400).json({ message: 'Please enter all fields' });
    }

    try {
        const user = await User.findOne({ username });
        if (!user) {
            return res.status(400).json({ message: 'Invalid credentials' });
        }

        const isMatch = await user.comparePassword(password);
        if (!isMatch) {
            return res.status(400).json({ message: 'Invalid credentials' });
        }

        // Set JWT as a httpOnly cookie
        const token = generateToken(user._id);
        res.cookie('jwt', token, {
            httpOnly: true,
            secure: process.env.NODE_ENV === 'production',
            maxAge: 3600000 // 1 hour
        });

        res.json({
            message: 'Logged in successfully',
            user: {
                id: user._id,
                username: user.username,
                isAdmin: user.isAdmin // Crucial for frontend redirection
            }
        });

    } catch (error) {
        console.error(error);
        res.status(500).json({ message: 'Server error during login' });
    }
});

// @route   GET /api/logout
// @desc    Logout user by clearing cookie
// @access  Private (but accessible after successful login)
app.get('/api/logout', (req, res) => {
    res.cookie('jwt', '', {
        httpOnly: true,
        expires: new Date(0) // Expire the cookie immediately
    });
    res.status(200).json({ message: 'Logged out successfully' });
});


// @route   POST /api/upload-qr
// @desc    Upload QR code image and perform analysis
// @access  Private (Regular Users Only)
app.post('/api/upload-qr', protect, upload.single('qrCodeImage'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ message: 'No QR code image uploaded.' });
    }
    // Prevent admin from using this endpoint (optional, but good practice)
    if (req.user && req.user.isAdmin) {
        fs.unlink(req.file.path, (err) => { // Clean up uploaded file
            if (err) console.error(`Error deleting temp file ${req.file.path}:`, err);
        });
        return res.status(403).json({ message: 'Admins cannot use this upload feature directly.' });
    }


    const tempImagePath = req.file.path; // Path to the temporary uploaded file
    const newFileName = `${req.file.filename}-${Date.now()}${path.extname(req.file.originalname)}`;
    const permanentImagePath = path.join(UPLOAD_DIR, newFileName);
    const userId = req.user.id; // Get user ID from authenticated request

    try {
        // --- Call Python ML Model for Prediction ---
        const pythonProcess = spawn('python', [
            path.join(__dirname, 'ml_inference', 'predict_model.py'),
            tempImagePath // Use the temporary path for prediction
        ]);

        let pythonOutput = '';
        // Pipe Python script's stderr to Node.js process's stderr
        pythonProcess.stderr.pipe(process.stderr);

        pythonProcess.stdout.on('data', (data) => {
            pythonOutput += data.toString();
        });

        pythonProcess.on('close', async (code) => {
            if (code !== 0) {
                console.error(`Python script exited with code ${code}.`);
                // Clean up temp file on ML failure
                fs.unlink(tempImagePath, (err) => {
                    if (err) console.error(`Error deleting temp file ${tempImagePath} on ML failure:`, err);
                });
                return res.status(500).json({ message: 'ML analysis failed. Check server console for Python errors.' });
            }

            try {
                const mlResult = JSON.parse(pythonOutput.trim()); 
                const { predictedClass, confidence } = mlResult;

                // --- Move file to permanent storage ---
                fs.rename(tempImagePath, permanentImagePath, async (err) => {
                    if (err) {
                        console.error(`Error moving file from ${tempImagePath} to ${permanentImagePath}:`, err);
                        return res.status(500).json({ message: 'Error saving QR code image permanently.' });
                    }

                    // Save scan details to MongoDB with permanent path
                    const qrScan = await QRCodeScan.create({
                        userId,
                        imageId: newFileName, // Use new permanent filename
                        filePath: permanentImagePath, // Store the permanent path
                        predictedClass,
                        confidence
                    });

                    res.json({
                        message: 'QR code analyzed successfully.',
                        prediction: {
                            predictedClass,
                            confidence,
                            scanId: qrScan._id,
                            imageUrl: `/uploaded_qr_codes/${newFileName}` // URL to access the image from frontend
                        }
                    });
                });

            } catch (parseError) {
                console.error('Error parsing Python output or saving to DB:', parseError);
                console.error('Python raw output (troubleshooting):', pythonOutput);
                // Clean up temp file on parsing/DB error
                fs.unlink(tempImagePath, (err) => {
                    if (err) console.error(`Error deleting temp file ${tempImagePath} on parse/DB error:`, err);
                });
                res.status(500).json({ message: 'Error processing ML result from Python script.', rawOutput: pythonOutput });
            }
        });

    } catch (error) {
        console.error('Server error during QR code upload/analysis:', error);
        // Clean up temp file on server error
        if (req.file && fs.existsSync(req.file.path)) {
             fs.unlink(req.file.path, (err) => {
                if (err) console.error(`Error deleting temp file ${req.file.path} on server error:`, err);
            });
        }
        res.status(500).json({ message: 'Server error during QR code upload/analysis.' });
    }
});

// @route   GET /api/dashboard
// @desc    Get user's scanned QR codes (Regular Users Only)
// @access  Private
app.get('/api/dashboard', protect, async (req, res) => {
    // Prevent admin from using this endpoint (optional)
    if (req.user && req.user.isAdmin) {
        return res.status(403).json({ message: 'Admins have a separate dashboard.' });
    }
    try {
        const qrScans = await QRCodeScan.find({ userId: req.user.id }).sort({ scanDate: -1 });
        // Modify filePath to imageUrl for frontend
        const scansWithUrls = qrScans.map(scan => ({
            ...scan.toObject(), // Convert Mongoose document to plain object
            imageUrl: `/uploaded_qr_codes/${path.basename(scan.filePath)}` // Create URL
        }));
        res.json(scansWithUrls);
    } catch (error) {
        console.error('Server error fetching dashboard data:', error);
        res.status(500).json({ message: 'Server error fetching dashboard data.' });
    }
});

// @route   PUT /api/report-qr/:id
// @desc    Report a QR code scan
// @access  Private (Regular Users Only)
app.put('/api/report-qr/:id', protect, async (req, res) => {
    // Prevent admin from using this endpoint (optional)
    if (req.user && req.user.isAdmin) {
        return res.status(403).json({ message: 'Admins cannot report QR codes via this route.' });
    }

    const { reportDetails } = req.body;
    try {
        const qrScan = await QRCodeScan.findOneAndUpdate(
            { _id: req.params.id, userId: req.user.id }, // Ensure user owns the scan
            { reported: true, reportDetails: reportDetails || 'User reported this QR code' },
            { new: true } // Return the updated document
        );

        if (!qrScan) {
            return res.status(404).json({ message: 'QR code scan not found or not authorized.' });
        }
        res.json({ message: 'QR code reported successfully.', qrScan });
    } catch (error) {
        console.error('Server error reporting QR code:', error);
        res.status(500).json({ message: 'Server error reporting QR code.' });
    }
});

// --- Admin Routes ---

// @route   GET /api/admin/dashboard
// @desc    Get all reported QR codes for admin dashboard
// @access  Private (Admin Only)
app.get('/api/admin/dashboard', adminProtect, async (req, res) => {
    try {
        // Find all reported QR codes and populate user details
        const reportedQrScans = await QRCodeScan.find({ reported: true })
                                                .populate('userId', 'username') // Fetch username of the reporting user
                                                .sort({ scanDate: -1 });
        
        // Convert filePath to imageUrl for frontend
        const scansWithUrls = reportedQrScans.map(scan => ({
            ...scan.toObject(),
            imageUrl: `/uploaded_qr_codes/${path.basename(scan.filePath)}`
        }));
        res.json(scansWithUrls);

    } catch (error) {
        console.error('Admin error fetching reported QR codes:', error);
        res.status(500).json({ message: 'Server error fetching reported QR codes.' });
    }
});

// @route   POST /api/admin/verify-qr/:id
// @desc    Admin verifies a reported QR code using ML model
// @access  Private (Admin Only)
app.post('/api/admin/verify-qr/:id', adminProtect, async (req, res) => {
    const scanId = req.params.id;

    try {
        const qrScan = await QRCodeScan.findById(scanId);

        if (!qrScan) {
            return res.status(404).json({ message: 'QR code scan not found.' });
        }
        
        // The imagePath is now the permanent path
        const imagePath = qrScan.filePath; 
        if (!fs.existsSync(imagePath)) {
            // This error should be rare now if files are moved correctly
            return res.status(404).json({ message: 'Original uploaded QR code image file not found on server for analysis. It might have been deleted or moved. (Admin Verify)' });
        }

        // --- Call Python ML Model for Prediction ---
        const pythonProcess = spawn('python', [
            path.join(__dirname, 'ml_inference', 'predict_model.py'),
            imagePath // Use the permanent path for prediction
        ]);

        let pythonOutput = '';
        pythonProcess.stderr.pipe(process.stderr); // Pipe Python errors to Node.js stderr

        pythonProcess.stdout.on('data', (data) => {
            pythonOutput += data.toString();
        });

        pythonProcess.on('close', async (code) => {
            if (code !== 0) {
                console.error(`Python script exited with code ${code}.`);
                return res.status(500).json({ message: 'ML verification failed. Check server console for Python errors.' });
            }

            try {
                const mlResult = JSON.parse(pythonOutput.trim());
                const { predictedClass, confidence } = mlResult;

                // Update the QR code scan record with admin verification results
                qrScan.adminVerified = true;
                qrScan.adminVerifiedClass = predictedClass;
                qrScan.adminVerifiedConfidence = confidence;
                qrScan.adminVerificationDate = new Date();
                await qrScan.save();

                res.json({
                    message: 'QR code verified successfully.',
                    prediction: {
                        predictedClass,
                        confidence,
                        scanId: qrScan._id
                    }
                });

            } catch (parseError) {
                console.error('Error parsing Python output or updating DB after admin verify:', parseError);
                console.error('Python raw output (troubleshooting):', pythonOutput);
                res.status(500).json({ message: 'Error processing ML result during admin verification.', rawOutput: pythonOutput });
            }
        });

    } catch (error) {
        console.error('Server error during admin QR code verification:', error);
        res.status(500).json({ message: 'Server error during admin QR code verification.' });
    }
});

// @route   DELETE /api/admin/remove-qr/:id
// @desc    Admin removes a QR code scan record
// @access  Private (Admin Only)
app.delete('/api/admin/remove-qr/:id', adminProtect, async (req, res) => {
    const scanId = req.params.id;
    try {
        const qrScan = await QRCodeScan.findByIdAndDelete(scanId);

        if (!qrScan) {
            return res.status(404).json({ message: 'QR code scan not found.' });
        }
        // Also delete the original uploaded image file from permanent storage
        if (qrScan.filePath && fs.existsSync(qrScan.filePath)) {
            fs.unlink(qrScan.filePath, (err) => {
                if (err) console.error(`Error deleting associated permanent file ${qrScan.filePath}:`, err);
            });
        }
        res.json({ message: 'QR code scan removed successfully.' });
    } catch (error) {
        console.error('Server error removing QR code scan:', error);
        res.status(500).json({ message: 'Server error removing QR code scan.' });
    }
});


// --- Start Server ---
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));

