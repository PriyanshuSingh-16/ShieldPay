const mongoose = require('mongoose');

const QRCodeScanSchema = new mongoose.Schema({
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User', // Reference to the User model
        required: true
    },
    imageId: { // The unique ID generated for the image (e.g., from Multer's filename)
        type: String,
        required: true
    },
    filePath: { // Path to the uploaded image file (temporary in 'uploads/')
        type: String,
        required: true
    },
    scanDate: {
        type: Date,
        default: Date.now
    },
    predictedClass: { // Initial prediction by ML model when uploaded by user ("Safe", "Suspicious", "Malicious")
        type: String,
        required: true
    },
    confidence: { // Initial confidence array from the ML model
        type: [Number], // Store as an array of numbers
        required: true
    },
    reported: { // True if user reported it
        type: Boolean,
        default: false
    },
    reportDetails: { // Details provided by user when reporting
        type: String,
        default: null
    },
    adminVerified: { // NEW: True if admin has run verification
        type: Boolean,
        default: false
    },
    adminVerifiedClass: { // NEW: ML model result after admin verification
        type: String,
        default: null
    },
    adminVerifiedConfidence: { // NEW: ML model confidence after admin verification
        type: [Number],
        default: null
    },
    adminVerificationDate: { // NEW: Timestamp of admin verification
        type: Date,
        default: null
    }
});

module.exports = mongoose.model('QRCodeScan', QRCodeScanSchema);
