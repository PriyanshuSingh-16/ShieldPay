const jwt = require('jsonwebtoken');
const User = require('./models/User'); // Assuming User model is in models/User.js

// Middleware to protect general user routes
const protect = async (req, res, next) => {
    let token;

    // Check for token in cookies
    if (req.cookies && req.cookies.jwt) {
        token = req.cookies.jwt;
    }

    if (!token) {
        return res.status(401).json({ message: 'Not authorized, no token' });
    }

    try {
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        req.user = await User.findById(decoded.id).select('-password'); // Get user without password
        if (!req.user) {
            return res.status(401).json({ message: 'Not authorized, user not found' });
        }
        next();
    } catch (error) {
        console.error(error);
        return res.status(401).json({ message: 'Not authorized, token failed' });
    }
};

// Middleware to protect admin routes
const adminProtect = async (req, res, next) => {
    // First, use the regular protect middleware to ensure user is logged in
    await protect(req, res, () => {
        if (req.user && req.user.isAdmin) {
            next(); // User is logged in and is an admin
        } else {
            // If protect middleware already sent a 401, this won't execute.
            // But if user is found but not admin, send 403.
            return res.status(403).json({ message: 'Not authorized as admin' });
        }
    });
};

module.exports = { protect, adminProtect };
