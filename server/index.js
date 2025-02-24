const express = require('express');
const dotenv = require('dotenv');
const connectDB = require('./configs/db');
const authRoutes = require('./routes/authRoutes');

// Get the JWT Secret Key
// const crypto = require('crypto');
// const secret = crypto.randomBytes(64).toString('hex');
// console.log(secret);

// Load environment variables
dotenv.config();
const app = express();

// Connect to database
connectDB();

// Middleware
app.use(express.json());

// Routes
app.use('/api/auth', authRoutes);

// Start the server
const PORT = process.env.PORT || 8088;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
