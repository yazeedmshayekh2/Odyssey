# MongoDB Setup for Car Insurance System

This document provides instructions for setting up MongoDB for the Car Insurance System.

## Installation

### Linux (Ubuntu/Debian)

```bash
# Import MongoDB public GPG key
wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -

# Create a list file for MongoDB
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list

# Update package database
sudo apt-get update

# Install MongoDB
sudo apt-get install -y mongodb-org

# Start MongoDB service
sudo systemctl start mongod

# Enable MongoDB to start on boot
sudo systemctl enable mongod
```

### macOS (using Homebrew)

```bash
# Install MongoDB
brew tap mongodb/brew
brew install mongodb-community

# Start MongoDB service
brew services start mongodb-community
```

### Windows

1. Download the MongoDB Community Server from [MongoDB Download Center](https://www.mongodb.com/try/download/community)
2. Run the installer and follow the installation wizard
3. Choose "Complete" installation
4. Install MongoDB as a service
5. Start the MongoDB service from Windows Services

## Initial Setup

After installing MongoDB, you need to create a database and set up some initial collections:

```bash
# Start MongoDB shell
mongosh

# Create car_insurance database
use car_insurance

# Create collections
db.createCollection("users")
db.createCollection("car_profiles")
db.createCollection("damage_reports")

# Optional: Create indexes for faster queries
db.users.createIndex({ "email": 1 }, { unique: true })
db.users.createIndex({ "username": 1 }, { unique: true })
db.car_profiles.createIndex({ "license_plate": 1 })
db.car_profiles.createIndex({ "owner_id": 1 })
```

## Environment Configuration

Create a `.env` file in the root of your project with the following content:

```
# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017/
DB_NAME=car_insurance

# Flask Configuration
SECRET_KEY=your_secret_key_here
DEBUG=True
PORT=5000

# Upload Configuration
MAX_CONTENT_LENGTH=16777216  # 16MB in bytes
```

Replace `your_secret_key_here` with a strong random key for your application.

## Verify Installation

To verify that MongoDB is running correctly:

```bash
# Check MongoDB service status
sudo systemctl status mongod  # Linux
brew services list  # macOS

# Connect to MongoDB
mongosh
```

You should see the MongoDB shell prompt, indicating that you've successfully connected to the MongoDB server.

## Common Issues

### MongoDB not starting

If MongoDB doesn't start, check the logs:

```bash
# Linux
sudo tail -f /var/log/mongodb/mongod.log

# macOS
tail -f /usr/local/var/log/mongodb/mongo.log
```

### Permission issues

If you encounter permission issues, ensure that the MongoDB data directory has the correct permissions:

```bash
# Linux
sudo chown -R mongodb:mongodb /var/lib/mongodb
sudo chmod 755 /var/lib/mongodb
```

### Connection issues

If you can't connect to MongoDB, make sure the MongoDB service is running and that you're using the correct connection string:

```bash
# Check if MongoDB is listening on port 27017
nc -zv localhost 27017
``` 