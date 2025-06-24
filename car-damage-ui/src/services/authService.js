import axios from 'axios';

const API_URL = 'http://localhost:5000/api';

// Create axios instance with base URL
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Add JWT token to requests if available
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Authentication service functions
const authService = {
  // Register a new user
  register: async (userData) => {
    try {
      console.log('Registering user with data:', userData);
      console.log('API URL:', `${API_URL}/auth/register`);
      
      // Make the API call
      const response = await axios.post(`${API_URL}/auth/register`, userData, {
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      console.log('Registration response:', response.data);
      
      if (response.data.token) {
        localStorage.setItem('token', response.data.token);
        localStorage.setItem('user', JSON.stringify(response.data.user));
      }
      return response.data;
    } catch (error) {
      console.error('Registration error:', error);
      if (error.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        console.error('Error response data:', error.response.data);
        console.error('Error response status:', error.response.status);
        console.error('Error response headers:', error.response.headers);
        throw error.response.data || { message: `Server error: ${error.response.status}` };
      } else if (error.request) {
        // The request was made but no response was received
        console.error('Error request:', error.request);
        throw { message: 'No response from server. Please check if the backend is running.' };
      } else {
        // Something happened in setting up the request that triggered an Error
        console.error('Error message:', error.message);
        throw { message: error.message || 'Network error' };
      }
    }
  },

  // Login user
  login: async (email, password) => {
    try {
      console.log('Logging in user:', email);
      
      // Use fetch API for consistency
      const response = await fetch(`${API_URL}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email, password })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Login response received:', data);
      
      if (data.token) {
        // Store token in localStorage
        localStorage.setItem('token', data.token);
        console.log('Token stored, length:', data.token.length);
        
        // Store user in localStorage
        localStorage.setItem('user', JSON.stringify(data.user));
      } else {
        console.error('No token received in login response');
      }
      
      return data;
    } catch (error) {
      console.error('Login error:', error);
      
      // More detailed error handling
      if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
        throw { message: 'Failed to connect to server. Please check your connection.' };
      } else if (error.message && error.message.includes('HTTP error')) {
        throw { message: error.message };
      } else {
        throw error.response?.data || { message: 'Network error' };
      }
    }
  },

  // Logout user
  logout: () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
  },

  // Get current user profile
  getCurrentUser: async () => {
    try {
      const response = await api.get('/auth/me');
      return response.data;
    } catch (error) {
      console.error('getCurrentUser error:', error);
      
      if (error.response) {
        // Server responded with error status
        console.error('Error response status:', error.response.status);
        console.error('Error response data:', error.response.data);
        
        if (error.response.status === 401) {
          throw { message: 'Authentication expired', status: 401 };
        } else if (error.response.status >= 500) {
          throw { message: 'Server error', status: error.response.status };
        }
        
        throw error.response.data || { message: `Server error: ${error.response.status}` };
      } else if (error.request) {
        // Request made but no response received
        console.error('No response received:', error.request);
        throw { message: 'Network error - no response from server' };
      } else {
        // Error setting up request
        console.error('Request setup error:', error.message);
        throw { message: error.message || 'Network error' };
      }
    }
  },

  // Upload user documents
  uploadDocuments: async (licenseImage, carImage) => {
    try {
      const formData = new FormData();
      formData.append('licenseImage', licenseImage);
      formData.append('carImage', carImage);
      
      const response = await axios.post(`${API_URL}/auth/upload-documents`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      return response.data;
    } catch (error) {
      throw error.response?.data || { message: 'Network error' };
    }
  },

  // Check if user is authenticated
  isAuthenticated: () => {
    return !!localStorage.getItem('token');
  },

  // Get authentication token
  getToken: () => {
    return localStorage.getItem('token');
  },

  // Get current user from localStorage
  getUser: () => {
    const user = localStorage.getItem('user');
    return user ? JSON.parse(user) : null;
  }
};

export default authService; 