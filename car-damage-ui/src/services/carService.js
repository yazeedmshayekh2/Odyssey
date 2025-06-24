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

// Car service functions
const carService = {
  // Debug function to check token and API connectivity
  testApi: async () => {
    try {
      const token = localStorage.getItem('token');
      console.log('Debug - Token for API test:', token ? `${token.substring(0, 10)}...` : 'No token');
      
      // Test both fetch and axios
      const fetchResponse = await fetch(`${API_URL}/health`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      const fetchData = await fetchResponse.json();
      console.log('Debug - Fetch health response:', fetchData);
      
      const axiosResponse = await axios.get(`${API_URL}/health`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      console.log('Debug - Axios health response:', axiosResponse.data);
      
      return {
        success: true,
        fetch: fetchData,
        axios: axiosResponse.data,
        token: !!token
      };
    } catch (error) {
      console.error('Debug - API test error:', error);
      return {
        success: false,
        error: error.message
      };
    }
  },

  // Check if a license plate is already in use
  checkLicensePlateExists: async (licensePlate, excludeCarId = null) => {
    try {
      const token = localStorage.getItem('token');
      
      // Use fetch API to get all user cars
      const response = await fetch(`${API_URL}/cars/`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      
      // If successful, check if any car has the same license plate
      if (data.success) {
        const cars = data.cars || [];
        // Find car with matching license plate, excluding the current car being edited
        const existingCar = cars.find(car => 
          car.licensePlate.toLowerCase() === licensePlate.toLowerCase() && 
          (!excludeCarId || car.id !== excludeCarId)
        );
        
        return {
          exists: !!existingCar,
          car: existingCar || null
        };
      }
      
      return { exists: false, car: null };
    } catch (error) {
      console.error('Error checking license plate:', error);
      throw { message: 'Failed to check license plate. Please try again.' };
    }
  },

  // Get list of available car parts for insurance
  getCarParts: async () => {
    try {
      const token = localStorage.getItem('token');
      
      // Use fetch API directly with proper authorization
      const response = await fetch(`${API_URL}/cars/parts`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error fetching car parts:', error);
      
      // More detailed error handling
      if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
        throw { message: 'Failed to load car parts. Please check your connection.' };
      } else if (error.message && error.message.includes('HTTP error')) {
        throw { message: error.message };
      } else {
        throw error.response?.data || { message: 'Network error' };
      }
    }
  },

  // Get all cars for the current user
  getCars: async () => {
    try {
      const token = localStorage.getItem('token');
      
      // Use fetch API directly with proper authorization
      const response = await fetch(`${API_URL}/cars/`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error fetching cars:', error);
      
      // More detailed error handling
      if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
        throw { message: 'Failed to load cars. Please check your connection.' };
      } else if (error.message && error.message.includes('HTTP error')) {
        throw { message: error.message };
      } else {
        throw error.response?.data || { message: 'Network error' };
      }
    }
  },

  // Alias for backward compatibility
  getUserCars: async () => {
    return carService.getCars();
  },

  // Get a specific car by ID
  getCarById: async (carId) => {
    try {
      const token = localStorage.getItem('token');
      
      // Use fetch API directly with proper authorization
      const response = await fetch(`${API_URL}/cars/${carId}`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error fetching car details:', error);
      
      // More detailed error handling
      if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
        throw { message: 'Failed to load car details. Please check your connection.' };
      } else if (error.message && error.message.includes('HTTP error')) {
        throw { message: error.message };
      } else {
        throw error.response?.data || { message: 'Network error' };
      }
    }
  },

  // Register a new car
  registerCar: async (carData, images) => {
    try {
      const formData = new FormData();
      
      // Add car information to form data
      Object.keys(carData).forEach(key => {
        if (key === 'insuredParts' && Array.isArray(carData[key])) {
          carData[key].forEach(part => {
            formData.append('insuredParts', part);
          });
        } else {
          formData.append(key, carData[key]);
        }
      });
      
      // Add car image to form data if available
      if (images.carImage) formData.append('carImage', images.carImage);
      
      console.log('Sending request to register car:', `${API_URL}/cars`);
      console.log('Form data keys:', [...formData.keys()]);

      const token = localStorage.getItem('token');
      console.log('Token available:', !!token);
      
      // Use fetch API directly for multipart/form-data
      const response = await fetch(`${API_URL}/cars/`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
          // Don't set Content-Type for multipart/form-data with fetch
        },
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Response received:', data);
      return data;
    } catch (error) {
      console.error('Detailed error:', error);
      
      // More detailed error handling for fetch
      if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
        console.error('Network error - server might be down or unreachable');
        throw { message: 'No response from server. Please check your connection.' };
      } else if (error.message && error.message.includes('HTTP error')) {
        console.error('HTTP error:', error.message);
        throw { message: error.message };
      } else if (error.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        console.error('Error data:', error.response.data);
        console.error('Error status:', error.response.status);
        console.error('Error headers:', error.response.headers);
        throw error.response.data || { message: `Server error: ${error.response.status}` };
      } else if (error.request) {
        // The request was made but no response was received
        console.error('No response received:', error.request);
        throw { message: 'No response from server. Please check your connection.' };
      } else {
        // Something happened in setting up the request that triggered an Error
        console.error('Request setup error:', error.message);
        throw { message: `Request error: ${error.message}` };
      }
    }
  },

  // Update car information
  updateCar: async (carId, carData, images = {}) => {
    try {
      const formData = new FormData();
      
      // Add car information to form data
      Object.keys(carData).forEach(key => {
        if (key === 'insuredParts' && Array.isArray(carData[key])) {
          carData[key].forEach(part => {
            formData.append('insuredParts', part);
          });
        } else {
          formData.append(key, carData[key]);
        }
      });
      
      // Add car image to form data if available
      if (images.carImage) formData.append('carImage', images.carImage);
      
      const token = localStorage.getItem('token');
      
      // Use fetch API directly for multipart/form-data
      const response = await fetch(`${API_URL}/cars/${carId}`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`
          // Don't set Content-Type for multipart/form-data with fetch
        },
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      throw error.response?.data || { message: 'Network error' };
    }
  },

  // Delete a car
  deleteCar: async (carId) => {
    try {
      const token = localStorage.getItem('token');
      
      // Use fetch API directly with proper authorization
      const response = await fetch(`${API_URL}/cars/${carId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error deleting car:', error);
      
      // More detailed error handling
      if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
        throw { message: 'Failed to delete car. Please check your connection.' };
      } else if (error.message && error.message.includes('HTTP error')) {
        throw { message: error.message };
      } else {
        throw error.response?.data || { message: 'Network error' };
      }
    }
  },

  // Submit car for damage detection
  detectDamage: async (image) => {
    try {
      const formData = new FormData();
      formData.append('image', image);
      
      const token = localStorage.getItem('token');
      
      // Use fetch API directly for multipart/form-data
      const response = await fetch(`${API_URL}/detect-damage`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
          // Don't set Content-Type for multipart/form-data with fetch
        },
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      throw error.response?.data || { message: 'Network error' };
    }
  },
  
  // Compare accident image with registration images to identify new damage
  compareCarImages: async (carId, accidentImage) => {
    try {
      // Create FormData object
      const formData = new FormData();
      formData.append('image', accidentImage);
      
      const token = localStorage.getItem('token');
      
      // Use fetch API directly for multipart/form-data
      const response = await fetch(`${API_URL}/cars/${carId}/compare-damage`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
          // Don't set Content-Type for multipart/form-data with fetch
        },
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error comparing car images:', error);
      
      // More detailed error handling
      if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
        throw { message: 'Failed to compare images. Please check your connection.' };
      } else if (error.message && error.message.includes('HTTP error')) {
        throw { message: error.message };
      } else {
        throw error.response?.data || { message: 'Network error' };
      }
    }
  }
};

export default carService; 