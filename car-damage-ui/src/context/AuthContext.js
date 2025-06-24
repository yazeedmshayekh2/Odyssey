import React, { createContext, useState, useEffect, useContext } from 'react';
import authService from '../services/authService';
import debugAuth from '../utils/debugAuth';

// Create authentication context
const AuthContext = createContext();

// AuthProvider component to wrap application
export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Initialize auth state from localStorage on component mount
  useEffect(() => {
    const initializeAuth = async () => {
      try {
        const token = authService.getToken();
        if (token) {
          console.log('Found existing token, validating...');
          // Get user data from API
          const response = await authService.getCurrentUser();
          if (response.success) {
            setUser(response.user);
            setIsAuthenticated(true);
            console.log('Auth initialization successful');
          } else {
            console.log('Auth validation failed:', response);
            // If API call fails, clear localStorage
            authService.logout();
          }
        } else {
          console.log('No token found, user not authenticated');
        }
      } catch (err) {
        console.error('Auth initialization error:', err);
        // Handle different types of errors
        if (err.message === 'Network error' || err.message?.includes('Failed to fetch')) {
          // Network error - don't clear localStorage, just show as not authenticated
          console.log('Network error during auth check - keeping token for retry');
        } else {
          // Authentication error - clear localStorage
        authService.logout();
          console.log('Auth error - cleared localStorage');
        }
      } finally {
        setLoading(false);
      }
    };

    initializeAuth();
  }, []);

  // Register new user
  const register = async (userData) => {
    setLoading(true);
    setError(null);
    try {
      const response = await authService.register(userData);
      if (response.success) {
        setUser(response.user);
        setIsAuthenticated(true);
        return { success: true };
      }
    } catch (err) {
      setError(err.message || 'Registration failed');
      return { success: false, message: err.message || 'Registration failed' };
    } finally {
      setLoading(false);
    }
  };

  // Login user
  const login = async (email, password) => {
    setLoading(true);
    setError(null);
    try {
      const response = await authService.login(email, password);
      if (response.success) {
        setUser(response.user);
        setIsAuthenticated(true);
        return { success: true };
      }
    } catch (err) {
      setError(err.message || 'Login failed');
      return { success: false, message: err.message || 'Login failed' };
    } finally {
      setLoading(false);
    }
  };

  // Logout user
  const logout = () => {
    authService.logout();
    setUser(null);
    setIsAuthenticated(false);
  };

  // Upload user documents
  const uploadDocuments = async (licenseImage, carImage) => {
    setLoading(true);
    setError(null);
    try {
      const response = await authService.uploadDocuments(licenseImage, carImage);
      if (response.success) {
        // Update user information after successful upload
        const userResponse = await authService.getCurrentUser();
        if (userResponse.success) {
          setUser(userResponse.user);
        }
        return { success: true };
      }
    } catch (err) {
      setError(err.message || 'Document upload failed');
      return { success: false, message: err.message || 'Document upload failed' };
    } finally {
      setLoading(false);
    }
  };

  // Context value
  const value = {
    user,
    isAuthenticated,
    loading,
    error,
    register,
    login,
    logout,
    uploadDocuments
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

// Custom hook to use auth context
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export default AuthContext; 