// Debug utilities for authentication troubleshooting
const debugAuth = {
  // Log current auth state
  logAuthState: () => {
    const token = localStorage.getItem('token');
    const user = localStorage.getItem('user');
    
    console.log('=== AUTH DEBUG INFO ===');
    console.log('Token exists:', !!token);
    console.log('Token length:', token ? token.length : 0);
    console.log('Token preview:', token ? `${token.substring(0, 20)}...` : 'None');
    console.log('User data:', user ? JSON.parse(user) : 'None');
    console.log('======================');
    
    return {
      hasToken: !!token,
      tokenLength: token ? token.length : 0,
      hasUser: !!user,
      userData: user ? JSON.parse(user) : null
    };
  },

  // Clear all auth data
  clearAuth: () => {
    console.log('Clearing all authentication data...');
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    console.log('Authentication data cleared. Please refresh the page.');
  },

  // Test auth endpoint
  testAuthEndpoint: async () => {
    try {
      const token = localStorage.getItem('token');
      console.log('Testing auth endpoint...');
      
      const response = await fetch('http://localhost:5000/api/auth/me', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          ...(token && { 'Authorization': `Bearer ${token}` })
        }
      });
      
      const data = await response.json();
      console.log('Auth endpoint response:', {
        status: response.status,
        ok: response.ok,
        data: data
      });
      
      return {
        status: response.status,
        ok: response.ok,
        data: data
      };
    } catch (error) {
      console.error('Auth endpoint test failed:', error);
      return {
        error: error.message
      };
    }
  },

  // Full debug report
  fullReport: async () => {
    console.log('=== FULL AUTH DEBUG REPORT ===');
    const authState = debugAuth.logAuthState();
    const endpointTest = await debugAuth.testAuthEndpoint();
    
    console.log('Auth State:', authState);
    console.log('Endpoint Test:', endpointTest);
    console.log('=============================');
    
    return {
      authState,
      endpointTest
    };
  }
};

// Make available globally in development
if (process.env.NODE_ENV === 'development') {
  window.debugAuth = debugAuth;
}

export default debugAuth; 