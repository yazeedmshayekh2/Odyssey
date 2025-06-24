import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import styled from 'styled-components';
import { useAuth } from '../../context/AuthContext';
import authService from '../../services/authService';

// Styled components (reusing styles from Login component)
const RegisterContainer = styled.div`
  max-width: 500px;
  margin: 2rem auto;
  padding: 2rem;
  background-color: white;
  border-radius: 0.5rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
`;

const Title = styled.h2`
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 1.5rem;
  color: #1d3557;
  text-align: center;
`;

const FormGroup = styled.div`
  margin-bottom: 1.25rem;
`;

const FlexRow = styled.div`
  display: flex;
  gap: 1rem;
  
  @media (max-width: 768px) {
    flex-direction: column;
    gap: 0;
  }
`;

const Label = styled.label`
  display: block;
  margin-bottom: 0.5rem;
  color: #4b5563;
  font-weight: 500;
`;

const Input = styled.input`
  width: 100%;
  padding: 0.75rem;
  border: 1px solid #d1d5db;
  border-radius: 0.375rem;
  font-size: 1rem;
  &:focus {
    outline: none;
    border-color: #1d3557;
    box-shadow: 0 0 0 3px rgba(29, 53, 87, 0.2);
  }
`;

const Button = styled.button`
  width: 100%;
  padding: 0.75rem;
  background-color: #1d3557;
  color: white;
  border: none;
  border-radius: 0.375rem;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  transition: background-color 0.2s;
  &:hover {
    background-color: #14253e;
  }
  &:disabled {
    background-color: #9ca3af;
    cursor: not-allowed;
  }
`;

const ErrorMessage = styled.div`
  background-color: #fee2e2;
  color: #b91c1c;
  padding: 0.75rem;
  border-radius: 0.375rem;
  margin-bottom: 1rem;
`;

const LinkContainer = styled.div`
  text-align: center;
  margin-top: 1.5rem;
`;

const StyledLink = styled(Link)`
  color: #1d3557;
  text-decoration: none;
  font-weight: 500;
  &:hover {
    text-decoration: underline;
  }
`;

const Register = () => {
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
    firstName: '',
    lastName: ''
  });
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  const { register } = useAuth();
  const navigate = useNavigate();
  
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prevState => ({
      ...prevState,
      [name]: value
    }));
  };
  
  const validateForm = () => {
    if (!formData.username || !formData.email || !formData.password || 
        !formData.confirmPassword || !formData.firstName || !formData.lastName) {
      setError('All fields are required');
      return false;
    }
    
    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      return false;
    }
    
    if (formData.password.length < 6) {
      setError('Password must be at least 6 characters long');
      return false;
    }
    
    return true;
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validate form
    if (!validateForm()) {
      return;
    }
    
    setError('');
    setIsSubmitting(true);
    
    try {
      // Skip the context and use the service directly for better error handling
      const response = await authService.register(formData);
      
      if (response.success) {
        navigate('/documents');
      } else {
        setError(response.message || 'Registration failed');
      }
    } catch (err) {
      console.error('Registration error caught:', err);
      setError(err.message || 'Network error. Please check your connection and try again.');
    } finally {
      setIsSubmitting(false);
    }
  };
  
  // For debugging - check if backend is accessible
  const checkBackendConnection = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/health', { 
        method: 'GET'
      });
      
      if (response.ok) {
        console.log('Backend connection successful');
        alert('Backend connection successful');
      } else {
        console.error('Backend connection failed with status:', response.status);
        alert(`Backend returned error: ${response.status}`);
      }
    } catch (err) {
      console.error('Backend connection error:', err);
      alert(`Cannot connect to backend: ${err.message}`);
    }
  };
  
  return (
    <RegisterContainer>
      <Title>Create an account</Title>
      
      {error && <ErrorMessage>{error}</ErrorMessage>}
      
      <form onSubmit={handleSubmit}>
        <FlexRow>
          <FormGroup>
            <Label htmlFor="firstName">First Name</Label>
            <Input
              type="text"
              id="firstName"
              name="firstName"
              value={formData.firstName}
              onChange={handleChange}
              placeholder="Enter your first name"
              required
            />
          </FormGroup>
          
          <FormGroup>
            <Label htmlFor="lastName">Last Name</Label>
            <Input
              type="text"
              id="lastName"
              name="lastName"
              value={formData.lastName}
              onChange={handleChange}
              placeholder="Enter your last name"
              required
            />
          </FormGroup>
        </FlexRow>
        
        <FormGroup>
          <Label htmlFor="username">Username</Label>
          <Input
            type="text"
            id="username"
            name="username"
            value={formData.username}
            onChange={handleChange}
            placeholder="Choose a username"
            required
          />
        </FormGroup>
        
        <FormGroup>
          <Label htmlFor="email">Email</Label>
          <Input
            type="email"
            id="email"
            name="email"
            value={formData.email}
            onChange={handleChange}
            placeholder="Enter your email"
            required
          />
        </FormGroup>
        
        <FlexRow>
          <FormGroup>
            <Label htmlFor="password">Password</Label>
            <Input
              type="password"
              id="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              placeholder="Create a password"
              required
            />
          </FormGroup>
          
          <FormGroup>
            <Label htmlFor="confirmPassword">Confirm Password</Label>
            <Input
              type="password"
              id="confirmPassword"
              name="confirmPassword"
              value={formData.confirmPassword}
              onChange={handleChange}
              placeholder="Confirm your password"
              required
            />
          </FormGroup>
        </FlexRow>
        
        <Button type="submit" disabled={isSubmitting}>
          {isSubmitting ? 'Creating account...' : 'Create Account'}
        </Button>
        
        {process.env.NODE_ENV === 'development' && (
          <Button 
            type="button" 
            onClick={checkBackendConnection}
            style={{ marginTop: '10px', backgroundColor: '#4a5568' }}
          >
            Check Server Connection
          </Button>
        )}
      </form>
      
      <LinkContainer>
        <p>
          Already have an account? <StyledLink to="/login">Sign in</StyledLink>
        </p>
      </LinkContainer>
    </RegisterContainer>
  );
};

export default Register; 