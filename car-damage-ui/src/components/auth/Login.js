import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import styled from 'styled-components';
import { useAuth } from '../../context/AuthContext';

// Styled components
const LoginContainer = styled.div`
  max-width: 450px;
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

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  const { login } = useAuth();
  const navigate = useNavigate();
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validate form
    if (!email || !password) {
      setError('Email and password are required');
      return;
    }
    
    setError('');
    setIsSubmitting(true);
    
    try {
      const result = await login(email, password);
      if (result.success) {
        navigate('/dashboard');
      } else {
        setError(result.message || 'Invalid credentials');
      }
    } catch (err) {
      setError(err.message || 'An error occurred while logging in');
    } finally {
      setIsSubmitting(false);
    }
  };
  
  return (
    <LoginContainer>
      <Title>Sign in to your account</Title>
      
      {error && <ErrorMessage>{error}</ErrorMessage>}
      
      <form onSubmit={handleSubmit}>
        <FormGroup>
          <Label htmlFor="email">Email</Label>
          <Input
            type="email"
            id="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Enter your email"
            required
          />
        </FormGroup>
        
        <FormGroup>
          <Label htmlFor="password">Password</Label>
          <Input
            type="password"
            id="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Enter your password"
            required
          />
        </FormGroup>
        
        <Button type="submit" disabled={isSubmitting}>
          {isSubmitting ? 'Signing in...' : 'Sign In'}
        </Button>
      </form>
      
      <LinkContainer>
        <p>
          Don't have an account? <StyledLink to="/register">Sign up</StyledLink>
        </p>
      </LinkContainer>
    </LoginContainer>
  );
};

export default Login; 