import React, { useState } from 'react';
import axios from 'axios';
import styled from 'styled-components';
import './App.css';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';

// Import components
import Header from './components/layout/Header';
import Dashboard from './components/layout/Dashboard';
import Login from './components/auth/Login';
import Register from './components/auth/Register';
import DocumentUpload from './components/auth/DocumentUpload';
import CarList from './components/cars/CarList';
import CarForm from './components/cars/CarForm';
import DamageDetection from './components/cars/DamageDetection';
import DamageReport from './components/damage/DamageReport';
import DamageReportsList from './components/damage/DamageReportsList';
import NewAccidentReport from './components/damage/NewAccidentReport';

// Original damage detection UI
import OriginalApp from './OriginalApp';

// Styled Components
const AppContainer = styled.div`
  min-height: 100vh;
  background-color: #f8f9fa;
`;

const HeaderContent = styled.div`
  width: 100%;
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1rem;
`;

const Title = styled.h1`
  font-size: 1.875rem;
  font-weight: bold;
  display: flex;
  align-items: center;
`;

const Subtitle = styled.p`
  margin-top: 0.5rem;
`;

const MainContent = styled.main`
  min-height: calc(100vh - 64px);
`;

const Card = styled.div`
  background-color: white;
  border-radius: 0.5rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  padding: 1.5rem;
  margin-bottom: 1.5rem;
`;

const SectionTitle = styled.h2`
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 1rem;
  color: #1d3557;
`;

const FormGroup = styled.div`
  margin-bottom: 1rem;
`;

const Label = styled.label`
  display: block;
  color: #4b5563;
  margin-bottom: 0.5rem;
`;

const FileInputWrapper = styled.div`
  display: flex;
  align-items: center;
`;

const FileInputButton = styled.label`
  cursor: pointer;
  background-color: #e63946;
  color: white;
  padding: 0.5rem 1rem;
  border-radius: 0.25rem;
  &:hover {
    background-color: #c1121f;
  }
  transition: background-color 0.2s ease;
`;

const FileInputText = styled.span`
  margin-left: 0.75rem;
  color: #6b7280;
`;

const PreviewContainer = styled.div`
  margin-bottom: 1rem;
`;

const PreviewImage = styled.img`
  max-width: 100%;
  height: auto;
  border-radius: 0.25rem;
  border: 1px solid #e5e7eb;
`;

const ErrorMessage = styled.div`
  background-color: #fee2e2;
  border-left: 4px solid #ef4444;
  color: #b91c1c;
  padding: 1rem;
  margin-bottom: 1rem;
`;

const ButtonGroup = styled.div`
  display: flex;
  gap: 0.75rem;
`;

const PrimaryButton = styled.button`
  background-color: #1d3557;
  color: white;
  padding: 0.5rem 1.5rem;
  border-radius: 0.25rem;
  &:hover {
    background-color: #14253e;
  }
  &:disabled {
    opacity: 0.5;
  }
  transition: background-color 0.2s ease;
  font-weight: 500;
`;

const SecondaryButton = styled.button`
  background-color: #e5e7eb;
  color: #1f2937;
  padding: 0.5rem 1.5rem;
  border-radius: 0.25rem;
  &:hover {
    background-color: #d1d5db;
  }
  transition: background-color 0.2s ease;
  font-weight: 500;
`;

const ResultsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr;
  gap: 1.5rem;
  
  @media (min-width: 768px) {
    grid-template-columns: 1fr 1fr;
  }
`;

const ResultColumn = styled.div``;

const ResultCard = styled.div`
  background-color: ${props => props.light ? '#f1faee' : '#f9fafb'};
  border-radius: 0.5rem;
  padding: 1.25rem;
  margin-bottom: ${props => props.marginBottom ? '1rem' : '0'};
`;

const ResultTitle = styled.h3`
  font-size: 1.25rem;
  font-weight: 500;
  margin-bottom: 0.75rem;
  color: #4b5563;
`;

const DamagePercentage = styled.div`
  font-size: 2.25rem;
  font-weight: 700;
  color: #e63946;
`;

const DamageLabel = styled.div`
  color: #6b7280;
`;

const DetectedIssuesTitle = styled.h4`
  font-weight: 500;
  color: #4b5563;
  margin-bottom: 0.5rem;
`;

const IssuesList = styled.ul`
  margin: 0;
  padding: 0;
  list-style: none;
`;

const IssueItem = styled.li`
  display: flex;
  align-items: center;
  margin-bottom: 0.5rem;
`;

const IssueDot = styled.span`
  display: inline-block;
  width: 0.75rem;
  height: 0.75rem;
  background-color: #e63946;
  border-radius: 50%;
  margin-right: 0.5rem;
`;

const IssueText = styled.span`
  font-weight: 500;
`;

const IssueConfidence = styled.span`
  margin-left: 0.5rem;
  color: #6b7280;
`;

const ImageBorder = styled.div`
  border: 1px solid #e5e7eb;
  border-radius: 0.25rem;
  overflow: hidden;
`;

const UploadPlaceholder = styled.div`
  background-color: #f9fafb;
  border: 2px dashed #e5e7eb;
  border-radius: 0.5rem;
  padding: 3rem;
  text-align: center;
`;

const PlaceholderContent = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  color: #6b7280;
`;

const Footer = styled.footer`
  background-color: #1f2937;
  color: white;
  padding: 1rem 0;
  margin-top: 2rem;
`;

const FooterContent = styled.div`
  width: 100%;
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1rem;
  text-align: center;
`;

// Icons
const UploadIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 20 20" fill="currentColor">
    <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM6.293 6.707a1 1 0 010-1.414l3-3a1 1 0 011.414 0l3 3a1 1 0 01-1.414 1.414L11 5.414V13a1 1 0 11-2 0V5.414L7.707 6.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
  </svg>
);

// PrivateRoute component to protect routes that require authentication
const PrivateRoute = ({ children }) => {
  const { isAuthenticated, loading } = useAuth();
  
  // If auth is still loading, don't render anything yet
  if (loading) {
    return <div>Loading...</div>;
  }
  
  // If not authenticated, redirect to login
  if (!isAuthenticated) {
    return <Navigate to="/login" />;
  }
  
  // If authenticated, render the protected component
  return children;
};

function App() {
  return (
    <AuthProvider>
      <Router>
        <AppContainer>
          <Header />
          <MainContent>
            <Routes>
              {/* Public routes */}
              <Route path="/login" element={<Login />} />
              <Route path="/register" element={<Register />} />
              
              {/* Protected routes */}
              <Route 
                path="/dashboard" 
                element={
                  <PrivateRoute>
                    <Dashboard />
                  </PrivateRoute>
                } 
              />
              
              <Route 
                path="/documents" 
                element={
                  <PrivateRoute>
                    <DocumentUpload />
                  </PrivateRoute>
                } 
              />
              
              <Route 
                path="/cars" 
                element={
                  <PrivateRoute>
                    <CarList />
                  </PrivateRoute>
                } 
              />
              
              <Route 
                path="/cars/add" 
                element={
                  <PrivateRoute>
                    <CarForm />
                  </PrivateRoute>
                } 
              />
              
              <Route 
                path="/cars/:id" 
                element={
                  <PrivateRoute>
                    <CarForm viewMode={true} />
                  </PrivateRoute>
                } 
              />
              
              <Route 
                path="/cars/:id/edit" 
                element={
                  <PrivateRoute>
                    <CarForm editMode={true} />
                  </PrivateRoute>
                } 
              />
              
              <Route 
                path="/damage-detection" 
                element={
                  <PrivateRoute>
                    <DamageDetection />
                  </PrivateRoute>
                } 
              />
              
              <Route 
                path="/damage-reports" 
                element={
                  <PrivateRoute>
                    <DamageReportsList />
                  </PrivateRoute>
                } 
              />
              
              <Route 
                path="/damage-reports/:reportId" 
                element={
                  <PrivateRoute>
                    <DamageReport />
                  </PrivateRoute>
                } 
              />
              
              <Route 
                path="/accident-report" 
                element={
                  <PrivateRoute>
                    <NewAccidentReport />
                  </PrivateRoute>
                } 
              />
              
              {/* Redirect root to appropriate place based on auth status */}
              <Route 
                path="/" 
                element={
                  <Navigate to="/dashboard" />
                } 
              />
              
              {/* Original damage detection UI as a fallback */}
              <Route path="/original" element={<OriginalApp />} />
            </Routes>
          </MainContent>
        </AppContainer>
      </Router>
    </AuthProvider>
  );
}

export default App;
