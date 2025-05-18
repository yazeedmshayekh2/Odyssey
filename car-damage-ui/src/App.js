import React, { useState } from 'react';
import axios from 'axios';
import styled from 'styled-components';
import './App.css';

// Styled Components
const AppContainer = styled.div`
  min-height: 100vh;
  background-color: #f8f9fa;
`;

const Header = styled.header`
  background-color: #1d3557;
  color: white;
  padding: 1.5rem 0;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
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

const Main = styled.main`
  width: 100%;
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem 1rem;
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

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Handle file selection
  const handleFileSelect = (e) => {
    setError(null);
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      // Create preview
      const reader = new FileReader();
      reader.onload = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!selectedFile) {
      setError('Please select an image to upload');
      return;
    }

    setIsLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('image', selectedFile);
    
    try {
      const response = await axios.post('http://localhost:5000/api/detect-damage', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      setResult(response.data);
    } catch (error) {
      console.error('Error during damage detection:', error);
      setError(error.response?.data?.message || 'Error processing the image. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle reset
  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <AppContainer>
      <Header>
        <HeaderContent>
          <Title>
            <span style={{ marginRight: '0.5rem' }}>🚗</span> 
            Car Damage Detection
          </Title>
          <Subtitle>Upload a car image to detect and analyze damaged areas</Subtitle>
        </HeaderContent>
      </Header>

      <Main>
        {/* Upload Section */}
        <Card>
          <SectionTitle>Upload Car Image</SectionTitle>
          
          <form onSubmit={handleSubmit}>
            <FormGroup>
              <Label>Select an image</Label>
              <FileInputWrapper>
                <FileInputButton>
                  <span>Choose File</span>
                  <input 
                    type="file" 
                    style={{ display: 'none' }} 
                    accept="image/*" 
                    onChange={handleFileSelect} 
                  />
                </FileInputButton>
                <FileInputText>
                  {selectedFile ? selectedFile.name : 'No file chosen'}
                </FileInputText>
              </FileInputWrapper>
            </FormGroup>

            {preview && (
              <PreviewContainer>
                <Label>Preview:</Label>
                <div style={{ maxWidth: '400px' }}>
                  <PreviewImage 
                    src={preview} 
                    alt="Preview" 
                  />
                </div>
              </PreviewContainer>
            )}

            {error && (
              <ErrorMessage>
                <p>{error}</p>
              </ErrorMessage>
            )}

            <ButtonGroup>
              <PrimaryButton 
                type="submit" 
                disabled={isLoading || !selectedFile}
              >
                {isLoading ? 'Processing...' : 'Detect Damage'}
              </PrimaryButton>

              {(selectedFile || result) && (
                <SecondaryButton
                  type="button"
                  onClick={handleReset}
                >
                  Reset
                </SecondaryButton>
              )}
            </ButtonGroup>
          </form>
        </Card>

        {/* Results Section */}
        {result && (
          <Card>
            <SectionTitle>Damage Analysis Results</SectionTitle>
            
            <ResultsGrid>
              <ResultColumn>
                <ResultTitle>Processed Image</ResultTitle>
                <ImageBorder>
                  <img 
                    src={result.image_data} 
                    alt="Processed" 
                    style={{ width: '100%', height: 'auto' }}
                  />
                </ImageBorder>
              </ResultColumn>
              
              <ResultColumn>
                <ResultCard light marginBottom>
                  <ResultTitle>Damage Information</ResultTitle>
                  
                  <div style={{ marginBottom: '1rem' }}>
                    <DamagePercentage>
                      {result.damage_percentage.toFixed(2)}%
                    </DamagePercentage>
                    <DamageLabel>Damaged Area</DamageLabel>
                  </div>
                  
                  <div>
                    <DetectedIssuesTitle>Detected Issues:</DetectedIssuesTitle>
                    <IssuesList>
                      {result.damage_classes.map((item, index) => (
                        <IssueItem key={index}>
                          <IssueDot />
                          <IssueText>{item.class}</IssueText>
                          <IssueConfidence>
                            (Confidence: {(item.confidence * 100).toFixed(1)}%)
                          </IssueConfidence>
                        </IssueItem>
                      ))}
                    </IssuesList>
                  </div>
                </ResultCard>

                <ResultCard>
                  <ResultTitle>Repair Recommendation</ResultTitle>
                  <p style={{ color: '#6b7280' }}>
                    Based on the damage detection, this vehicle requires professional repair services.
                    {result.damage_percentage > 20 
                      ? ' The extensive damage suggests significant repair work is needed.' 
                      : ' Minor repairs should be sufficient to address the damage.'}
                  </p>
                </ResultCard>
              </ResultColumn>
            </ResultsGrid>
          </Card>
        )}

        {/* Upload placeholder when no image selected */}
        {!preview && !result && (
          <UploadPlaceholder>
            <PlaceholderContent>
              <UploadIcon />
              <p style={{ marginTop: '0.5rem' }}>Select a car image to start damage detection</p>
            </PlaceholderContent>
          </UploadPlaceholder>
        )}
      </Main>

      <Footer>
        <FooterContent>
          <p>&copy; {new Date().getFullYear()} Car Damage Detection System</p>
        </FooterContent>
      </Footer>
    </AppContainer>
  );
}

export default App;
