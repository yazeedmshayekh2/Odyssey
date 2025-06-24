import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import { useAuth } from '../../context/AuthContext';

// Styled components
const UploadContainer = styled.div`
  max-width: 700px;
  margin: 2rem auto;
  padding: 2rem;
  background-color: white;
  border-radius: 0.5rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
`;

const Title = styled.h2`
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
  color: #1d3557;
  text-align: center;
`;

const Subtitle = styled.p`
  text-align: center;
  color: #6b7280;
  margin-bottom: 2rem;
`;

const FormGroup = styled.div`
  margin-bottom: 1.5rem;
`;

const Label = styled.label`
  display: block;
  margin-bottom: 0.5rem;
  color: #4b5563;
  font-weight: 500;
`;

const FileInputWrapper = styled.div`
  display: flex;
  flex-direction: column;
`;

const FileInput = styled.input`
  display: none;
`;

const FileInputButton = styled.label`
  display: inline-block;
  cursor: pointer;
  background-color: #f3f4f6;
  border: 1px solid #d1d5db;
  padding: 0.75rem 1rem;
  border-radius: 0.375rem;
  color: #4b5563;
  font-weight: 500;
  text-align: center;
  margin-bottom: 0.5rem;
  transition: all 0.2s;
  
  &:hover {
    background-color: #e5e7eb;
  }
`;

const FileName = styled.span`
  color: #4b5563;
  font-size: 0.875rem;
`;

const ImagePreview = styled.div`
  margin-top: 1rem;
  border: 1px solid #e5e7eb;
  border-radius: 0.375rem;
  overflow: hidden;
  width: 100%;
  max-height: 200px;
  display: flex;
  justify-content: center;
  align-items: center;
`;

const PreviewImage = styled.img`
  max-width: 100%;
  max-height: 200px;
  object-fit: contain;
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

const DocumentRow = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const DocumentUpload = () => {
  const [licenseFile, setLicenseFile] = useState(null);
  const [carFile, setCarFile] = useState(null);
  const [licensePreview, setLicensePreview] = useState(null);
  const [carPreview, setCarPreview] = useState(null);
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  const { uploadDocuments } = useAuth();
  const navigate = useNavigate();
  
  const handleLicenseChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setLicenseFile(file);
      
      // Create preview URL
      const reader = new FileReader();
      reader.onloadend = () => {
        setLicensePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };
  
  const handleCarChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setCarFile(file);
      
      // Create preview URL
      const reader = new FileReader();
      reader.onloadend = () => {
        setCarPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validate files
    if (!licenseFile || !carFile) {
      setError('Both documents are required');
      return;
    }
    
    setError('');
    setIsSubmitting(true);
    
    try {
      const result = await uploadDocuments(licenseFile, carFile);
      if (result.success) {
        navigate('/dashboard');
      } else {
        setError(result.message || 'Document upload failed');
      }
    } catch (err) {
      setError(err.message || 'An error occurred during document upload');
    } finally {
      setIsSubmitting(false);
    }
  };
  
  return (
    <UploadContainer>
      <Title>Upload Your Documents</Title>
      <Subtitle>Please upload your driver's license and vehicle photos</Subtitle>
      
      {error && <ErrorMessage>{error}</ErrorMessage>}
      
      <form onSubmit={handleSubmit}>
        <DocumentRow>
          <FormGroup>
            <Label>Driver's License</Label>
            <FileInputWrapper>
              <FileInputButton htmlFor="licenseFile">
                Choose Driver's License Image
              </FileInputButton>
              <FileInput
                type="file"
                id="licenseFile"
                accept="image/*"
                onChange={handleLicenseChange}
              />
              <FileName>
                {licenseFile ? licenseFile.name : 'No file chosen'}
              </FileName>
              
              {licensePreview && (
                <ImagePreview>
                  <PreviewImage src={licensePreview} alt="License preview" />
                </ImagePreview>
              )}
            </FileInputWrapper>
          </FormGroup>
          
          <FormGroup>
            <Label>Car Photo</Label>
            <FileInputWrapper>
              <FileInputButton htmlFor="carFile">
                Choose Car Image
              </FileInputButton>
              <FileInput
                type="file"
                id="carFile"
                accept="image/*"
                onChange={handleCarChange}
              />
              <FileName>
                {carFile ? carFile.name : 'No file chosen'}
              </FileName>
              
              {carPreview && (
                <ImagePreview>
                  <PreviewImage src={carPreview} alt="Car preview" />
                </ImagePreview>
              )}
            </FileInputWrapper>
          </FormGroup>
        </DocumentRow>
        
        <Button type="submit" disabled={isSubmitting || !licenseFile || !carFile}>
          {isSubmitting ? 'Uploading...' : 'Upload Documents'}
        </Button>
      </form>
    </UploadContainer>
  );
};

export default DocumentUpload; 