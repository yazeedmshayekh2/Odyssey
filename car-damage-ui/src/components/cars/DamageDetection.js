import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { useAuth } from '../../context/AuthContext';
import carService from '../../services/carService';
import damageService from '../../services/damageService';
import { useNavigate } from 'react-router-dom';

// Styled components
const Container = styled.div`
  max-width: 1000px;
  margin: 2rem auto;
  padding: 0 1rem;
`;

const Title = styled.h2`
  font-size: 1.875rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
  color: #1d3557;
`;

const Subtitle = styled.p`
  color: #6b7280;
  margin-bottom: 2rem;
`;

const Card = styled.div`
  background-color: white;
  border-radius: 0.5rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  padding: 1.5rem;
  margin-bottom: 1.5rem;
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

const Select = styled.select`
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
  background-color: #1d3557;
  border: none;
  padding: 0.75rem 1rem;
  border-radius: 0.375rem;
  color: white;
  font-weight: 500;
  text-align: center;
  margin-bottom: 0.5rem;
  transition: all 0.2s;
  
  &:hover {
    background-color: #14253e;
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
  max-height: 300px;
  display: flex;
  justify-content: center;
  align-items: center;
`;

const PreviewImage = styled.img`
  max-width: 100%;
  max-height: 300px;
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

const ResultsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr;
  gap: 1.5rem;
  
  @media (min-width: 768px) {
    grid-template-columns: 1fr 1fr;
  }
`;

const ResultColumn = styled.div``;

const ResultCard = styled.div.withConfig({
  shouldForwardProp: (prop) => !['light', 'marginBottom'].includes(prop)
})`
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

const ButtonGroup = styled.div`
  display: flex;
  gap: 0.75rem;
`;

const CancelButton = styled.button`
  flex: 1;
  padding: 0.75rem;
  background-color: #f3f4f6;
  color: #4b5563;
  border: 1px solid #d1d5db;
  border-radius: 0.375rem;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  transition: background-color 0.2s;
  
  &:hover {
    background-color: #e5e7eb;
  }
`;

const ErrorMessage = styled.div`
  background-color: #fee2e2;
  color: #b91c1c;
  padding: 0.75rem;
  border-radius: 0.375rem;
  margin-bottom: 1rem;
`;

const LoadingSpinner = styled.div`
  display: inline-block;
  width: 1.5rem;
  height: 1.5rem;
  border: 3px solid rgba(255, 255, 255, 0.3);
  border-radius: 50%;
  border-top-color: white;
  animation: spin 1s ease-in-out infinite;
  margin-right: 0.5rem;
  
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
`;

const PrimaryButton = styled.button`
  padding: 0.75rem 1rem;
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
`;

const SecondaryButton = styled.button`
  padding: 0.75rem 1rem;
  background-color: #f3f4f6;
  color: #4b5563;
  border: 1px solid #d1d5db;
  border-radius: 0.375rem;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  transition: background-color 0.2s;
  
  &:hover {
    background-color: #e5e7eb;
  }
`;

const DamageDetection = () => {
  const [cars, setCars] = useState([]);
  const [selectedCar, setSelectedCar] = useState('');
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [carsLoading, setCarsLoading] = useState(true);
  const [error, setError] = useState('');
  
  const { user } = useAuth();
  const navigate = useNavigate();
  
  // Fetch user's cars
  useEffect(() => {
    const fetchCars = async () => {
      try {
        const response = await carService.getUserCars();
        if (response.success) {
          setCars(response.cars);
          if (response.cars.length > 0) {
            setSelectedCar(response.cars[0].id);
          }
        }
      } catch (err) {
        setError('Error loading your cars. Please try again later.');
      } finally {
        setCarsLoading(false);
      }
    };
    
    fetchCars();
  }, []);
  
  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!imageFile) {
      setError('Please select an image to upload');
      return;
    }
    
    if (!selectedCar) {
      setError('Please select a car first');
      return;
    }
    
    setLoading(true);
    setError('');
    
    try {
      // Get the selected car details
      const selectedCarData = cars.find(car => car.id === selectedCar);
      const carInfo = selectedCarData ? {
        make: selectedCarData.make,
        model: selectedCarData.model,
        year: selectedCarData.year,
        licensePlate: selectedCarData.licensePlate
      } : null;
      
      // Call damage detection API directly with car information (which saves to database)
      const response = await damageService.detectDamage(imageFile, selectedCar, carInfo);
      
      console.log('DEBUG: Component received response:', {
        success: response.success,
        damage_detected: response.damage_detected,
        damage_percentage: response.damage_percentage,
        damage_classes_count: response.damage_classes ? response.damage_classes.length : 0,
        has_result_image: !!response.result_image_data,
        result_image_preview: response.result_image_data ? response.result_image_data.substring(0, 100) + '...' : 'None',
        report_id: response.report_id
      });
      
      // If detection was successful and saved to database, try to create a detailed report entry
      if (response.status === 'success' && response.report_id) {
        try {
          // Create a comprehensive damage report entry via the damage API
          const damageReportData = {
            car_id: selectedCar,
            car_info: carInfo || {
              make: 'Unknown',
              model: 'Unknown',
              year: 'Unknown',
              licensePlate: 'Unknown'
            },
            damage_detected: response.damage_percentage > 0,
            damage_percentage: response.damage_percentage,
            damage_types: response.damage_classes.map(cls => cls.class),
            detected_damages: response.damage_classes,
            confidence_scores: response.damage_classes.reduce((acc, cls) => {
              acc[cls.class] = cls.confidence;
              return acc;
            }, {}),
            images: {
              original_filename: response.original_image,
              result_image_data: response.result_image_data
            },
            status: 'completed',
            notes: `Automated damage detection completed. ${response.damage_classes.length} damage types detected.`
          };
          
          // Save the detailed report via damage API
          await damageService.createDamageReport(damageReportData);
          
          // Update the response to include integration success
          response.integration_success = true;
          response.car_info = damageReportData.car_info;
        } catch (integrationError) {
          console.warn('Failed to create detailed damage report:', integrationError);
          // Don't fail the whole process if integration fails
          response.integration_success = false;
        }
      }
      
      setResult(response);
    } catch (err) {
      setError('Error processing the image. Please try again.');
      console.error('Damage detection error:', err);
    } finally {
      setLoading(false);
    }
  };
  
  const handleReset = () => {
    setImageFile(null);
    setImagePreview(null);
    setResult(null);
    setError('');
  };
  
  const handleCreateReport = () => {
    if (result && selectedCar) {
      // Navigate to accident report form with pre-filled data
      navigate('/accident-report', {
        state: {
          carId: selectedCar,
          damageData: {
            percentage: result.damage_percentage,
            types: result.damage_classes,
            images: [result.result_image_data],
            detectionResults: result
          }
        }
      });
    }
  };

  const handleViewReports = () => {
    navigate('/damage-reports');
  };
  
  return (
    <Container>
      <Title>Car Damage Detection</Title>
      <Subtitle>Upload an image of your car to detect and analyze damaged areas</Subtitle>
      
      <Card>
        <form onSubmit={handleSubmit}>
          <FormGroup>
            <Label htmlFor="car">Select Car</Label>
            <Select 
              id="car" 
              value={selectedCar} 
              onChange={(e) => setSelectedCar(e.target.value)}
              disabled={carsLoading || cars.length === 0}
            >
              {carsLoading ? (
                <option>Loading cars...</option>
              ) : cars.length === 0 ? (
                <option>No cars registered</option>
              ) : (
                cars.map(car => (
                  <option key={car.id} value={car.id}>
                    {car.make} {car.model} ({car.year}) - {car.licensePlate}
                  </option>
                ))
              )}
            </Select>
          </FormGroup>
          
          <FormGroup>
            <Label>Upload Car Image</Label>
            <FileInputWrapper>
              <FileInputButton htmlFor="damageImage">
                Choose Image
              </FileInputButton>
              <FileInput
                type="file"
                id="damageImage"
                accept="image/*"
                onChange={handleImageChange}
              />
              <FileName>
                {imageFile ? imageFile.name : 'No file chosen'}
              </FileName>
              
              {imagePreview && (
                <ImagePreview>
                  <PreviewImage src={imagePreview} alt="Preview" />
                </ImagePreview>
              )}
            </FileInputWrapper>
          </FormGroup>
          
          {error && <ErrorMessage>{error}</ErrorMessage>}
          
          <ButtonGroup>
            <Button type="submit" disabled={loading || !imageFile || cars.length === 0}>
              {loading ? (
                <>
                  <LoadingSpinner /> Processing...
                </>
              ) : 'Detect Damage'}
            </Button>
            
            {imageFile && (
              <CancelButton type="button" onClick={handleReset}>
                Reset
              </CancelButton>
            )}
          </ButtonGroup>
        </form>
      </Card>
      
      {/* Results Section */}
      {result && (
        <Card>
          <ResultTitle>Damage Analysis Results</ResultTitle>
          
          <ResultsGrid>
            <ResultColumn>
              <ResultTitle>Processed Image</ResultTitle>
              <ImageBorder>
                {result.result_image_data ? (
                  <img 
                    src={result.result_image_data} 
                    alt="Processed" 
                    style={{ width: '100%', height: 'auto' }}
                    onError={(e) => {
                      console.error('DEBUG: Image failed to load:', {
                        src: e.target.src ? e.target.src.substring(0, 100) + '...' : 'None',
                        error: e.error
                      });
                    }}
                  />
                ) : (
                  <div style={{ padding: '1rem', textAlign: 'center', color: '#666' }}>
                    No processed image available
                  </div>
                )}
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
                
                {/* Integration Status */}
                {result.report_id && (
                  <div style={{ marginTop: '1rem', padding: '0.75rem', backgroundColor: '#d1fae5', borderRadius: '0.375rem', border: '1px solid #10b981' }}>
                    <p style={{ color: '#047857', fontSize: '0.875rem', margin: 0 }}>
                      ✓ Results saved to your damage reports (ID: {result.report_id})
                    </p>
                  </div>
                )}
                
                {result.integration_success === false && (
                  <div style={{ marginTop: '1rem', padding: '0.75rem', backgroundColor: '#fef3c7', borderRadius: '0.375rem', border: '1px solid #f59e0b' }}>
                    <p style={{ color: '#92400e', fontSize: '0.875rem', margin: 0 }}>
                      ⚠ Detection completed but integration with reports partially failed
                    </p>
                  </div>
                )}
                
                {/* Action Buttons */}
                <div style={{ marginTop: '1rem', display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                  <PrimaryButton onClick={handleCreateReport}>
                    Create Detailed Report
                  </PrimaryButton>
                  <SecondaryButton onClick={handleViewReports}>
                    View All Reports
                  </SecondaryButton>
                  {result.car_info && (
                    <SecondaryButton onClick={() => navigate(`/cars/${selectedCar}`)}>
                      View Car Details
                    </SecondaryButton>
                  )}
                </div>
              </ResultCard>
            </ResultColumn>
          </ResultsGrid>
        </Card>
      )}
    </Container>
  );
};

export default DamageDetection; 