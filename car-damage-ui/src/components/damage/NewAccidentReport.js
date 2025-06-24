import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import { useAuth } from '../../context/AuthContext';
import carService from '../../services/carService';

// Styled components
const Container = styled.div`
  max-width: 800px;
  margin: 2rem auto;
  padding: 0 1rem;
`;

const Title = styled.h1`
  font-size: 1.875rem;
  font-weight: 600;
  color: #1d3557;
  margin-bottom: 2rem;
`;

const Card = styled.div`
  background-color: white;
  border-radius: 0.5rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  padding: 1.5rem;
  margin-bottom: 1.5rem;
`;

const SectionTitle = styled.h2`
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: 1rem;
  color: #1d3557;
`;

const FormGroup = styled.div`
  margin-bottom: 1rem;
`;

const Label = styled.label`
  display: block;
  font-weight: 500;
  color: #374151;
  margin-bottom: 0.5rem;
`;

const Input = styled.input`
  width: 100%;
  padding: 0.5rem;
  border: 1px solid #d1d5db;
  border-radius: 0.375rem;
  font-size: 1rem;
  
  &:focus {
    outline: none;
    border-color: #1d3557;
    box-shadow: 0 0 0 3px rgba(29, 53, 87, 0.1);
  }
`;

const Select = styled.select`
  width: 100%;
  padding: 0.5rem;
  border: 1px solid #d1d5db;
  border-radius: 0.375rem;
  font-size: 1rem;
  background-color: white;
  
  &:focus {
    outline: none;
    border-color: #1d3557;
    box-shadow: 0 0 0 3px rgba(29, 53, 87, 0.1);
  }
`;

const TextArea = styled.textarea`
  width: 100%;
  padding: 0.5rem;
  border: 1px solid #d1d5db;
  border-radius: 0.375rem;
  font-size: 1rem;
  min-height: 100px;
  resize: vertical;
  
  &:focus {
    outline: none;
    border-color: #1d3557;
    box-shadow: 0 0 0 3px rgba(29, 53, 87, 0.1);
  }
`;

const FileInputWrapper = styled.div`
  border: 2px dashed #d1d5db;
  border-radius: 0.5rem;
  padding: 2rem;
  text-align: center;
  transition: border-color 0.2s;
  
  &:hover {
    border-color: #1d3557;
  }
  
  &.drag-over {
    border-color: #1d3557;
    background-color: #f8f9fa;
  }
`;

const FileInput = styled.input`
  display: none;
`;

const FileInputLabel = styled.label`
  cursor: pointer;
  color: #1d3557;
  font-weight: 500;
  
  &:hover {
    text-decoration: underline;
  }
`;

const FilePreview = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 1rem;
  margin-top: 1rem;
`;

const PreviewItem = styled.div`
  position: relative;
  border-radius: 0.375rem;
  overflow: hidden;
  border: 1px solid #e5e7eb;
`;

const PreviewImage = styled.img`
  width: 100%;
  height: 100px;
  object-fit: cover;
`;

const RemoveButton = styled.button`
  position: absolute;
  top: 0.25rem;
  right: 0.25rem;
  background-color: #ef4444;
  color: white;
  border: none;
  border-radius: 50%;
  width: 1.5rem;
  height: 1.5rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  
  &:hover {
    background-color: #dc2626;
  }
`;

const ButtonGroup = styled.div`
  display: flex;
  gap: 1rem;
  justify-content: flex-end;
  margin-top: 2rem;
`;

const Button = styled.button`
  padding: 0.75rem 1.5rem;
  border-radius: 0.375rem;
  font-weight: 500;
  cursor: pointer;
  transition: background-color 0.2s;
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const PrimaryButton = styled(Button)`
  background-color: #1d3557;
  color: white;
  border: none;
  
  &:hover:not(:disabled) {
    background-color: #14253e;
  }
`;

const SecondaryButton = styled(Button)`
  background-color: #e5e7eb;
  color: #374151;
  border: none;
  
  &:hover:not(:disabled) {
    background-color: #d1d5db;
  }
`;

const ErrorMessage = styled.div`
  background-color: #fee2e2;
  border: 1px solid #fecaca;
  color: #dc2626;
  padding: 0.75rem;
  border-radius: 0.375rem;
  margin-bottom: 1rem;
`;

const LoadingSpinner = styled.div`
  display: inline-block;
  width: 1rem;
  height: 1rem;
  border: 2px solid #ffffff;
  border-radius: 50%;
  border-top-color: transparent;
  animation: spin 1s ease-in-out infinite;
  
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
`;

const NewAccidentReport = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [cars, setCars] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [formData, setFormData] = useState({
    carId: '',
    incidentDate: '',
    incidentTime: '',
    location: '',
    description: '',
    damageDescription: '',
    witnesses: '',
    policeReport: false,
    policeReportNumber: '',
    images: []
  });

  useEffect(() => {
    const fetchCars = async () => {
      try {
        const response = await carService.getCars();
        // Handle different response structures
        if (response.success && response.cars) {
          setCars(response.cars);
        } else if (response.data && response.data.cars) {
          setCars(response.data.cars);
        } else if (Array.isArray(response)) {
          setCars(response);
        } else {
          setCars([]);
        }
      } catch (err) {
        console.error('Error fetching cars:', err);
        setError('Failed to load cars. Please try again.');
        setCars([]); // Set empty array on error
      }
    };

    fetchCars();
  }, []);

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files);
    
    files.forEach(file => {
      const reader = new FileReader();
      reader.onload = (event) => {
        setFormData(prev => ({
          ...prev,
          images: [...prev.images, {
            file,
            preview: event.target.result,
            id: Date.now() + Math.random()
          }]
        }));
      };
      reader.readAsDataURL(file);
    });
  };

  const removeImage = (imageId) => {
    setFormData(prev => ({
      ...prev,
      images: prev.images.filter(img => img.id !== imageId)
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      // Validate required fields
      if (!formData.carId || !formData.incidentDate || !formData.location) {
        throw new Error('Please fill in all required fields');
      }

      const reportData = new FormData();
      
      // Add form fields
      Object.keys(formData).forEach(key => {
        if (key !== 'images') {
          reportData.append(key, formData[key]);
        }
      });

      // Add images
      formData.images.forEach((image, index) => {
        reportData.append(`images`, image.file);
      });

      // TODO: Replace with actual API call
      // await damageService.createAccidentReport(reportData);
      
      // Mock API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Navigate to reports list
      navigate('/damage-reports');
    } catch (err) {
      setError(err.message || 'Failed to submit accident report. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = () => {
    navigate(-1);
  };

  return (
    <Container>
      <Title>New Accident Report</Title>
      
      {error && <ErrorMessage>{error}</ErrorMessage>}

      <form onSubmit={handleSubmit}>
        {/* Basic Information */}
        <Card>
          <SectionTitle>Basic Information</SectionTitle>
          
          <FormGroup>
            <Label htmlFor="carId">Vehicle *</Label>
            <Select
              id="carId"
              name="carId"
              value={formData.carId}
              onChange={handleInputChange}
              required
            >
              <option value="">Select a vehicle</option>
              {cars.map(car => (
                <option key={car._id} value={car._id}>
                  {car.make} {car.model} {car.year} ({car.licensePlate})
                </option>
              ))}
            </Select>
          </FormGroup>

          <FormGroup>
            <Label htmlFor="incidentDate">Incident Date *</Label>
            <Input
              type="date"
              id="incidentDate"
              name="incidentDate"
              value={formData.incidentDate}
              onChange={handleInputChange}
              max={new Date().toISOString().split('T')[0]}
              required
            />
          </FormGroup>

          <FormGroup>
            <Label htmlFor="incidentTime">Incident Time</Label>
            <Input
              type="time"
              id="incidentTime"
              name="incidentTime"
              value={formData.incidentTime}
              onChange={handleInputChange}
            />
          </FormGroup>

          <FormGroup>
            <Label htmlFor="location">Location *</Label>
            <Input
              type="text"
              id="location"
              name="location"
              value={formData.location}
              onChange={handleInputChange}
              placeholder="Address or description of incident location"
              required
            />
          </FormGroup>
        </Card>

        {/* Incident Details */}
        <Card>
          <SectionTitle>Incident Details</SectionTitle>
          
          <FormGroup>
            <Label htmlFor="description">Incident Description</Label>
            <TextArea
              id="description"
              name="description"
              value={formData.description}
              onChange={handleInputChange}
              placeholder="Describe what happened during the incident"
            />
          </FormGroup>

          <FormGroup>
            <Label htmlFor="damageDescription">Damage Description</Label>
            <TextArea
              id="damageDescription"
              name="damageDescription"
              value={formData.damageDescription}
              onChange={handleInputChange}
              placeholder="Describe the damage to your vehicle"
            />
          </FormGroup>

          <FormGroup>
            <Label htmlFor="witnesses">Witnesses</Label>
            <TextArea
              id="witnesses"
              name="witnesses"
              value={formData.witnesses}
              onChange={handleInputChange}
              placeholder="Names and contact information of any witnesses"
            />
          </FormGroup>
        </Card>

        {/* Police Report */}
        <Card>
          <SectionTitle>Police Report</SectionTitle>
          
          <FormGroup>
            <Label>
              <Input
                type="checkbox"
                name="policeReport"
                checked={formData.policeReport}
                onChange={handleInputChange}
                style={{ width: 'auto', marginRight: '0.5rem' }}
              />
              Police report filed
            </Label>
          </FormGroup>

          {formData.policeReport && (
            <FormGroup>
              <Label htmlFor="policeReportNumber">Police Report Number</Label>
              <Input
                type="text"
                id="policeReportNumber"
                name="policeReportNumber"
                value={formData.policeReportNumber}
                onChange={handleInputChange}
                placeholder="Enter police report number"
              />
            </FormGroup>
          )}
        </Card>

        {/* Images */}
        <Card>
          <SectionTitle>Images</SectionTitle>
          
          <FileInputWrapper>
            <FileInput
              type="file"
              id="images"
              multiple
              accept="image/*"
              onChange={handleFileChange}
            />
            <FileInputLabel htmlFor="images">
              📷 Click to upload images or drag and drop
            </FileInputLabel>
            <div style={{ marginTop: '0.5rem', color: '#6b7280', fontSize: '0.875rem' }}>
              Upload photos of the damage and accident scene
            </div>
          </FileInputWrapper>

          {formData.images.length > 0 && (
            <FilePreview>
              {formData.images.map(image => (
                <PreviewItem key={image.id}>
                  <PreviewImage src={image.preview} alt="Preview" />
                  <RemoveButton onClick={() => removeImage(image.id)}>
                    ×
                  </RemoveButton>
                </PreviewItem>
              ))}
            </FilePreview>
          )}
        </Card>

        {/* Action Buttons */}
        <ButtonGroup>
          <SecondaryButton type="button" onClick={handleCancel}>
            Cancel
          </SecondaryButton>
          <PrimaryButton type="submit" disabled={loading}>
            {loading ? (
              <>
                <LoadingSpinner /> Submitting...
              </>
            ) : (
              'Submit Report'
            )}
          </PrimaryButton>
        </ButtonGroup>
      </form>
    </Container>
  );
};

export default NewAccidentReport; 