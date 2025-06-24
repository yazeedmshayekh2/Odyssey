import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import styled from 'styled-components';
import carService from '../../services/carService';

// Styled components
const FormContainer = styled.div`
  max-width: 800px;
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
`;

// New styled components for improved view mode
const ViewContainer = styled.div`
  max-width: 1000px;
  margin: 2rem auto;
  background-color: white;
  border-radius: 0.5rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  overflow: hidden;
`;

const CarHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.5rem 2rem;
  background-color: #1d3557;
  color: white;
`;

const CarTitle = styled.h2`
  font-size: 1.75rem;
  font-weight: 600;
  margin: 0;
`;

const CarSubtitle = styled.p`
  font-size: 1rem;
  margin: 0.25rem 0 0 0;
  opacity: 0.8;
`;

const CarImageContainer = styled.div`
  width: 100%;
  height: 350px;
  background-color: #f3f4f6;
  overflow: hidden;
  position: relative;
`;

const CarImage = styled.img`
  width: 100%;
  height: 100%;
  object-fit: cover;
  object-position: center;
`;

const NoImagePlaceholder = styled.div`
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #9ca3af;
  font-size: 1.25rem;
  background-color: #f8f9fa;
  border-radius: 8px;
  text-align: center;
  padding: 0 20px;
`;

const CarIcon = styled.div`
  font-size: 4rem;
  margin-bottom: 1rem;
  color: #cbd5e1;
`;

const CarMakeModel = styled.div`
  font-size: 1.5rem;
  font-weight: 600;
  color: #64748b;
  margin-bottom: 0.5rem;
`;

const CarYear = styled.div`
  font-size: 1rem;
  color: #94a3b8;
`;

const CarDetails = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
  padding: 2rem;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const DetailSection = styled.div`
  margin-bottom: 1.5rem;
`;

const SectionTitle = styled.h3`
  font-size: 1.25rem;
  font-weight: 500;
  margin-bottom: 1rem;
  color: #1d3557;
  border-bottom: 2px solid #e5e7eb;
  padding-bottom: 0.5rem;
`;

const DetailItem = styled.div`
  display: flex;
  margin-bottom: 0.75rem;
`;

const DetailLabel = styled.div`
  font-weight: 500;
  color: #4b5563;
  width: 40%;
`;

const DetailValue = styled.div`
  color: #1f2937;
  width: 60%;
`;

const InsuredPartsList = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 0.5rem;
`;

const InsuredPart = styled.div`
  background-color: #e0f2fe;
  border: 1px solid #bae6fd;
  color: #0369a1;
  border-radius: 0.375rem;
  padding: 0.25rem 0.75rem;
  font-size: 0.875rem;
`;

const ActionButtons = styled.div`
  display: flex;
  justify-content: flex-end;
  gap: 1rem;
  padding: 1rem 2rem 2rem;
`;

// Original styled components
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

const CheckboxGroup = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.5rem;
  margin-top: 0.5rem;
  
  @media (min-width: 768px) {
    grid-template-columns: repeat(3, 1fr);
  }
`;

const CheckboxLabel = styled.label`
  display: flex;
  align-items: center;
  font-size: 0.875rem;
  color: #4b5563;
`;

const Checkbox = styled.input`
  margin-right: 0.5rem;
`;

const FileInputLabel = styled.label`
  display: block;
  margin-bottom: 0.5rem;
  color: #4b5563;
  font-weight: 500;
`;

const FileInputButton = styled.label`
  display: inline-block;
  cursor: pointer;
  background-color: #1d3557;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 0.375rem;
  color: white;
  font-weight: 500;
  margin-bottom: 0.5rem;
  
  &:hover {
    background-color: #14253e;
  }
`;

const FileInput = styled.input`
  display: none;
`;

const FileName = styled.span`
  display: block;
  color: #4b5563;
  font-size: 0.875rem;
  margin-bottom: 0.5rem;
`;

const ImagePreview = styled.div`
  margin-top: 0.5rem;
  max-width: 300px;
  border: 1px solid #d1d5db;
  border-radius: 0.375rem;
  overflow: hidden;
`;

const PreviewImage = styled.img`
  width: 100%;
  height: auto;
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
  margin-top: 1rem;
  
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

const ButtonGroup = styled.div`
  display: flex;
  gap: 1rem;
  margin-top: 1rem;
`;

const SecondaryButton = styled(Button)`
  background-color: #6b7280;
  
  &:hover {
    background-color: #4b5563;
  }
`;

const LicensePlateWarning = styled.div`
  color: #b91c1c;
  font-size: 0.875rem;
  margin-top: 0.25rem;
`;

// Add new component for direct img use
const DirectImage = ({ src, alt, className }) => {
  const [imgSrc, setImgSrc] = React.useState(src);
  const [error, setError] = React.useState(false);
  
  // When src changes, update imgSrc and reset error
  React.useEffect(() => {
    setImgSrc(src);
    setError(false);
  }, [src]);
  
  const handleError = () => {
    console.error('Image failed to load:', imgSrc);
    
    // Try alternative formats if src doesn't include http
    if (!error && !imgSrc.startsWith('data:')) {
      setError(true);
      
      // Try with /static prefix
      if (!imgSrc.includes('/static/')) {
        const baseUrl = 'http://localhost:5000';
        const imagePath = imgSrc.replace(baseUrl, '');
        const newSrc = `${baseUrl}/static${imagePath.startsWith('/') ? imagePath : `/${imagePath}`}`;
        console.log('Trying alternative path:', newSrc);
        setImgSrc(newSrc);
        return;
      }
    }
    
    // Fallback to placeholder if all attempts fail
    setImgSrc("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100' viewBox='0 0 100 100'%3E%3Crect width='100' height='100' fill='%23f3f4f6'/%3E%3Ctext x='50' y='50' font-family='Arial' font-size='14' text-anchor='middle' fill='%239ca3af'%3ENo Image%3C/text%3E%3C/svg%3E");
  };
  
  return (
    <img 
      src={imgSrc} 
      alt={alt} 
      className={className}
      style={{ width: '100%', height: '100%', objectFit: 'cover' }}
      onError={handleError}
    />
  );
};

const CarForm = ({ viewMode, editMode }) => {
  const { id } = useParams();
  const [formData, setFormData] = useState({
    make: '',
    model: '',
    year: new Date().getFullYear(),
    color: '',
    licensePlate: '',
    weight: 0,
    totalWeight: 0,
    seats: 0,
    engineType: '', 
    countryOrigin: '',
    chassisNo: '',
    engineNo: '',
    insuranceExpiry: '',
    insurancePolicy: '',
    insuredParts: []
  });
  
  const [carImage, setCarImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  
  const [carParts, setCarParts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  const [licensePlateExists, setLicensePlateExists] = useState(false);
  const [checkingLicensePlate, setCheckingLicensePlate] = useState(false);
  const [existingCar, setExistingCar] = useState(null);
  
  const navigate = useNavigate();
  
  // Generate years array for dropdown
  const currentYear = new Date().getFullYear();
  const years = Array.from({ length: 50 }, (_, i) => currentYear - i);
  
  // Fetch car parts on component mount
  useEffect(() => {
    const fetchCarParts = async () => {
      try {
        const response = await carService.getCarParts();
        if (response.success) {
          setCarParts(response.parts);
        }
      } catch (err) {
        console.error('Error fetching car parts:', err);
      }
    };
    
    fetchCarParts();
  }, []);
  
  // Fetch car details if in view or edit mode
  useEffect(() => {
    if (id && (viewMode || editMode)) {
      const fetchCarDetails = async () => {
        try {
          setLoading(true);
          console.log('Fetching car details for ID:', id);
          
          const response = await carService.getCarById(id);
          console.log('Car details response:', response);
          
          if (response.success) {
            // Map API response to form data
            const car = response.car;
            setFormData({
              make: car.make || '',
              model: car.model || '',
              year: car.year || new Date().getFullYear(),
              color: car.color || '',
              licensePlate: car.licensePlate || '',
              weight: car.weight || 0,
              totalWeight: car.totalWeight || 0,
              seats: car.seats || 0,
              engineType: car.engineType || '', 
              countryOrigin: car.countryOrigin || '',
              chassisNo: car.chassisNo || '',
              engineNo: car.engineNo || '',
              insuranceExpiry: car.insuranceExpiry || '',
              insurancePolicy: car.insurancePolicy || '',
              insuredParts: car.insuredParts || []
            });
            
            // Debug the images array
            console.log('Raw car images data:', car.images);
            
            // Set image preview if car has images
            if (car.images && car.images.length > 0) {
              // Check if the image path is a full URL or needs to be constructed
              const imagePath = car.images[0];
              console.log('Original image path from API:', imagePath);
              
              if (typeof imagePath === 'string') {
                // Try different path formats
                if (!imagePath.startsWith('http') && !imagePath.startsWith('data:')) {
                  // Add trailing slash to base URL if needed
                  const baseUrl = 'http://localhost:5000';
                  
                  // Try relative path format first
                  const staticPath = imagePath.startsWith('/') 
                    ? `${baseUrl}${imagePath}` 
                    : `${baseUrl}/${imagePath}`;
                  
                  console.log('Setting image preview to:', staticPath);
                  setImagePreview(staticPath);
                } else {
                  setImagePreview(imagePath);
                }
                
                // Log success message - this helps debugging timing
                console.log('Image preview set to:', imagePath);
              } else {
                console.error('Unexpected image path format:', imagePath);
              }
            } else {
              console.log('No images found for this car');
            }
          } else {
            setError('Failed to load car details.');
          }
        } catch (err) {
          console.error('Error fetching car details:', err);
          setError('Failed to load car details. Please try again later.');
        } finally {
          setLoading(false);
        }
      };
      
      fetchCarDetails();
    }
  }, [id, viewMode, editMode]);
  
  // Function to check license plate with debounce
  const checkLicensePlate = useCallback(async (plate) => {
    if (!plate || plate.length < 2) {
      setLicensePlateExists(false);
      setExistingCar(null);
      return;
    }
    
    try {
      setCheckingLicensePlate(true);
      const result = await carService.checkLicensePlateExists(plate, editMode ? id : null);
      setLicensePlateExists(result.exists);
      setExistingCar(result.car);
    } catch (err) {
      console.error('Error checking license plate:', err);
    } finally {
      setCheckingLicensePlate(false);
    }
  }, [id, editMode]);
  
  // Debounce license plate check to avoid too many API calls
  useEffect(() => {
    if (viewMode) return; // Don't check in view mode
    
    const timer = setTimeout(() => {
      checkLicensePlate(formData.licensePlate);
    }, 500); // 500ms delay
    
    return () => clearTimeout(timer);
  }, [formData.licensePlate, checkLicensePlate, viewMode]);
  
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };
  
  const handleCheckboxChange = (e) => {
    const { value, checked } = e.target;
    
    if (checked) {
      setFormData(prev => ({
        ...prev,
        insuredParts: [...prev.insuredParts, value]
      }));
    } else {
      setFormData(prev => ({
        ...prev,
        insuredParts: prev.insuredParts.filter(part => part !== value)
      }));
    }
  };
  
  const handleImageChange = (e) => {
    const { files } = e.target;
    if (files.length > 0) {
      const file = files[0];
      
      // Update image file
      setCarImage(file);
      
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
    
    // Basic validation
    if (!formData.make || !formData.model || !formData.licensePlate) {
      setError('Make, model, and license plate are required.');
      return;
    }
    
    setLoading(true);
    setError('');
    
    try {
      // Check if license plate is already in use (exclude current car if editing)
      const plateCheck = await carService.checkLicensePlateExists(
        formData.licensePlate, 
        editMode ? id : null
      );
      
      if (plateCheck.exists) {
        setError(`License plate "${formData.licensePlate}" is already in use by another car (${plateCheck.car.make} ${plateCheck.car.model}). Please use a different license plate.`);
        setLoading(false);
        return;
      }
      
      // Test API connection
      console.log("Testing API connection with fetch...");
      const testResponse = await fetch('http://localhost:5000/api/health');
      const testData = await testResponse.json();
      console.log("Health check response:", testData);
      
      const imageData = carImage ? { carImage } : {};
      let response;
      
      if (editMode && id) {
        // Update existing car
        console.log('Updating car:', id);
        response = await carService.updateCar(id, formData, imageData);
      } else {
        // Register new car
        console.log('Registering new car');
        response = await carService.registerCar(formData, imageData);
      }
      
      console.log("Response received:", response);
      
      if (response.success) {
        navigate('/cars');
      } else {
        setError(response.message || `Failed to ${editMode ? 'update' : 'register'} car.`);
      }
    } catch (err) {
      console.error(`Error ${editMode ? 'updating' : 'registering'} car:`, err);
      setError(err.message || 'Network error. Please try again.');
    } finally {
      setLoading(false);
    }
  };
  
  const handleCancel = () => {
    navigate('/cars');
  };
  
  if (loading) {
    return <div>Loading...</div>;
  }
  
  // Determine title based on mode
  const title = viewMode ? 'Car Details' : (editMode ? 'Edit Car' : 'Register New Car');
  
  // Render function
  return viewMode ? (
    <ViewContainer>
      <CarHeader>
        <div>
          <CarTitle>{formData.make} {formData.model}</CarTitle>
          <CarSubtitle>{formData.year} • {formData.licensePlate}</CarSubtitle>
        </div>
        <Button onClick={() => navigate(`/cars/${id}/edit`)}>
          Edit Car
        </Button>
      </CarHeader>
      
      <CarImageContainer>
        <NoImagePlaceholder>
          <CarIcon>🚗</CarIcon>
          <CarMakeModel>{formData.make} {formData.model}</CarMakeModel>
          <CarYear>{formData.year} • {formData.color}</CarYear>
        </NoImagePlaceholder>
      </CarImageContainer>
      
      <CarDetails>
        <div>
          <DetailSection>
            <SectionTitle>Basic Information</SectionTitle>
            <DetailItem>
              <DetailLabel>Make</DetailLabel>
              <DetailValue>{formData.make}</DetailValue>
            </DetailItem>
            <DetailItem>
              <DetailLabel>Model</DetailLabel>
              <DetailValue>{formData.model}</DetailValue>
            </DetailItem>
            <DetailItem>
              <DetailLabel>Year</DetailLabel>
              <DetailValue>{formData.year}</DetailValue>
            </DetailItem>
            <DetailItem>
              <DetailLabel>Color</DetailLabel>
              <DetailValue>{formData.color}</DetailValue>
            </DetailItem>
            <DetailItem>
              <DetailLabel>License Plate</DetailLabel>
              <DetailValue>{formData.licensePlate}</DetailValue>
            </DetailItem>
          </DetailSection>
          
          <DetailSection>
            <SectionTitle>Technical Specifications</SectionTitle>
            <DetailItem>
              <DetailLabel>Chassis No.</DetailLabel>
              <DetailValue>{formData.chassisNo || 'N/A'}</DetailValue>
            </DetailItem>
            <DetailItem>
              <DetailLabel>Engine No.</DetailLabel>
              <DetailValue>{formData.engineNo || 'N/A'}</DetailValue>
            </DetailItem>
            <DetailItem>
              <DetailLabel>Engine Type</DetailLabel>
              <DetailValue>{formData.engineType || 'N/A'}</DetailValue>
            </DetailItem>
          </DetailSection>
        </div>
        
        <div>
          <DetailSection>
            <SectionTitle>Specifications</SectionTitle>
            <DetailItem>
              <DetailLabel>Seats</DetailLabel>
              <DetailValue>{formData.seats}</DetailValue>
            </DetailItem>
            <DetailItem>
              <DetailLabel>Weight</DetailLabel>
              <DetailValue>{formData.weight} kg</DetailValue>
            </DetailItem>
            <DetailItem>
              <DetailLabel>Total Weight</DetailLabel>
              <DetailValue>{formData.totalWeight} kg</DetailValue>
            </DetailItem>
            <DetailItem>
              <DetailLabel>Country of Origin</DetailLabel>
              <DetailValue>{formData.countryOrigin || 'N/A'}</DetailValue>
            </DetailItem>
          </DetailSection>
          
          <DetailSection>
            <SectionTitle>Insurance Information</SectionTitle>
            <DetailItem>
              <DetailLabel>Policy Number</DetailLabel>
              <DetailValue>{formData.insurancePolicy || 'N/A'}</DetailValue>
            </DetailItem>
            <DetailItem>
              <DetailLabel>Expiry Date</DetailLabel>
              <DetailValue>
                {formData.insuranceExpiry ? new Date(formData.insuranceExpiry).toLocaleDateString() : 'N/A'}
              </DetailValue>
            </DetailItem>
            <DetailItem>
              <DetailLabel>Insured Parts</DetailLabel>
              <DetailValue>
                {formData.insuredParts && formData.insuredParts.length > 0 ? (
                  <InsuredPartsList>
                    {formData.insuredParts.map(part => (
                      <InsuredPart key={part}>{part}</InsuredPart>
                    ))}
                  </InsuredPartsList>
                ) : (
                  'No insured parts'
                )}
              </DetailValue>
            </DetailItem>
          </DetailSection>
        </div>
      </CarDetails>
      
      <ActionButtons>
        <SecondaryButton onClick={handleCancel}>
          Back to Cars
        </SecondaryButton>
        <Button onClick={() => navigate('/damage-detection')}>
          Report Damage
        </Button>
      </ActionButtons>
    </ViewContainer>
  ) : (
    <FormContainer>
      <Title>{title}</Title>
      
      {error && <ErrorMessage>{error}</ErrorMessage>}
      
      <form onSubmit={handleSubmit}>
        <FormGroup>
          <Label htmlFor="make">Make</Label>
          <Input
            type="text"
            id="make"
            name="make"
            value={formData.make}
            onChange={handleChange}
            placeholder="Enter car make (e.g., Toyota, Honda)"
            required
            readOnly={viewMode}
          />
        </FormGroup>
        
        <FormGroup>
          <Label htmlFor="model">Model</Label>
          <Input
            type="text"
            id="model"
            name="model"
            value={formData.model}
            onChange={handleChange}
            placeholder="Enter car model (e.g., Camry, Civic)"
            required
            readOnly={viewMode}
          />
        </FormGroup>
        
        <FormGroup>
          <Label htmlFor="year">Year</Label>
          {viewMode ? (
            <Input 
              type="text" 
              value={formData.year} 
              readOnly 
            />
          ) : (
            <Select
              id="year"
              name="year"
              value={formData.year}
              onChange={handleChange}
              required
              disabled={viewMode}
            >
              {years.map(year => (
                <option key={year} value={year}>{year}</option>
              ))}
            </Select>
          )}
        </FormGroup>
        
        <FormGroup>
          <Label htmlFor="color">Color</Label>
          <Input
            type="text"
            id="color"
            name="color"
            value={formData.color}
            onChange={handleChange}
            placeholder="Enter car color"
            required
            readOnly={viewMode}
          />
        </FormGroup>
        
        <FormGroup>
          <Label htmlFor="licensePlate">License Plate</Label>
          <Input
            type="text"
            id="licensePlate"
            name="licensePlate"
            value={formData.licensePlate}
            onChange={handleChange}
            placeholder="Enter license plate number"
            required
            readOnly={viewMode}
          />
          {!viewMode && licensePlateExists && (
            <LicensePlateWarning>
              This license plate is already in use by another car ({existingCar?.make} {existingCar?.model})
            </LicensePlateWarning>
          )}
        </FormGroup>
        
        <FormGroup>
          <Label>Insured Parts</Label>
          {viewMode ? (
            <div style={{ marginTop: '8px' }}>
              {formData.insuredParts.length > 0 ? 
                formData.insuredParts.map(part => (
                  <div key={part} style={{ marginBottom: '4px' }}>{part}</div>
                )) : 
                <div style={{ color: '#6b7280' }}>No insured parts selected</div>
              }
            </div>
          ) : (
            <CheckboxGroup>
              {carParts.map(part => (
                <CheckboxLabel key={part}>
                  <Checkbox
                    type="checkbox"
                    value={part}
                    checked={formData.insuredParts.includes(part)}
                    onChange={handleCheckboxChange}
                    disabled={viewMode}
                  />
                  {part}
                </CheckboxLabel>
              ))}
            </CheckboxGroup>
          )}
        </FormGroup>
        
        <FormGroup>
          <FileInputLabel>Car Image</FileInputLabel>
          {viewMode ? (
            <div>
              {imagePreview ? (
                <ImagePreview>
                  <PreviewImage src={imagePreview} alt="Car image" />
                </ImagePreview>
              ) : (
                <div style={{ color: '#6b7280', marginTop: '8px' }}>
                  No image available
                </div>
              )}
            </div>
          ) : (
            <>
              <FileInputButton htmlFor="carImage">Choose Image</FileInputButton>
              <FileInput
                type="file"
                id="carImage"
                name="carImage"
                accept="image/*"
                onChange={handleImageChange}
              />
              <FileName>{carImage ? carImage.name : 'No file chosen'}</FileName>
              
              {imagePreview && (
                <ImagePreview>
                  <PreviewImage src={imagePreview} alt="Car image" />
                </ImagePreview>
              )}
            </>
          )}
        </FormGroup>
        
        <FormGroup>
          <Label htmlFor="chassisNo">Chassis No.</Label>
          <Input
            type="text"
            id="chassisNo"
            name="chassisNo"
            value={formData.chassisNo}
            onChange={handleChange}
            placeholder="Enter chassis number"
            required
            readOnly={viewMode}
          />
        </FormGroup>
        
        <FormGroup>
          <Label htmlFor="engineNo">Engine No.</Label>
          <Input
            type="text"
            id="engineNo"
            name="engineNo"
            value={formData.engineNo}
            onChange={handleChange}
            placeholder="Enter engine number"
            readOnly={viewMode}
          />
        </FormGroup>
        
        <FormGroup>
          <Label htmlFor="seats">Number of Seats</Label>
          <Input
            type="number"
            id="seats"
            name="seats"
            value={formData.seats}
            onChange={handleChange}
            min="1"
            max="9"
            readOnly={viewMode}
          />
        </FormGroup>
        
        <FormGroup>
          <Label htmlFor="weight">Weight (kg)</Label>
          <Input
            type="number"
            id="weight"
            name="weight"
            value={formData.weight}
            onChange={handleChange}
            min="0"
            readOnly={viewMode}
          />
        </FormGroup>
        
        <FormGroup>
          <Label htmlFor="totalWeight">Total Weight (kg)</Label>
          <Input
            type="number"
            id="totalWeight"
            name="totalWeight"
            value={formData.totalWeight}
            onChange={handleChange}
            min="0"
            readOnly={viewMode}
          />
        </FormGroup>
        
        <FormGroup>
          <Label htmlFor="countryOrigin">Country of Origin</Label>
          <Input
            type="text"
            id="countryOrigin"
            name="countryOrigin"
            value={formData.countryOrigin}
            onChange={handleChange}
            readOnly={viewMode}
          />
        </FormGroup>
        
        <FormGroup>
          <Label htmlFor="engineType">Engine Type</Label>
          <Input
            type="text"
            id="engineType"
            name="engineType"
            value={formData.engineType}
            onChange={handleChange}
            readOnly={viewMode}
          />
        </FormGroup>
        
        <FormGroup>
          <Label htmlFor="insuranceExpiry">Insurance Expiry Date</Label>
          <Input
            type="date"
            id="insuranceExpiry"
            name="insuranceExpiry"
            value={formData.insuranceExpiry}
            onChange={handleChange}
            readOnly={viewMode}
          />
        </FormGroup>
        
        <FormGroup>
          <Label htmlFor="insurancePolicy">Insurance Policy Number</Label>
          <Input
            type="text"
            id="insurancePolicy"
            name="insurancePolicy"
            value={formData.insurancePolicy}
            onChange={handleChange}
            readOnly={viewMode}
          />
        </FormGroup>
        
        <ButtonGroup>
          <SecondaryButton type="button" onClick={handleCancel}>
            Back
          </SecondaryButton>
          {!viewMode && (
            <Button type="submit" disabled={loading}>
              {loading ? (editMode ? 'Updating...' : 'Registering...') : (editMode ? 'Update Car' : 'Register Car')}
            </Button>
          )}
        </ButtonGroup>
      </form>
    </FormContainer>
  );
};

export default CarForm; 