import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import carService from '../../services/carService';
import DamageIndicator from '../damage/DamageIndicator';

// DirectImage component for better image handling
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

// Styled components
const CarListContainer = styled.div`
  max-width: 1000px;
  margin: 2rem auto;
  padding: 0 1rem;
`;

const Title = styled.h2`
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 1rem;
  color: #1d3557;
`;

const AddCarButton = styled(Link)`
  background-color: #1d3557;
  color: white;
  text-decoration: none;
  padding: 0.5rem 1rem;
  border-radius: 0.375rem;
  font-size: 0.875rem;
  display: inline-flex;
  align-items: center;
  margin-bottom: 1.5rem;
  
  &:hover {
    background-color: #14253e;
  }
`;

const PlusIcon = styled.span`
  margin-right: 0.375rem;
  font-size: 1.125rem;
`;

const CarsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 1.5rem;
`;

const CarCard = styled.div`
  background-color: white;
  border-radius: 0.5rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  overflow: hidden;
  transition: transform 0.2s, box-shadow 0.2s;
  
  &:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
  }
`;

const CarImage = styled.div`
  height: 150px;
  background-color: #f3f4f6;
  position: relative;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
`;

const CarImagePlaceholder = styled.div`
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #9ca3af;
  background-color: #f8f9fa;
  padding: 1rem;
  text-align: center;
`;

const CarIcon = styled.div`
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
  color: #cbd5e1;
`;

const CarPlaceholderText = styled.div`
  font-size: 0.9rem;
  color: #94a3b8;
`;

const CarDetails = styled.div`
  padding: 1rem;
`;

const CarName = styled.h3`
  font-size: 1.125rem;
  font-weight: 600;
  color: #1d3557;
  margin-bottom: 0.25rem;
`;

const CarInfo = styled.div`
  color: #6b7280;
  font-size: 0.875rem;
  margin-bottom: 0.75rem;
`;

const CarPlate = styled.div`
  background-color: #f1faee;
  color: #1d3557;
  padding: 0.25rem 0.5rem;
  border-radius: 0.25rem;
  font-size: 0.75rem;
  font-weight: 600;
  display: inline-block;
  margin-bottom: 0.75rem;
`;

const ButtonRow = styled.div`
  display: flex;
  gap: 0.5rem;
  margin-top: 0.75rem;
`;

const Button = styled.button`
  flex: 1;
  padding: 0.5rem;
  font-size: 0.75rem;
  border: none;
  border-radius: 0.25rem;
  cursor: pointer;
  
  &:hover {
    opacity: 0.9;
  }
`;

const ViewButton = styled(Button)`
  background-color: #457b9d;
  color: white;
`;

const EditButton = styled(Button)`
  background-color: #e9c46a;
  color: #000;
`;

const DeleteButton = styled(Button)`
  background-color: #e63946;
  color: white;
`;

const EmptyState = styled.div`
  text-align: center;
  padding: 3rem;
  background-color: white;
  border-radius: 0.5rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
`;

const EmptyIcon = styled.div`
  font-size: 3rem;
  margin-bottom: 1rem;
  color: #9ca3af;
`;

const EmptyText = styled.p`
  color: #4b5563;
  margin-bottom: 1.5rem;
`;

const EmptyButton = styled(Link)`
  background-color: #1d3557;
  color: white;
  text-decoration: none;
  padding: 0.75rem 1.5rem;
  border-radius: 0.375rem;
  display: inline-block;
  
  &:hover {
    background-color: #14253e;
  }
`;

const ErrorMessage = styled.div`
  background-color: #fee2e2;
  color: #b91c1c;
  padding: 1rem;
  border-radius: 0.375rem;
  margin-bottom: 1.5rem;
`;

const LoadingMessage = styled.div`
  text-align: center;
  padding: 2rem;
  color: #6b7280;
`;

const CarList = () => {
  const [cars, setCars] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const navigate = useNavigate();
  
  // Fetch user cars on component mount
  useEffect(() => {
    const fetchCars = async () => {
      try {
        // Debug token information
        const token = localStorage.getItem('token');
        console.log('Token available for car list:', !!token);
        console.log('Token length:', token ? token.length : 0);
        
        // Test API connectivity first
        console.log('Testing API connectivity...');
        const testResult = await carService.testApi();
        console.log('API test result:', testResult);
        
        const response = await carService.getUserCars();
        if (response.success) {
          setCars(response.cars);
        }
      } catch (err) {
        setError('Failed to load cars. Please try again later.');
        console.error('Error fetching cars:', err);
      } finally {
        setLoading(false);
      }
    };
    
    fetchCars();
  }, []);
  
  const handleDelete = async (carId) => {
    if (window.confirm('Are you sure you want to delete this car?')) {
      try {
        const response = await carService.deleteCar(carId);
        if (response.success) {
          // Remove car from state
          setCars(cars.filter(car => car.id !== carId));
        }
      } catch (err) {
        setError('Failed to delete car. Please try again later.');
        console.error('Error deleting car:', err);
      }
    }
  };
  
  if (loading) {
    return <LoadingMessage>Loading cars...</LoadingMessage>;
  }
  
  return (
    <CarListContainer>
      <Title>My Cars</Title>
      
      {error && <ErrorMessage>{error}</ErrorMessage>}
      
      <AddCarButton to="/cars/add">
        <PlusIcon>+</PlusIcon>
        Add New Car
      </AddCarButton>
      
      {cars.length === 0 ? (
        <EmptyState>
          <EmptyIcon>🚗</EmptyIcon>
          <EmptyText>You haven't registered any cars yet.</EmptyText>
          <EmptyButton to="/cars/add">Register your first car</EmptyButton>
        </EmptyState>
      ) : (
        <CarsGrid>
          {cars.map(car => {
            // Process image URL
            let imageUrl = null;
            if (car.images && car.images.length > 0) {
              const imagePath = car.images[0];
              console.log(`Car ${car.id} image path:`, imagePath);
              
              if (!imagePath.startsWith('http') && !imagePath.startsWith('data:')) {
                const baseUrl = 'http://localhost:5000';
                imageUrl = imagePath.startsWith('/') 
                  ? `${baseUrl}${imagePath}` 
                  : `${baseUrl}/${imagePath}`;
              } else {
                imageUrl = imagePath;
              }
            }
            
            return (
              <CarCard key={car.id}>
                <CarImage>
                  <CarImagePlaceholder>
                    <CarIcon>🚗</CarIcon>
                    <CarPlaceholderText>{car.make} {car.model}</CarPlaceholderText>
                  </CarImagePlaceholder>
                </CarImage>
                
                <CarDetails>
                  <CarName>{car.make} {car.model}</CarName>
                  <CarInfo>{car.year} • {car.color}</CarInfo>
                  <CarPlate>{car.licensePlate}</CarPlate>
                  
                  {/* Show damage indicator if initial assessment exists */}
                  {car.initialDamagePercentage !== undefined && (
                    <DamageIndicator damagePercentage={car.initialDamagePercentage} />
                  )}
                  
                  <ButtonRow>
                    <ViewButton onClick={() => navigate(`/cars/${car.id}`)}>
                      View
                    </ViewButton>
                    <EditButton onClick={() => navigate(`/cars/${car.id}/edit`)}>
                      Edit
                    </EditButton>
                    <DeleteButton onClick={() => handleDelete(car.id)}>
                      Delete
                    </DeleteButton>
                  </ButtonRow>
                  
                  {/* Show damage reports button if car has an initial assessment */}
                  {car.initialAssessmentId && (
                    <ViewButton 
                      onClick={() => navigate(`/cars/${car.id}/damage-reports`)}
                      style={{ marginTop: '8px', backgroundColor: '#0ea5e9' }}
                    >
                      Damage Reports {car.hasAccidentAssessments && '🔔'}
                    </ViewButton>
                  )}
                </CarDetails>
              </CarCard>
            );
          })}
        </CarsGrid>
      )}
    </CarListContainer>
  );
};

export default CarList; 