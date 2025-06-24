import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import styled from 'styled-components';

// Styled components
const Container = styled.div`
  max-width: 1000px;
  margin: 2rem auto;
  padding: 0 1rem;
`;

const Header = styled.div`
  display: flex;
  justify-content: between;
  align-items: center;
  margin-bottom: 2rem;
`;

const BackButton = styled.button`
  background-color: #6b7280;
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 0.375rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  
  &:hover {
    background-color: #4b5563;
  }
`;

const Title = styled.h1`
  font-size: 1.875rem;
  font-weight: 600;
  color: #1d3557;
  margin: 0 auto;
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

const ImageGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1rem;
  margin-bottom: 1.5rem;
`;

const ImageContainer = styled.div`
  border-radius: 0.5rem;
  overflow: hidden;
  border: 1px solid #e5e7eb;
`;

const Image = styled.img`
  width: 100%;
  height: 200px;
  object-fit: cover;
`;

const ImageLabel = styled.div`
  padding: 0.5rem;
  background-color: #f9fafb;
  font-size: 0.875rem;
  color: #6b7280;
  text-align: center;
`;

const InfoGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
`;

const InfoItem = styled.div`
  padding: 1rem;
  background-color: #f8f9fa;
  border-radius: 0.375rem;
`;

const InfoLabel = styled.div`
  font-size: 0.875rem;
  color: #6b7280;
  margin-bottom: 0.25rem;
`;

const InfoValue = styled.div`
  font-size: 1rem;
  font-weight: 600;
  color: #1d3557;
`;

const DamagePercentage = styled.div`
  font-size: 2rem;
  font-weight: 700;
  color: ${props => {
    if (props.percentage > 50) return '#ef4444';
    if (props.percentage > 20) return '#f59e0b';
    return '#10b981';
  }};
`;

const DamageList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;
`;

const DamageItem = styled.li`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem;
  border-bottom: 1px solid #e5e7eb;
  
  &:last-child {
    border-bottom: none;
  }
`;

const DamageName = styled.span`
  font-weight: 500;
  color: #1d3557;
`;

const DamageConfidence = styled.span`
  font-size: 0.875rem;
  color: #6b7280;
`;

const LoadingState = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  height: 200px;
  color: #6b7280;
`;

const ErrorState = styled.div`
  background-color: #fee2e2;
  border: 1px solid #fecaca;
  color: #dc2626;
  padding: 1rem;
  border-radius: 0.375rem;
  text-align: center;
`;

const DamageReport = () => {
  const { reportId } = useParams();
  const navigate = useNavigate();
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchReport = async () => {
      try {
        setLoading(true);
        // TODO: Replace with actual API call
        // const response = await damageService.getReport(reportId);
        // setReport(response.data.report);
        
        // Mock data for now
        setTimeout(() => {
          setReport({
            id: reportId,
            carId: 'mock-car-id',
            carInfo: {
              make: 'Toyota',
              model: 'Camry',
              year: 2020,
              licensePlate: 'ABC-123'
            },
            damagePercentage: 25.5,
            reportDate: new Date().toISOString(),
            images: {
              original: '/path/to/original.jpg',
              processed: '/path/to/processed.jpg'
            },
            detectedDamages: [
              { type: 'scratch', confidence: 0.89, area: 15.2 },
              { type: 'dent', confidence: 0.75, area: 8.7 }
            ],
            status: 'completed'
          });
          setLoading(false);
        }, 1000);
      } catch (err) {
        setError('Failed to load damage report');
        setLoading(false);
      }
    };

    if (reportId) {
      fetchReport();
    }
  }, [reportId]);

  const handleBack = () => {
    navigate(-1);
  };

  if (loading) {
    return (
      <Container>
        <LoadingState>Loading damage report...</LoadingState>
      </Container>
    );
  }

  if (error) {
    return (
      <Container>
        <ErrorState>{error}</ErrorState>
      </Container>
    );
  }

  if (!report) {
    return (
      <Container>
        <ErrorState>Damage report not found</ErrorState>
      </Container>
    );
  }

  return (
    <Container>
      <Header>
        <BackButton onClick={handleBack}>
          ← Back
        </BackButton>
        <Title>Damage Report</Title>
        <div style={{ width: '80px' }}></div> {/* Spacer for center alignment */}
      </Header>

      {/* Car Information */}
      <Card>
        <SectionTitle>Vehicle Information</SectionTitle>
        <InfoGrid>
          <InfoItem>
            <InfoLabel>Make & Model</InfoLabel>
            <InfoValue>{report.carInfo.make} {report.carInfo.model}</InfoValue>
          </InfoItem>
          <InfoItem>
            <InfoLabel>Year</InfoLabel>
            <InfoValue>{report.carInfo.year}</InfoValue>
          </InfoItem>
          <InfoItem>
            <InfoLabel>License Plate</InfoLabel>
            <InfoValue>{report.carInfo.licensePlate}</InfoValue>
          </InfoItem>
          <InfoItem>
            <InfoLabel>Report Date</InfoLabel>
            <InfoValue>{new Date(report.reportDate).toLocaleDateString()}</InfoValue>
          </InfoItem>
        </InfoGrid>
      </Card>

      {/* Damage Summary */}
      <Card>
        <SectionTitle>Damage Assessment</SectionTitle>
        <InfoGrid>
          <InfoItem>
            <InfoLabel>Overall Damage</InfoLabel>
            <DamagePercentage percentage={report.damagePercentage}>
              {report.damagePercentage}%
            </DamagePercentage>
          </InfoItem>
          <InfoItem>
            <InfoLabel>Status</InfoLabel>
            <InfoValue style={{ 
              color: report.status === 'completed' ? '#10b981' : '#f59e0b' 
            }}>
              {report.status.charAt(0).toUpperCase() + report.status.slice(1)}
            </InfoValue>
          </InfoItem>
        </InfoGrid>
      </Card>

      {/* Images */}
      <Card>
        <SectionTitle>Images</SectionTitle>
        <ImageGrid>
          <ImageContainer>
            <Image src={report.images.original} alt="Original" />
            <ImageLabel>Original Image</ImageLabel>
          </ImageContainer>
          <ImageContainer>
            <Image src={report.images.processed} alt="Processed" />
            <ImageLabel>Damage Analysis</ImageLabel>
          </ImageContainer>
        </ImageGrid>
      </Card>

      {/* Detected Damages */}
      <Card>
        <SectionTitle>Detected Damages</SectionTitle>
        <DamageList>
          {report.detectedDamages.map((damage, index) => (
            <DamageItem key={index}>
              <div>
                <DamageName>{damage.type.charAt(0).toUpperCase() + damage.type.slice(1)}</DamageName>
                <div style={{ fontSize: '0.875rem', color: '#6b7280' }}>
                  Area: {damage.area}%
                </div>
              </div>
              <DamageConfidence>
                {Math.round(damage.confidence * 100)}% confidence
              </DamageConfidence>
            </DamageItem>
          ))}
        </DamageList>
      </Card>
    </Container>
  );
};

export default DamageReport; 