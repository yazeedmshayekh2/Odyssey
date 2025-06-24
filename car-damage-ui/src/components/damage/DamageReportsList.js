import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import { useAuth } from '../../context/AuthContext';
import damageService from '../../services/damageService';

// Styled components
const Container = styled.div`
  max-width: 1000px;
  margin: 2rem auto;
  padding: 0 1rem;
`;

const Title = styled.h1`
  font-size: 1.875rem;
  font-weight: 600;
  color: #1d3557;
  margin-bottom: 2rem;
`;

const FilterContainer = styled.div`
  display: flex;
  gap: 1rem;
  margin-bottom: 1.5rem;
  flex-wrap: wrap;
`;

const FilterSelect = styled.select`
  padding: 0.5rem;
  border: 1px solid #d1d5db;
  border-radius: 0.375rem;
  background-color: white;
  color: #374151;
`;

const ReportsList = styled.div`
  display: grid;
  gap: 1rem;
`;

const ReportCard = styled.div`
  background-color: white;
  border-radius: 0.5rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  padding: 1.5rem;
  transition: transform 0.2s, box-shadow 0.2s;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 15px -3px rgba(0, 0, 0, 0.1);
  }
`;

const ReportHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 1rem;
`;

const ReportInfo = styled.div`
  flex: 1;
`;

const CarInfo = styled.div`
  font-size: 1.125rem;
  font-weight: 600;
  color: #1d3557;
  margin-bottom: 0.25rem;
`;

const ReportDate = styled.div`
  font-size: 0.875rem;
  color: #6b7280;
`;

const DamageIndicator = styled.div`
  display: flex;
  flex-direction: column;
  align-items: flex-end;
`;

const DamagePercentage = styled.div`
  font-size: 1.5rem;
  font-weight: 700;
  color: ${props => {
    if (props.percentage > 50) return '#ef4444';
    if (props.percentage > 20) return '#f59e0b';
    return '#10b981';
  }};
`;

const DamageLabel = styled.div`
  font-size: 0.75rem;
  color: #6b7280;
  text-transform: uppercase;
  letter-spacing: 0.05em;
`;

const ReportContent = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-bottom: 1rem;
`;

const DetailItem = styled.div`
  padding: 0.75rem;
  background-color: #f8f9fa;
  border-radius: 0.375rem;
`;

const DetailLabel = styled.div`
  font-size: 0.75rem;
  color: #6b7280;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 0.25rem;
`;

const DetailValue = styled.div`
  font-weight: 500;
  color: #1d3557;
`;

const StatusBadge = styled.span`
  display: inline-block;
  padding: 0.25rem 0.5rem;
  border-radius: 0.25rem;
  font-size: 0.75rem;
  font-weight: 500;
  text-transform: uppercase;
  background-color: ${props => {
    switch (props.status) {
      case 'completed': return '#dcfce7';
      case 'processing': return '#fef3c7';
      case 'failed': return '#fee2e2';
      default: return '#f3f4f6';
    }
  }};
  color: ${props => {
    switch (props.status) {
      case 'completed': return '#166534';
      case 'processing': return '#92400e';
      case 'failed': return '#dc2626';
      default: return '#6b7280';
    }
  }};
`;

const ActionButton = styled.button`
  background-color: #1d3557;
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 0.375rem;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  transition: background-color 0.2s;
  
  &:hover {
    background-color: #14253e;
  }
`;

const SecondaryButton = styled.button`
  background-color: #f3f4f6;
  color: #4b5563;
  border: 1px solid #d1d5db;
  padding: 0.5rem 1rem;
  border-radius: 0.375rem;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  transition: background-color 0.2s;
  
  &:hover {
    background-color: #e5e7eb;
  }
`;

const ReportActions = styled.div`
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin-top: 1rem;
`;

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 2rem;
  flex-wrap: wrap;
  gap: 1rem;
`;

const Subtitle = styled.p`
  color: #6b7280;
  margin: 0;
  font-size: 1rem;
`;

const EmptyTitle = styled.h3`
  color: #374151;
  margin: 0.5rem 0;
`;

const EmptyDescription = styled.p`
  color: #6b7280;
  margin-bottom: 1.5rem;
`;

const ErrorMessage = styled.div`
  background-color: #fee2e2;
  border: 1px solid #fecaca;
  color: #dc2626;
  padding: 1rem;
  border-radius: 0.375rem;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const EmptyState = styled.div`
  text-align: center;
  padding: 3rem 1rem;
  color: #6b7280;
`;

const EmptyIcon = styled.div`
  font-size: 3rem;
  margin-bottom: 1rem;
  color: #d1d5db;
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

const DamageReportsList = () => {
  const { user } = useAuth();
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [statusFilter, setStatusFilter] = useState('all');
  const [carFilter, setCarFilter] = useState('all');
  const navigate = useNavigate();

  useEffect(() => {
    const fetchReports = async () => {
      try {
        setLoading(true);
        console.log('DEBUG: Fetching damage reports...');
        const response = await damageService.getReports();
        console.log('DEBUG: Damage reports response:', response);
        
        if (response.success) {
          setReports(response.reports || []);
        } else {
          // If API fails, show mock data as fallback
          setReports([]);
        }
      } catch (err) {
        console.error('Error fetching damage reports:', err);
        setError('Failed to load damage reports');
        // Set empty reports instead of failing completely
        setReports([]);
      } finally {
        setLoading(false);
      }
    };

    fetchReports();
  }, []);

  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown date';
    try {
      return new Date(dateString).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch {
      return 'Invalid date';
    }
  };

  const getDamageLevel = (percentage) => {
    if (percentage > 50) return { level: 'high', text: 'High', color: '#ef4444' };
    if (percentage > 20) return { level: 'medium', text: 'Medium', color: '#f59e0b' };
    if (percentage > 0) return { level: 'low', text: 'Low', color: '#10b981' };
    return { level: 'none', text: 'None', color: '#6b7280' };
  };

  const handleViewReport = (reportId) => {
    navigate(`/damage-reports/${reportId}`);
  };

  const handleStartDetection = () => {
    navigate('/damage-detection');
  };

  const filteredReports = reports.filter(report => {
    if (statusFilter !== 'all' && report.status !== statusFilter) {
      return false;
    }
    if (carFilter !== 'all' && report.car_id !== carFilter) {
      return false;
    }
    return true;
  });

  // Get unique cars properly
  const uniqueCarsMap = new Map();
  reports.forEach(report => {
    if (report.car_id && !uniqueCarsMap.has(report.car_id)) {
      const carInfo = report.car_info || {};
      uniqueCarsMap.set(report.car_id, {
        id: report.car_id,
        info: `${carInfo.make || 'Unknown'} ${carInfo.model || 'Model'} ${carInfo.licensePlate ? '(' + carInfo.licensePlate + ')' : ''}`
      });
    }
  });
  const uniqueCars = Array.from(uniqueCarsMap.values());

  if (loading) {
    return (
      <Container>
        <LoadingState>Loading damage reports...</LoadingState>
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

  return (
    <Container>
      <Header>
        <div>
          <Title>Damage Reports</Title>
          <Subtitle>View and manage your car damage detection reports</Subtitle>
        </div>
        <ActionButton onClick={handleStartDetection}>
          New Detection
        </ActionButton>
      </Header>

      {error && (
        <ErrorMessage>
          {error}
          <button onClick={() => window.location.reload()} style={{ marginLeft: '1rem', padding: '0.25rem 0.5rem' }}>
            Retry
          </button>
        </ErrorMessage>
      )}

      {/* Filters */}
      <FilterContainer>
        <FilterSelect 
          value={statusFilter} 
          onChange={(e) => setStatusFilter(e.target.value)}
        >
          <option value="all">All Statuses</option>
          <option value="completed">Completed</option>
          <option value="processing">Processing</option>
          <option value="failed">Failed</option>
        </FilterSelect>

        <FilterSelect 
          value={carFilter} 
          onChange={(e) => setCarFilter(e.target.value)}
        >
          <option value="all">All Cars</option>
          {uniqueCars.map(car => (
            <option key={car.id} value={car.id}>{car.info}</option>
          ))}
        </FilterSelect>
      </FilterContainer>

      {/* Reports List */}
      {filteredReports.length === 0 ? (
        <EmptyState>
          <EmptyIcon>📋</EmptyIcon>
          <EmptyTitle>No damage reports found</EmptyTitle>
          <EmptyDescription>
            {statusFilter !== 'all' || carFilter !== 'all' 
              ? 'Try adjusting your filters to see more results.'
              : 'You haven\'t run any damage detection yet. Upload an image of your car to get started.'
            }
          </EmptyDescription>
          <ActionButton onClick={handleStartDetection}>
            Start Damage Detection
          </ActionButton>
        </EmptyState>
      ) : (
        <ReportsList>
          {filteredReports.map(report => {
            const damageInfo = getDamageLevel(report.damage_percentage || 0);
            const hasCarInfo = report.car_info && (report.car_info.make !== 'Unknown' || report.car_info.model !== 'Unknown');
            
            return (
              <ReportCard key={report._id || report.id}>
                <ReportHeader>
                  <ReportInfo>
                    <CarInfo>
                      {hasCarInfo ? (
                        <div>
                          <strong>{report.car_info.make} {report.car_info.model}</strong>
                          {report.car_info.year && ` (${report.car_info.year})`}
                          {report.car_info.licensePlate && (
                            <div style={{ fontSize: '0.875rem', color: '#6b7280' }}>
                              License: {report.car_info.licensePlate}
                            </div>
                          )}
                        </div>
                      ) : (
                        <div style={{ color: '#6b7280' }}>Car information not available</div>
                      )}
                    </CarInfo>
                    <ReportDate>
                      {formatDate(report.created_at)}
                    </ReportDate>
                  </ReportInfo>
                  <DamageIndicator>
                    <DamagePercentage percentage={report.damage_percentage}>
                      {(report.damage_percentage || 0).toFixed(1)}%
                    </DamagePercentage>
                    <DamageLabel>Damage</DamageLabel>
                  </DamageIndicator>
                </ReportHeader>

                <ReportContent>
                  <DetailItem>
                    <DetailLabel>Status</DetailLabel>
                    <DetailValue>
                      <StatusBadge status={report.status}>
                        {(report.status || 'completed').charAt(0).toUpperCase() + (report.status || 'completed').slice(1)}
                      </StatusBadge>
                    </DetailValue>
                  </DetailItem>
                  <DetailItem>
                    <DetailLabel>Damage Level</DetailLabel>
                    <DetailValue>
                      <DamagePercentage percentage={report.damage_percentage}>
                        {damageInfo.text}
                      </DamagePercentage>
                    </DetailValue>
                  </DetailItem>
                  {report.damage_types && report.damage_types.length > 0 && (
                    <DetailItem>
                      <DetailLabel>Detected Issues</DetailLabel>
                      <DetailValue>
                        {report.damage_types.join(', ')}
                      </DetailValue>
                    </DetailItem>
                  )}
                </ReportContent>

                <ReportActions>
                  <ActionButton onClick={() => handleViewReport(report._id || report.id)}>
                    View Details
                  </ActionButton>
                  {report.images && report.images.result_image_data && (
                    <SecondaryButton onClick={() => {
                      const link = document.createElement('a');
                      link.href = report.images.result_image_data;
                      link.download = `damage-report-${report._id || report.id}.jpg`;
                      link.click();
                    }}>
                      Download Image
                    </SecondaryButton>
                  )}
                </ReportActions>
              </ReportCard>
            );
          })}
        </ReportsList>
      )}
    </Container>
  );
};

export default DamageReportsList; 