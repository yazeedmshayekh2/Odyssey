import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import { useAuth } from '../../context/AuthContext';
import carService from '../../services/carService';
import damageService from '../../services/damageService';
import DamageIndicator from '../damage/DamageIndicator';

const DashboardContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem 1rem;
`;

const PageHeader = styled.div`
  margin-bottom: 2rem;
`;

const PageTitle = styled.h1`
  font-size: 2.5rem;
  font-weight: bold;
  color: #1d3557;
  margin-bottom: 0.5rem;
`;

const PageSubtitle = styled.p`
  color: #6b7280;
  font-size: 1.125rem;
`;

const DashboardGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr;
  gap: 2rem;
  
  @media (min-width: 768px) {
    grid-template-columns: 2fr 1fr;
  }
`;

const MainColumn = styled.div`
  display: flex;
  flex-direction: column;
  gap: 2rem;
`;

const SideColumn = styled.div`
  display: flex;
  flex-direction: column;
  gap: 2rem;
`;

const Card = styled.div`
  background-color: white;
  border-radius: 0.75rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  padding: 1.5rem;
`;

const CardHeader = styled.div`
  display: flex;
  justify-content: between;
  align-items: center;
  margin-bottom: 1rem;
`;

const CardTitle = styled.h2`
  font-size: 1.5rem;
  font-weight: 600;
  color: #1d3557;
`;

const ViewAllLink = styled(Link)`
  color: #e63946;
  margin-left: auto;
  text-decoration: none;
  font-weight: 500;
  &:hover {
    text-decoration: underline;
  }
`;

const QuickActionCard = styled(Card)`
  background: linear-gradient(135deg, #1d3557 0%, #457b9d 100%);
  color: white;
`;

const QuickActionTitle = styled.h3`
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: 1rem;
`;

const QuickActionGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
`;

const QuickActionButton = styled(Link)`
  background-color: rgba(255, 255, 255, 0.15);
  color: white;
  text-decoration: none;
  padding: 1rem;
  border-radius: 0.5rem;
  text-align: center;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background-color 0.2s ease;
  
  &:hover {
    background-color: rgba(255, 255, 255, 0.25);
  }
`;

const CarCard = styled.div`
  border: 1px solid #e5e7eb;
  border-radius: 0.5rem;
  padding: 1rem;
  margin-bottom: 1rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const CarInfo = styled.div`
  flex: 1;
`;

const CarName = styled.h4`
  font-weight: 600;
  margin-bottom: 0.25rem;
  color: #1d3557;
`;

const CarDetails = styled.p`
  color: #6b7280;
  font-size: 0.875rem;
`;

const CarActions = styled.div`
  display: flex;
  gap: 0.5rem;
`;

const ActionButton = styled(Link)`
  background-color: #e63946;
  color: white;
  padding: 0.375rem 0.75rem;
  border-radius: 0.25rem;
  text-decoration: none;
  font-size: 0.875rem;
  
  &:hover {
    background-color: #c1121f;
  }
`;

const ReportCard = styled.div`
  border-left: 4px solid #e63946;
  background-color: #f9fafb;
  padding: 1rem;
  margin-bottom: 1rem;
  border-radius: 0 0.5rem 0.5rem 0;
`;

const ReportDate = styled.div`
  color: #6b7280;
  font-size: 0.875rem;
  margin-bottom: 0.5rem;
`;

const ReportTitle = styled.h4`
  font-weight: 600;
  color: #1d3557;
  margin-bottom: 0.25rem;
`;

const ReportSummary = styled.p`
  color: #4b5563;
  font-size: 0.875rem;
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  margin-bottom: 1.5rem;
`;

const StatCard = styled.div`
  background-color: #f8f9fa;
  padding: 1rem;
  border-radius: 0.5rem;
  text-align: center;
`;

const StatNumber = styled.div`
  font-size: 2rem;
  font-weight: bold;
  color: #e63946;
`;

const StatLabel = styled.div`
  color: #6b7280;
  font-size: 0.875rem;
`;

const EmptyState = styled.div`
  text-align: center;
  color: #6b7280;
  padding: 2rem;
`;

const LoadingState = styled.div`
  text-align: center;
  color: #6b7280;
  padding: 1rem;
`;

const Dashboard = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [cars, setCars] = useState([]);
  const [recentReports, setRecentReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    totalCars: 0,
    totalReports: 0,
    pendingReports: 0,
    activeIssues: 0
  });

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        
        // Fetch user's cars
        const carsResponse = await carService.getUserCars();
        let userCars = [];
        if (carsResponse.success && carsResponse.cars) {
          userCars = carsResponse.cars;
        } else if (carsResponse.data && carsResponse.data.cars) {
          userCars = carsResponse.data.cars;
        } else if (Array.isArray(carsResponse)) {
          userCars = carsResponse;
        }
        setCars(userCars.slice(0, 3)); // Show only first 3 cars
        
        // Fetch recent damage reports
        const reportsResponse = await damageService.getUserReports();
        let reports = [];
        if (reportsResponse.success && reportsResponse.reports) {
          reports = reportsResponse.reports;
        } else if (reportsResponse.data && reportsResponse.data.reports) {
          reports = reportsResponse.data.reports;
        } else if (Array.isArray(reportsResponse)) {
          reports = reportsResponse;
        }
        setRecentReports(reports.slice(0, 3)); // Show only last 3 reports
        
        // Calculate stats
        setStats({
          totalCars: userCars.length,
          totalReports: reports.length,
          pendingReports: reports.filter(r => r.status === 'pending').length,
          activeIssues: reports.filter(r => r.damage_detected && r.damage_percentage > 10).length
        });
        
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  if (loading) {
    return (
      <DashboardContainer>
        <LoadingState>Loading dashboard...</LoadingState>
      </DashboardContainer>
    );
  }

  return (
    <DashboardContainer>
      <PageHeader>
        <PageTitle>Welcome back, {user?.name || 'User'}!</PageTitle>
        <PageSubtitle>Here's an overview of your vehicles and recent activity</PageSubtitle>
      </PageHeader>

      <DashboardGrid>
        <MainColumn>
          {/* My Cars Section */}
          <Card>
            <CardHeader>
              <CardTitle>My Vehicles</CardTitle>
              <ViewAllLink to="/cars">View All</ViewAllLink>
            </CardHeader>
            
            {cars.length === 0 ? (
              <EmptyState>
                <p>No vehicles registered yet.</p>
                <ActionButton to="/cars/add" style={{ marginTop: '1rem' }}>
                  Add Your First Vehicle
                </ActionButton>
              </EmptyState>
            ) : (
              <>
                {cars.map((car, index) => (
                  <CarCard key={car._id || car.id || `car-${index}`}>
                    <CarInfo>
                      <CarName>{car.year} {car.make} {car.model}</CarName>
                      <CarDetails>
                        VIN: {car.vin} • License: {car.license_plate || car.licensePlate}
                      </CarDetails>
                    </CarInfo>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                      <DamageIndicator carId={car._id || car.id} />
                      <CarActions>
                        <ActionButton to={`/cars/${car._id || car.id}`}>View</ActionButton>
                      </CarActions>
                    </div>
                  </CarCard>
                ))}
              </>
            )}
          </Card>

          {/* Recent Reports Section */}
          <Card>
            <CardHeader>
              <CardTitle>Recent Damage Reports</CardTitle>
              <ViewAllLink to="/damage-reports">View All</ViewAllLink>
            </CardHeader>
            
            {recentReports.length === 0 ? (
              <EmptyState>
                <p>No damage reports yet.</p>
                <ActionButton to="/damage-detection" style={{ marginTop: '1rem' }}>
                  Start Damage Detection
                </ActionButton>
              </EmptyState>
            ) : (
              <>
                {recentReports.map((report, index) => (
                  <ReportCard key={report._id || report.id || `report-${index}`}>
                    <ReportDate>
                      {new Date(report.created_at || report.createdAt || Date.now()).toLocaleDateString()}
                    </ReportDate>
                    <ReportTitle>
                      {report.car_info?.year || report.carInfo?.year} {report.car_info?.make || report.carInfo?.make} {report.car_info?.model || report.carInfo?.model}
                    </ReportTitle>
                    <ReportSummary>
                      {report.damage_detected || report.damageDetected ? 
                        `${report.damage_percentage || report.damagePercentage || 0}% damage detected` : 
                        'No damage detected'
                      }
                    </ReportSummary>
                  </ReportCard>
                ))}
              </>
            )}
          </Card>
        </MainColumn>

        <SideColumn>
          {/* Quick Actions */}
          <QuickActionCard>
            <QuickActionTitle>Quick Actions</QuickActionTitle>
            <QuickActionGrid>
              <QuickActionButton to="/damage-detection">
                Detect Damage
              </QuickActionButton>
              <QuickActionButton to="/cars/add">
                Add Vehicle
              </QuickActionButton>
              <QuickActionButton to="/accident-report">
                Report Accident
              </QuickActionButton>
              <QuickActionButton to="/documents">
                Upload Documents
              </QuickActionButton>
            </QuickActionGrid>
          </QuickActionCard>

          {/* Statistics */}
          <Card>
            <CardTitle>Statistics</CardTitle>
            <StatsGrid>
              <StatCard>
                <StatNumber>{stats.totalCars}</StatNumber>
                <StatLabel>Total Vehicles</StatLabel>
              </StatCard>
              <StatCard>
                <StatNumber>{stats.totalReports}</StatNumber>
                <StatLabel>Total Reports</StatLabel>
              </StatCard>
              <StatCard>
                <StatNumber>{stats.pendingReports}</StatNumber>
                <StatLabel>Pending Reports</StatLabel>
              </StatCard>
              <StatCard>
                <StatNumber>{stats.activeIssues}</StatNumber>
                <StatLabel>Active Issues</StatLabel>
              </StatCard>
            </StatsGrid>
          </Card>
        </SideColumn>
      </DashboardGrid>
    </DashboardContainer>
  );
};

export default Dashboard; 