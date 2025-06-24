import React from 'react';
import styled from 'styled-components';

// Styled components
const IndicatorContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const StatusDot = styled.div`
  width: 0.75rem;
  height: 0.75rem;
  border-radius: 50%;
  background-color: ${props => {
    switch (props.status) {
      case 'high': return '#ef4444';
      case 'medium': return '#f59e0b';
      case 'low': return '#10b981';
      case 'none':
      default: return '#6b7280';
    }
  }};
`;

const StatusText = styled.span`
  font-size: 0.75rem;
  color: ${props => {
    switch (props.status) {
      case 'high': return '#dc2626';
      case 'medium': return '#d97706';
      case 'low': return '#059669';
      case 'none':
      default: return '#6b7280';
    }
  }};
  font-weight: 500;
`;

const DamageIndicator = ({ damageReports = [], className }) => {
  // Calculate damage status based on reports
  const getDamageStatus = () => {
    if (!damageReports || damageReports.length === 0) {
      return { status: 'none', text: 'No damage reports' };
    }

    // Find the highest damage level from all reports
    const highDamageCount = damageReports.filter(report => 
      report.damagePercentage > 50 || report.severity === 'high'
    ).length;
    
    const mediumDamageCount = damageReports.filter(report => 
      report.damagePercentage > 20 && report.damagePercentage <= 50 || report.severity === 'medium'
    ).length;
    
    const lowDamageCount = damageReports.filter(report => 
      report.damagePercentage <= 20 || report.severity === 'low'
    ).length;

    if (highDamageCount > 0) {
      return { 
        status: 'high', 
        text: `${damageReports.length} report${damageReports.length > 1 ? 's' : ''} (High damage)` 
      };
    } else if (mediumDamageCount > 0) {
      return { 
        status: 'medium', 
        text: `${damageReports.length} report${damageReports.length > 1 ? 's' : ''} (Medium damage)` 
      };
    } else if (lowDamageCount > 0) {
      return { 
        status: 'low', 
        text: `${damageReports.length} report${damageReports.length > 1 ? 's' : ''} (Minor damage)` 
      };
    }

    return { status: 'none', text: 'No damage detected' };
  };

  const { status, text } = getDamageStatus();

  return (
    <IndicatorContainer className={className}>
      <StatusDot status={status} />
      <StatusText status={status}>{text}</StatusText>
    </IndicatorContainer>
  );
};

export default DamageIndicator; 