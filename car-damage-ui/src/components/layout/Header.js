import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import { useAuth } from '../../context/AuthContext';

// Styled components
const HeaderContainer = styled.header`
  background-color: #1d3557;
  color: white;
  padding: 1rem 0;
`;

const HeaderContent = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1rem;
`;

const Logo = styled(Link)`
  font-size: 1.5rem;
  font-weight: 700;
  color: white;
  text-decoration: none;
  display: flex;
  align-items: center;

  &:hover {
    color: #f1faee;
  }
`;

const LogoIcon = styled.span`
  margin-right: 0.5rem;
`;

const NavMenu = styled.nav`
  display: flex;
  align-items: center;
`;

const NavList = styled.ul`
  display: flex;
  list-style: none;
  margin: 0;
  padding: 0;
`;

const NavItem = styled.li`
  margin-left: 1.5rem;
`;

const NavLink = styled(Link)`
  color: white;
  text-decoration: none;
  position: relative;

  &:hover {
    color: #f1faee;
  }

  &::after {
    content: '';
    position: absolute;
    width: 0;
    height: 2px;
    bottom: -4px;
    left: 0;
    right: 2px;
    background-color: #f1faee;
    transition: width 0.2s ease;
  }

  &:hover::after {
    width: 100%;
  }
`;

const AuthButton = styled.button`
  background-color: ${props => props.$primary ? '#e63946' : 'transparent'};
  color: white;
  padding: 0.5rem ${props => props.$primary ? '1rem' : '0'};
  border: ${props => props.$primary ? 'none' : '1px solid #23355600'};
  border-radius: 0.25rem;
  font-size: 0.875rem;
  cursor: pointer;
  margin-left: 1rem;

  &:hover {
    background-color: ${props => props.$primary ? '#c1121f' : 'rgba(255, 255, 255, 0.1)'};
  }
`;

const UserInfo = styled.div`
  display: flex;
  align-items: center;
`;

const UserName = styled.span`
  margin-right: 1rem;
  font-weight: 500;
  margin-left: 15px;

`;

const Header = () => {
  const { user, isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <HeaderContainer>
      <HeaderContent>
        <Logo to="/">
          <LogoIcon>🚗</LogoIcon>
          <span>Odyssey</span>
        </Logo>

        <NavMenu>
          <NavList>
            <NavItem>
              <NavLink to="/">Home</NavLink>
            </NavItem>
            
            {isAuthenticated && (
              <>
                <NavItem>
                  <NavLink to="/cars">My Cars</NavLink>
                </NavItem>
                <NavItem>
                  <NavLink to="/damage-detection">Damage Detection</NavLink>
                </NavItem>
                <NavItem>
                  <NavLink to="/damage-reports">Damage Reports</NavLink>
                </NavItem>
                <NavItem>
                  <NavLink to="/accident-report">Report Accident</NavLink>
                </NavItem>
              </>
            )}
          </NavList>

          {isAuthenticated ? (
            <UserInfo>
              <UserName>Hello, {user?.firstName || 'User'}</UserName>
              <AuthButton onClick={handleLogout}>Logout</AuthButton>
            </UserInfo>
          ) : (
            <>
              <AuthButton onClick={() => navigate('/login')}>Login</AuthButton>
              <AuthButton $primary onClick={() => navigate('/register')}>Register</AuthButton>
            </>
          )}
        </NavMenu>
      </HeaderContent>
    </HeaderContainer>
  );
};

export default Header; 