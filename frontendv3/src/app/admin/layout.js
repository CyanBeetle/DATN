"use client";
import { useState, useEffect } from 'react';
import { 
  Layout, 
  Menu, 
  Button, 
  Avatar, 
  Dropdown, 
  Spin, 
  Alert,
  Typography,
  theme,
  message
} from 'antd';
import {
  DashboardOutlined,
  DatabaseOutlined,
  CameraOutlined,
  LineChartOutlined,
  MessageOutlined,
  UserOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  LogoutOutlined,
  SettingOutlined
} from '@ant-design/icons';
import { useRouter, usePathname } from 'next/navigation';
import { useAuth } from '@/context/authContext';
import Link from 'next/link';
import axios from 'axios';

const { Header, Sider, Content } = Layout;
const { Title, Text } = Typography;

const AdminLayout = ({ children }) => {
  const { user, isAuthenticated, logout } = useAuth();
  const router = useRouter();
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const { token } = theme.useToken();
  
  // Check authentication and admin role on mount
  useEffect(() => {
    // If authentication state is still being determined by AuthProvider, wait.
    if (isAuthenticated === undefined) {
      setLoading(true); // Show loading while auth state is resolving
      return;
    }
    
    // If not authenticated, redirect to login
    if (isAuthenticated === false) {
      router.push('/auth/login');
      return; // Stop further execution
    }
    
    // If authenticated, but user data is not yet loaded (e.g. initial load), wait.
    // Or if user role is not admin, redirect.
    if (!user || user.role?.toLowerCase() !== 'admin') {
      // If user data is loaded and role is not admin, redirect
      if (user && user.role?.toLowerCase() !== 'admin') {
        message.error('You do not have permission to access the admin panel');
        router.push('/user/map'); // Redirect to default user page
        return; // Stop further execution
      }
      // If user data is not yet loaded, keep loading
      // This scenario can happen if isAuthenticated becomes true before user object is fully populated.
      setLoading(true);
      return;
    }
    
    // Successful authentication and admin authorization
    setLoading(false);
    setError(null); // Clear any previous errors

    // The periodic token refresh using setInterval and refreshToken() has been removed.
    // Token refresh is now handled by the axiosInstance interceptor when a 401 is encountered.

  }, [isAuthenticated, user, router]); // Dependency array updated
  
  // Handle logout
  const handleLogout = async () => {
    try {
      setLoading(true);
      await logout();
      message.success('Successfully logged out');
      router.push('/auth/login');
    } catch (err) {
      console.error('Logout error:', err);
      message.error('Failed to logout properly');
    } finally {
      setLoading(false);
    }
  };
  
  // User dropdown menu
  const userMenu = {
    items: [
      {
        key: 'profile',
        label: 'Settings',
        icon: <SettingOutlined />,
        onClick: () => router.push('/user/settings')
      },
      {
        key: 'logout',
        label: 'Logout',
        icon: <LogoutOutlined />,
        onClick: handleLogout
      }
    ]
  };
  
  // Get current active menu item
  const getActiveMenuItem = () => {
    if (pathname.includes('/admin/training-data')) return 'training-data';
    if (pathname.includes('/admin/prediction-models')) return 'prediction-models';
    if (pathname.includes('/admin/cameras')) return 'cameras';
    if (pathname.includes('/admin/reports')) return 'reports';
    return 'dashboard';
  };
  
  // Not authenticated or not admin view
  if (!isAuthenticated && isAuthenticated !== undefined) {
    return (
      <div style={{ padding: '24px' }}>
        <Alert
          message="Authentication Required"
          description="Please log in with an admin account to access this page."
          type="error"
          showIcon
          action={
            <Button type="primary" onClick={() => router.push('/auth/login')}>
              Login
            </Button>
          }
        />
      </div>
    );
  }
  
  if (isAuthenticated && user && user.role && user.role.toLowerCase() !== 'admin') {
    return (
      <div style={{ padding: '24px' }}>
        <Alert
          message="Access Denied"
          description="You do not have permission to access this page. This page is restricted to admin users only."
          type="error"
          showIcon
          action={
            <Button type="primary" onClick={() => router.push('/user/map')}>
              Go to User Dashboard
            </Button>
          }
        />
      </div>
    );
  }
  
  // Display error if authentication check failed
  if (error) {
    return (
      <div style={{ padding: '24px' }}>
        <Alert
          message="Authentication Error"
          description={error}
          type="error"
          showIcon
          action={
            <Button type="primary" onClick={() => router.push('/auth/login')}>
              Try Login Again
            </Button>
          }
        />
      </div>
    );
  }
  
  // Still loading
  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <Spin size="large" tip="Loading admin dashboard..." />
      </div>
    );
  }
  
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider 
        trigger={null} 
        collapsible 
        collapsed={collapsed}
        width={250}
        style={{
          overflow: 'auto',
          height: '100vh',
          position: 'fixed',
          left: 0,
          top: 0,
          bottom: 0,
          backgroundColor: token.colorBgContainer
        }}
      >
        <div 
          style={{ 
            display: 'flex',
            alignItems: 'center',
            justifyContent: collapsed ? 'center' : 'flex-start',
            padding: collapsed ? '16px 0' : '16px',
            borderBottom: `1px solid ${token.colorBorderSecondary}`
          }}
        >
          {collapsed ? (
            <Link href="/admin/dashboard">
              <div style={{ padding: '8px', cursor: 'pointer' }}>
                <DashboardOutlined style={{ fontSize: '24px', color: token.colorPrimary }} />
              </div>
            </Link>
          ) : (
            <Link href="/admin/dashboard">
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', cursor: 'pointer' }}>
                <DashboardOutlined style={{ fontSize: '24px', color: token.colorPrimary }} />
                <Title level={4} style={{ margin: 0 }}>Admin Panel</Title>
              </div>
            </Link>
          )}
        </div>
        
        <Menu
          mode="inline"
          selectedKeys={[getActiveMenuItem()]}
          style={{ borderRight: 0 }}
          items={[
            {
              key: 'dashboard',
              icon: <DashboardOutlined />,
              label: 'Admin Dashboard',
              onClick: () => router.push('/admin/dashboard')
            },
            {
              key: 'prediction-models',
              icon: <LineChartOutlined />,
              label: 'Prediction Models',
              onClick: () => router.push('/admin/prediction-models')
            },
            {
              key: 'cameras',
              icon: <CameraOutlined />,
              label: 'Camera Management',
              onClick: () => router.push('/admin/managecamera')
            },
            {
              key: 'reports',
              icon: <MessageOutlined />,
              label: 'Process Reports',
              onClick: () => router.push('/admin/reports')
            },
            {
              type: 'divider',
            },
            {
              key: 'user-view',
              icon: <UserOutlined />,
              label: 'Switch to User View',
              onClick: () => router.push('/user/map')
            }
          ]}
        />
      </Sider>
      
      <Layout style={{ marginLeft: collapsed ? 80 : 250, transition: 'all 0.2s' }}>
        <Header 
          style={{ 
            padding: '0 16px', 
            display: 'flex', 
            alignItems: 'center', 
            backgroundColor: token.colorBgContainer,
            boxShadow: '0 1px 4px rgba(0, 0, 0, 0.05)',
            position: 'sticky',
            top: 0,
            zIndex: 1,
            width: '100%'
          }}
        >
          <Button
            type="text"
            icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={() => setCollapsed(!collapsed)}
            style={{ fontSize: '16px', width: 64, height: 64 }}
          />
          
          <div style={{ flex: '1 1 0%' }} />
          
          <Dropdown menu={userMenu} placement="bottomRight">
            <div style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
              <Avatar icon={<UserOutlined />} style={{ marginRight: '8px' }} />
              <div>
                <Text strong>{user?.username || 'Admin'}</Text>
              </div>
            </div>
          </Dropdown>
        </Header>
        
        <Content>
          <div 
            style={{ 
              minHeight: '100%',
              backgroundColor: token.colorBgLayout
            }}
          >
            {children}
          </div>
        </Content>
      </Layout>
    </Layout>
  );
};

export default AdminLayout;
