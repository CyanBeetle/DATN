"use client";
import { useAuth } from '@/context/authContext';
import {
  AimOutlined,
  BellOutlined,
  CameraOutlined,
  DashboardOutlined,
  DatabaseOutlined,
  FileTextOutlined,
  HomeOutlined,
  LineChartOutlined,
  LogoutOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  MessageOutlined,
  RobotOutlined,
  SettingOutlined,
  UserOutlined
} from '@ant-design/icons';
import {
  Alert,
  Avatar,
  Badge,
  Button,
  Dropdown,
  Layout,
  Menu,
  Spin,
  Typography,
  message,
  theme
} from 'antd';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';

// Use AimOutlined as a replacement for MapOutlined
const MapOutlined = AimOutlined;

const { Header, Sider, Content } = Layout;
const { Title, Text } = Typography;

const UserLayout = ({ children }) => {
  const { user, isAuthenticated, logout } = useAuth();
  const router = useRouter();
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const { token } = theme.useToken();
  const isAdmin = user?.role?.toLowerCase() === 'admin';

  // Check authentication status on mount
  useEffect(() => {
    // If authentication state is still being determined by AuthProvider, wait.
    if (isAuthenticated === undefined) {
      setLoading(true); // Show loading while auth state is resolving
      return;
    }

    // If not authenticated, redirect to login.
    if (isAuthenticated === false) {
      router.push('/auth/login');
      return; // Stop further execution in this effect
    }

    // If authenticated, stop loading.
    setLoading(false);
    setError(null); // Clear any previous errors

    // The periodic token refresh using setInterval and refreshToken() has been removed.
    // Token refresh is now handled by the axiosInstance interceptor when a 401 is encountered.

  }, [isAuthenticated, router]); // Dependency array updated

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

  // Get current active menu item based on pathname
  const getActiveMenuItem = () => {
    // Admin routes
    if (pathname.includes('/admin/training-data')) return 'admin-training-data';
    if (pathname.includes('/admin/prediction-models')) return 'admin-prediction-models';
    if (pathname.includes('/admin/cameras')) return 'admin-cameras';
    if (pathname.includes('/admin/reports')) return 'admin-reports';
    if (pathname.includes('/admin/dashboard')) return 'admin-dashboard';

    // User routes
    if (pathname.includes('/user/map')) return 'map';
    if (pathname.includes('/user/camera')) return 'camera';
    if (pathname.includes('/user/reports')) return 'reports';
    if (pathname.includes('/user/news')) return 'news';
    if (pathname.includes('/user/settings')) return 'settings';
    if (pathname.includes('/user/traffic-analysis')) return 'traffic-analysis';
    if (pathname.includes('/user/support')) return 'support';


    return 'map'; // Default to map
  };

  // Define menu items for regular users
  const userMenuItems = [
    {
      key: 'map',
      icon: <MapOutlined />,
      label: 'Traffic Map',
      onClick: () => router.push('/user/map')
    },
    {
      key: 'camera',
      icon: <CameraOutlined />,
      label: 'Camera View',
      onClick: () => router.push('/user/camera')
    },
    {
      key: 'traffic-analysis',
      icon: <LineChartOutlined />,
      label: 'Traffic Forecast',
      onClick: () => router.push('/user/traffic-analysis')
    },
    {
      key: 'news',
      icon: <FileTextOutlined />,
      label: 'News & Weather',
      onClick: () => router.push('/user/news')
    },
    {
      key: 'reports',
      icon: <MessageOutlined />,
      label: 'Reports & Feedback',
      onClick: () => router.push('/user/reports')
    },
    {
      key: 'support',
      icon: <RobotOutlined />,
      label: 'Support & Help',
      onClick: () => router.push('/user/support/chatbot')
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: 'Account Settings',
      onClick: () => router.push('/user/settings')
    }
  ];

  // Define admin menu items (only shown to admin users)
  const adminMenuItems = [
    {
      key: 'admin-divider',
      type: 'divider'
    },
    {
      key: 'admin-section',
      type: 'group',
      label: 'Admin Panel'
    },
    {
      key: 'admin-dashboard',
      icon: <DashboardOutlined />,
      label: 'Admin Dashboard',
      onClick: () => router.push('/admin/dashboard')
    },
    {
      key: 'admin-prediction-models',
      icon: <LineChartOutlined />,
      label: 'Prediction Models',
      onClick: () => router.push('/admin/prediction-models')
    },
    {
      key: 'admin-cameras',
      icon: <CameraOutlined />,
      label: 'Camera Management',
      onClick: () => router.push('/admin/managecamera')
    },
    {
      key: 'admin-reports',
      icon: <MessageOutlined />,
      label: 'Process Reports',
      onClick: () => router.push('/admin/reports')
    }
  ];

  // Combine menu items based on user role
  const allMenuItems = isAdmin
    ? [...userMenuItems, ...adminMenuItems]
    : userMenuItems;

  // Not authenticated
  if (!isAuthenticated && isAuthenticated !== undefined) {
    return (
      <div style={{ padding: '24px' }}>
        <Alert
          message="Authentication Required"
          description="Please log in to access this page."
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
        <Spin size="large" tip="Loading..." />
      </div>
    );
  }

  return (
    <Layout style={{ minHeight: '100vh', padding: 0 }}>
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
            <Link href="/user/map">
              <div style={{ padding: '8px', cursor: 'pointer' }}>
                <HomeOutlined style={{ fontSize: '24px', color: token.colorPrimary }} />
              </div>
            </Link>
          ) : (
            <Link href="/user/map">
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', cursor: 'pointer' }}>
                <HomeOutlined style={{ fontSize: '24px', color: token.colorPrimary }} />
                <Title level={4} style={{ margin: 0 }}>
                  Traffic Monitor
                  {isAdmin && <Badge count="Admin" style={{ marginLeft: 8 }} />}
                </Title>
              </div>
            </Link>
          )}
        </div>

        <Menu
          mode="inline"
          selectedKeys={[getActiveMenuItem()]}
          style={{ borderRight: 0 }}
          items={allMenuItems}
        />
      </Sider>

      <Layout style={{ marginLeft: collapsed ? 80 : 250, transition: 'all 0.2s', height: '100%' }}>
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

          {/* Notifications bell - placeholder for future functionality */}
          <Badge count={0} style={{ marginRight: 16 }}>
            <Button type="text" icon={<BellOutlined />} size="large" />
          </Badge>

          <Dropdown menu={userMenu} placement="bottomRight">
            <div style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
              <Avatar
                icon={<UserOutlined />}
                style={{
                  marginRight: '8px',
                  backgroundColor: isAdmin ? '#ff4d4f' : '#1890ff'
                }}
              />
              <div>
                <Text strong>{user?.username || 'User'}</Text>
                {isAdmin && <Text type="secondary" style={{ fontSize: '12px', display: 'block' }}>Admin</Text>}
              </div>
            </div>
          </Dropdown>
        </Header>

        <Content>
          <div
            style={{
              height: '100%',
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

export default UserLayout;