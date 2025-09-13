"use client";

import { useState, useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { useAuth } from '@/context/authContext';
import { 
  Form, 
  Input, 
  Button, 
  Card, 
  Typography, 
  Alert, 
  Space, 
  Spin, 
  Divider,
  message 
} from 'antd';
import { 
  UserOutlined, 
  LockOutlined, 
  EyeInvisibleOutlined, 
  EyeOutlined, 
  SafetyOutlined 
} from '@ant-design/icons';

const { Title, Text } = Typography;

export default function LoginPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { login, loading, isAuthenticated } = useAuth();
  const [form] = Form.useForm();
  
  const [messageApi, contextHolder] = message.useMessage();
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  // Get callback URL from query params or use default route
  const callbackUrl = searchParams.get('callbackUrl') || '/';

  // Redirect if already authenticated (and not in the middle of this form submitting)
  useEffect(() => {
    if (isAuthenticated && !submitting) { 
      messageApi.success('You are already logged in. Redirecting...');
      const redirectTimer = setTimeout(() => {
        router.replace(decodeURIComponent(callbackUrl));
      }, 500);
      return () => clearTimeout(redirectTimer);
    }
  }, [isAuthenticated, submitting, router, callbackUrl, messageApi]);

  const onFinish = async (values) => {
    try {
      setError('');
      setSubmitting(true);
      
      const success = await login(values.username, values.password);
      
      if (success) {
        messageApi.success({
          content: 'Login successful! Redirecting...',
          duration: 1.5,
        });
        
        // Redirect after a short delay to allow the message to be seen.
        // The useEffect above will also catch isAuthenticated and redirect,
        // but this ensures a redirect specifically after this action.
        const redirectTimer = setTimeout(() => {
          router.replace(decodeURIComponent(callbackUrl));
        }, 1000); 
        // No need to clear this timer, it's a one-off for this successful login action.
      }
    } catch (error) {
      console.error('Login error:', error);
      
      let errorMessage = 'Login failed. Please check your credentials.';
      
      // Extract detailed error message if available
      if (error.response && error.response.data) {
        errorMessage = error.response.data.detail || errorMessage;
      }
      
      setError(errorMessage);
      messageApi.error(errorMessage);
    } finally {
      setSubmitting(false);
    }
  };

  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };
  
  // Validation rules
  const usernameRules = [
    { required: true, message: 'Please enter your username or email' },
    { min: 3, message: 'Username must be at least 3 characters' }
  ];
  
  const passwordRules = [
    { required: true, message: 'Please enter your password' },
    { min: 4, message: 'Password must be at least 4 characters' }
  ];

  return (
    <>
      {contextHolder}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)'
      }}>
        <Card 
          style={{ 
            width: 400, 
            boxShadow: '0 8px 16px rgba(0,0,0,0.12)', 
            borderRadius: '12px',
            padding: '10px'
          }}
        >
          <div style={{ textAlign: 'center', marginBottom: '24px' }}>
            <Space align="center" direction="vertical" size={8}>
              <SafetyOutlined style={{ fontSize: '36px', color: '#1890ff' }} />
              <Title level={2} style={{ margin: '8px 0' }}>Traffic Monitoring System</Title>
              <Text type="secondary">Sign in to access the system</Text>
            </Space>
          </div>
          
          {error && (
            <Alert 
              message="Authentication Error" 
              description={error} 
              type="error" 
              showIcon 
              style={{ marginBottom: '20px' }} 
              closable
              onClose={() => setError('')}
            />
          )}
          
          <Form
            form={form}
            name="login"
            layout="vertical"
            onFinish={onFinish}
            autoComplete="off"
            size="large"
            requiredMark={false}
          >
            <Form.Item name="username" rules={usernameRules} hasFeedback>
              <Input 
                prefix={<UserOutlined style={{ color: '#bfbfbf' }} />} 
                placeholder="Username or Email" 
                disabled={submitting || loading}
                autoFocus
              />
            </Form.Item>
            
            <Form.Item name="password" rules={passwordRules} hasFeedback>
              <Input.Password
                prefix={<LockOutlined style={{ color: '#bfbfbf' }} />}
                placeholder="Password"
                disabled={submitting || loading}
                type={showPassword ? 'text' : 'password'}
                iconRender={(visible) => (visible ? <EyeOutlined /> : <EyeInvisibleOutlined />)}
              />
            </Form.Item>
            
            <Form.Item>
              <Button 
                type="primary" 
                htmlType="submit" 
                block
                loading={submitting || loading}
                style={{ height: '44px' }}
              >
                Sign In
              </Button>
            </Form.Item>
          </Form>
          
          <Divider plain><Text type="secondary">System Information</Text></Divider>
          
          <Text type="secondary" style={{ display: 'block', textAlign: 'center', fontSize: '13px' }}>
            Access is restricted to authorized personnel.
            Account registration is managed by administrators.
          </Text>
          
          <Text type="secondary" style={{ display: 'block', textAlign: 'center', fontSize: '13px', marginTop: '8px' }}>
            For assistance, please contact system support.
          </Text>
        </Card>
      </div>
    </>
  );
}
