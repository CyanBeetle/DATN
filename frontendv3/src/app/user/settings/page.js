"use client";
import { useState, useEffect } from "react";
import axiosInstance from "@/utils/axiosInstance";
import { message } from "antd";
import { useAuth } from "@/context/authContext";
import { Form, Input, Button, Card, Alert, Spin, Typography, Divider } from "antd";
import {
  UserOutlined,
  LockOutlined,
  MailOutlined,
  SaveOutlined,
} from "@ant-design/icons";
import { useRouter } from 'next/navigation';

const { Title, Text } = Typography;

const SettingsPage = () => {
  const { user, isAuthenticated, fetchUser } = useAuth();
  const router = useRouter();
  const [passwordForm] = Form.useForm();
  const [messageApi, contextHolder] = message.useMessage();
  
  const [loadingPassword, setLoadingPassword] = useState(false);
  const [passwordError, setPasswordError] = useState(null);

  useEffect(() => {
    if (isAuthenticated === false) {
      router.push('/auth/login');
    }
  }, [isAuthenticated, router]);

  const handlePasswordChange = async (values) => {
    setLoadingPassword(true);
    setPasswordError(null);
    
    try {
      if (!values.oldPassword || !values.newPassword) {
        throw new Error("Please enter both current and new password");
      }

      const response = await axiosInstance.put(
        "/api/auth/me/password",
        { 
          old_password: values.oldPassword, 
          new_password: values.newPassword 
        }
      );

      if (response.status === 200) {
        messageApi.success(response.data.message || "Password changed successfully!");
        passwordForm.resetFields();
      } else {
        throw new Error(response.data?.detail || "Failed to change password");
      }
    } catch (error) {
      console.error('Error changing password:', error);
      let errorMessage = 'Failed to change password. Please try again.';
      if (axiosInstance.isAxiosError(error) && error.response) {
        errorMessage = error.response.data?.detail || error.message;
      } else if (error instanceof Error) {
        errorMessage = error.message;
      }
      setPasswordError(errorMessage);
      messageApi.error(errorMessage);
    } finally {
      setLoadingPassword(false);
    }
  };

  if (isAuthenticated === undefined) {
    return (
      <div style={{ padding: '24px', textAlign: 'center' }}>
        <Spin size="large" tip="Loading account settings..." />
      </div>
    );
  }

  return (
    <div style={{ padding: '24px', maxWidth: '600px', margin: '0 auto' }}>
      {contextHolder}
      <Title level={2} style={{ marginBottom: '24px' }}>Account Settings</Title>
      
      <Card title="User Information" style={{ marginBottom: '24px' }}>
        <Form layout="vertical">
          <Form.Item label="Username">
            <Input prefix={<UserOutlined />} value={user?.username || 'N/A'} disabled />
          </Form.Item>
          <Form.Item label="Email">
            <Input prefix={<MailOutlined />} value={user?.email || 'N/A'} disabled />
          </Form.Item>
          <Form.Item label="Role">
            <Input prefix={<UserOutlined />} value={user?.role || 'N/A'} disabled />
          </Form.Item>
        </Form>
      </Card>

      <Card title="Change Password">
        {passwordError && (
          <Alert 
            message={passwordError} 
            type="error" 
            showIcon 
            style={{ marginBottom: '16px' }}
            closable
            onClose={() => setPasswordError(null)}
          />
        )}
        
        <Form
          form={passwordForm}
          layout="vertical"
          onFinish={handlePasswordChange}
        >
          <Form.Item
            name="oldPassword"
            label="Current Password"
            rules={[
              { required: true, message: 'Please enter your current password' }
            ]}
          >
            <Input.Password 
              prefix={<LockOutlined />} 
              placeholder="Enter your current password" 
            />
          </Form.Item>
          
          <Form.Item
            name="newPassword"
            label="New Password"
            rules={[
              { required: true, message: 'Please enter a new password' }
            ]}
            hasFeedback
          >
            <Input.Password 
              prefix={<LockOutlined />} 
              placeholder="Enter your new password" 
            />
          </Form.Item>
          
          <Form.Item
            name="confirmPassword"
            label="Confirm New Password"
            dependencies={['newPassword']}
            hasFeedback
            rules={[
              { required: true, message: 'Please confirm your new password' },
              ({ getFieldValue }) => ({
                validator(_, value) {
                  if (!value || getFieldValue('newPassword') === value) {
                    return Promise.resolve();
                  }
                  return Promise.reject(new Error('The two passwords do not match'));
                },
              }),
            ]}
          >
            <Input.Password 
              prefix={<LockOutlined />} 
              placeholder="Confirm your new password" 
            />
          </Form.Item>
          
          <Form.Item>
            <Button 
              type="primary" 
              htmlType="submit" 
              icon={<SaveOutlined />}
              loading={loadingPassword}
            >
              Change Password
            </Button>
          </Form.Item>
        </Form>
      </Card>
    </div>
  );
};

export default SettingsPage;