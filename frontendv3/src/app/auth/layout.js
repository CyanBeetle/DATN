"use client";

import React from 'react';
import { AntdRegistry } from '@ant-design/nextjs-registry';
import { Layout } from 'antd';
const { Content } = Layout;

const AuthLayout = ({ children }) => {
  return (
    <Layout>
      <Content style={{height:'100vh', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
      <AntdRegistry>{children}</AntdRegistry>
      </Content>
    </Layout>
  );
} 

export default AuthLayout;