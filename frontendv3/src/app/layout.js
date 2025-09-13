'use client';

import { AuthProvider } from '@/context/authContext';
import { ThemeProvider } from '@/theme/themeConfig';
import { AntdRegistry } from '@ant-design/nextjs-registry';
import { React } from 'react';
import { App } from 'antd';
import './globals.css';

const RootLayout = ({ children }) => {
  return (
    <html lang="en">
      <body>
        <ThemeProvider>
          <AntdRegistry>
            <App>
              <AuthProvider>
                {children}
              </AuthProvider>
            </App>
          </AntdRegistry>
        </ThemeProvider>
      </body>
    </html>
  );
}


export default RootLayout;