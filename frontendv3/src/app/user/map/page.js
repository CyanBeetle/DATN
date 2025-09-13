"use client";

import { useAuth } from '@/context/authContext';
import {
  message,
  Spin,
} from 'antd';
import dynamic from 'next/dynamic';
import { useEffect, useState } from 'react';

// Dynamically import the new TrafficMap component
const TrafficMap = dynamic(() => import('@/components/map/TrafficMap'), {
  ssr: false,
  loading: () => <div style={{ height: 'calc(100vh - 64px)', display: 'flex', justifyContent: 'center', alignItems: 'center' }}><Spin size="large" tip="Loading Map..." /></div>
});

// Default center for Ho Chi Minh City
const INITIAL_CENTER = [10.866075511519412, 106.80244224301858];
const INITIAL_ZOOM = 15;

export default function MapPage() {
  const { isAuthenticated, user } = useAuth(); // using isAuthenticated from useAuth
  const [messageApi, contextHolder] = message.useMessage();
  const [loadingPage, setLoadingPage] = useState(true);

  useEffect(() => {
    if (isAuthenticated) {
      messageApi.success(`Welcome ${user?.displayName || 'User'} to the Traffic Monitoring System!`);
      setLoadingPage(false);
    }
    // If not authenticated, TrafficMap component itself will show an auth required message.
    // Or, middleware should redirect. If we reach here and not authenticated, it might be a brief moment before redirect.
    // We can also add a loading spinner here until isAuthenticated is confirmed.

  }, [isAuthenticated, user, messageApi]);

  // Show a loading spinner for the page until authentication status is known
  // and initial welcome message is potentially shown.
  // TrafficMap handles its own internal loading for map data.
  if (typeof isAuthenticated === 'undefined' || (isAuthenticated === false && loadingPage)) {
    // This condition tries to show loading until auth is determined
    // If not authenticated, TrafficMap shows its own message, or user is redirected.
    return (
      <div style={{ height: 'calc(100vh - 64px)', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <Spin size="large" />
      </div>
    );
  }

  return (
    <>
      {contextHolder}
      <div style={{ height: 'calc(100vh - 64px)', position: 'relative' }}>
        <TrafficMap
          initialView={{
            center: INITIAL_CENTER,
            zoom: INITIAL_ZOOM
          }}
        />
      </div>
    </>
  );
}