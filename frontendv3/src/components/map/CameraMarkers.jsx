"use client";

import { useMemo } from 'react';
import { Marker, Popup } from 'react-leaflet';
import { Typography, Tag } from 'antd';
import L from 'leaflet';

const { Text, Title } = Typography;

// Create custom camera icon
const createCameraIcon = (status) => {
  // console.log(`CameraMarkers: createCameraIcon called with status: ${status}`);
  const color = status === 'Active' ? '#4caf50' :
                status === 'Maintenance' ? '#ff9800' : '#757575';
  
  const icon = L.divIcon({
    className: 'camera-marker',
    html: `<div style="color:${color}; font-size: 24px;"><svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 0 24 24" width="24" fill="currentColor"><path d="M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4z"/></svg></div>`,
    iconSize: [24, 24],
    iconAnchor: [12, 24],
    popupAnchor: [0, -24]
  });
  // console.log(`CameraMarkers: icon created for status ${status}:`, icon);
  return icon;
};

// Helper function to map camera status to Ant Design Tag colors
const getStatusColor = (status) => {
  switch (status) {
    case 'Active':
      return 'success';
    case 'Maintenance':
      return 'warning';
    case 'Inactive':
      return 'error';
    default:
      return 'default';
  }
};

// Helper function to get congestion text and color
const getCongestionInfo = (level) => {
  const texts = {
    1: "Free flowing",
    2: "Light traffic",
    3: "Moderate traffic",
    4: "Heavy traffic",
    5: "Very heavy / Jammed",
  };
  const colors = {
    1: '#52c41a',
    2: '#8bc34a',
    3: '#fadb14',
    4: '#fa8c16',
    5: '#f5222d',
  };
  const text = texts[level] || "Unknown";
  const color = colors[level] || 'default';
  return { text, color, level };
};

const CameraMarkers = ({ cameras }) => {
  console.log("CameraMarkers: Component received cameras prop:", JSON.stringify(cameras));

  const markers = useMemo(() => {
    console.log("CameraMarkers: useMemo for markers triggered. Input cameras:", JSON.stringify(cameras));
    if (!cameras || !Array.isArray(cameras)) {
      console.log("CameraMarkers: useMemo - cameras prop is null, undefined, or not an array. Returning [].");
      return [];
    }
    const processed = cameras.map((camera, index) => {
      console.log(`CameraMarkers: useMemo - Processing camera ${index + 1}/${cameras.length}:`, JSON.stringify(camera));
      if (camera.lat == null || camera.lng == null) {
        console.warn(`CameraMarkers: useMemo - Camera ${camera.id || 'Unknown ID'} at index ${index} has null/undefined lat or lng. Skipping.`);
        return null; // Will be filtered out
      }
      return {
        ...camera,
        icon: createCameraIcon(camera.status)
      };
    }).filter(Boolean); // Filter out any nulls from skipped cameras
    console.log("CameraMarkers: useMemo - Processed markers array:", JSON.stringify(processed));
    return processed;
  }, [cameras]);
  
  if (!markers || markers.length === 0) {
    console.log("CameraMarkers: No valid markers to display after processing. cameras prop was:", JSON.stringify(cameras));
    return null;
  }

  console.log(`CameraMarkers: Rendering ${markers.length} markers.`);

  return (
    <>
      {markers.map((camera, index) => {
        // console.log(`CameraMarkers: Mapping marker ${index + 1} to JSX:`, JSON.stringify(camera));
        const congestion = getCongestionInfo(camera.congestion_level);
        // console.log(`CameraMarkers: Marker ${index + 1} congestion info:`, congestion);

        if (camera.lat == null || camera.lng == null) {
          console.warn(`CameraMarkers: Rendering - Skipping marker for camera ${camera.id || 'Unknown ID'} due to null/undefined lat or lng.`);
          return null; // Should have been caught by filter(Boolean) in useMemo, but as a safeguard.
        }
        
        return (
          <Marker 
            key={camera.id || `marker-${index}`} // Fallback key
            position={[camera.lat, camera.lng]} 
            icon={camera.icon}
          >
            <Popup>
              <div style={{ minWidth: 180, padding: '4px', fontSize: '12px' }}>
                <Title level={5} style={{ marginBottom: '2px', fontSize: '13px' }}>ID: {camera.id}</Title>
                <Text type="secondary" style={{ display: 'block', marginBottom: '4px' }}>
                  Lat: {camera.lat?.toFixed(4)}, Lng: {camera.lng?.toFixed(4)}
                </Text>
                <div style={{ marginTop: '4px' }}>
                    <Text>Congestion: </Text>
                    <Tag color={congestion.color}>
                        Level {congestion.level} - {congestion.text}
                    </Tag>
                </div>
              </div>
            </Popup>
          </Marker>
        );
      })}
    </>
  );
};

export default CameraMarkers;