"use client";

// import { useAuth } from '@/context/authContext'; // Removed
import { CarOutlined, AppstoreOutlined } from '@ant-design/icons';
// import { Button, Spin, Tooltip } from 'antd'; // Spin will be removed if not used elsewhere
import { Button, Tooltip, Spin } from 'antd'; // Keep Spin for loading map
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import dynamic from 'next/dynamic';
import { useEffect, useState } from 'react';
import { MapContainer, TileLayer, ZoomControl, useMap } from 'react-leaflet'; // Added useMap back
import axiosInstance from '@/utils/axiosInstance'; // Re-enabled
import CameraMarkers from '@/components/map/CameraMarkers'; // Re-enabled
// import CameraMarkers from '@/components/map/CameraMarkers'; // Still commented out

// import generateMockCameras from '@/utils/mockCameraData';

// Dynamically import RouteFinder to avoid SSR issues
const RouteFinder = dynamic(() => import('./RouteFinder'), {
  ssr: false
});

// Dynamically import HeatmapLayer to avoid SSR issues
const HeatmapLayer = dynamic(() => import('./HeatmapLayer'), {
    ssr: false,
    loading: () => <p>Loading heatmap...</p> // Optional loading state
});

// Configure Leaflet default icon path
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
});

// Internal component to handle map invalidation
const MapInvalidator = () => {
  const map = useMap();
  useEffect(() => {
    console.log("MapInvalidator: Effect triggered, calling invalidateSize.");
    map.invalidateSize();
  }, [map]); // Re-run if map instance changes (should be stable)
  return null; // This component does not render anything
};

const TrafficMap = ({ initialView = { center: [10.772764, 106.679060], zoom: 13 } }) => {
  const [loadingMap, setLoadingMap] = useState(true); // Renamed for clarity from initial map load
  const [loadingCameras, setLoadingCameras] = useState(true); // New state for camera loading
  const [showRouteFinder, setShowRouteFinder] = useState(false);
  const [isModal, setIsModal] = useState(false);
  const [cameras, setCameras] = useState([]); // Re-enabled
  const [showHeatmap, setShowHeatmap] = useState(false); // State for heatmap visibility
  // const [mapInstance, setMapInstance] = useState(null); // Removed as whenCreated was unreliable here

  useEffect(() => {
    const timer = setTimeout(() => {
      setLoadingMap(false);
    }, 1000);
    return () => clearTimeout(timer);
  }, []);

  // Re-enabled useEffect for fetching camera data
  useEffect(() => {
    const fetchCameraData = async () => {
      setLoadingCameras(true);
      try {
        console.log('TrafficMap: Attempting to fetch cameras via axiosInstance...');
        const response = await axiosInstance.get('/api/cameras');
        // console.log('TrafficMap: RAW camera data from backend:', JSON.stringify(response.data, null, 2)); // Log raw data
        
        if (response.data) {
          const transformedCameras = response.data.map(cam => {
            // Add detailed logging for each camera being transformed
            // console.log('TrafficMap: Transforming camera:', JSON.stringify(cam, null, 2));
            const lat = cam.location_detail?.latitude;
            const lng = cam.location_detail?.longitude;
            // console.log(`TrafficMap: Extracted lat: ${lat}, lng: ${lng} for camera ID: ${cam.id}`);
            return {
              id: cam.id,
              name: cam.name,
              lat: lat, 
              lng: lng, 
              status: cam.status,
              congestion_level: cam.congestion_level === null || cam.congestion_level === undefined ? 0 : cam.congestion_level,
              // intensity: Math.floor(Math.random() * 101) // Random intensity 0-100 for heatmap
              intensity: (cam.congestion_level === null || cam.congestion_level === undefined ? 0 : cam.congestion_level) * 20 // Scale 0-5 to 0-100
            };
          }); 
          console.log('TrafficMap: Transformed camera data before filter (with intensity):', JSON.stringify(transformedCameras, null, 2));
          
          const validCameras = transformedCameras.filter(cam => cam.lat != null && cam.lng != null);
          console.log('TrafficMap: Filtered camera data (with valid lat/lng):', JSON.stringify(validCameras, null, 2));
          setCameras(validCameras);

        } else {
          console.warn('TrafficMap: Fetched camera data is not an array or is empty:', response.data);
          setCameras([]);
        }
      } catch (error) {
        console.error('TrafficMap: Error fetching camera data:', error);
        if (error.response) {
          console.error('TrafficMap: Error response data:', error.response.data);
          console.error('TrafficMap: Error response status:', error.response.status);
        } else if (error.request) {
          console.error('TrafficMap: Error request:', error.request);
        } else {
          console.error('TrafficMap: Error message:', error.message);
        }
        setCameras([]);
      } finally {
        setLoadingCameras(false);
      }
    };

    fetchCameraData();
  }, []);

  // using mockCameraData
  // useEffect(() => {
  //   setLoadingCameras(true);
  //   const mockData = generateMockCameras();
  //   console.log('TrafficMap: MOCK camera data:', JSON.stringify(mockData, null, 2));

  //   const transformedCameras = mockData.map(cam => {
  //     const lat = cam.location_detail?.latitude; // Lưu ý: Sử dụng location_detail.latitude
  //     const lng = cam.location_detail?.longitude; // Lưu ý: Sử dụng location_detail.longitude
  //     return {
  //       id: cam.id,
  //       name: cam.name,
  //       lat: lat,
  //       lng: lng,
  //       status: cam.status,
  //       congestion_level: cam.congestion_level === null || cam.congestion_level === undefined ? 0 : cam.congestion_level,
  //       intensity: (cam.congestion_level === null || cam.congestion_level === undefined ? 0 : cam.congestion_level) * 20
  //     };
  //   });

  //   const validCameras = transformedCameras.filter(cam => cam.lat != null && cam.lng != null);
  //   console.log("validCameras: ", JSON.stringify(validCameras, null, 2));
  //   setCameras(validCameras);
  //   setLoadingCameras(false);
  // }, []);

  const handleMapCreated = (map) => { // Removed this approach
    console.log("TrafficMap: Map instance created by whenCreated:", map);
    setMapInstance(map);
  };

  const showModal = () => {
    setIsModal(!isModal);
  };

  const toggleHeatmap = () => {
    setShowHeatmap(prevState => !prevState);
  };

  return (
    <div style={{ position: 'relative', height: '100%', width: '100%' }}>
      {(loadingMap || loadingCameras) && ( // Show spinner if map OR cameras are loading
        <div style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: 'rgba(245, 245, 245, 0.75)',
          zIndex: 50
        }}>
          <Spin size="large" tip={loadingMap ? "Loading map..." : "Loading camera data..."} />
        </div>
      )}

      {!loadingMap && (
        <MapContainer
          // whenCreated={handleMapCreated} // Removed
          center={initialView.center}
          zoom={initialView.zoom}
          style={{ height: '100%', width: '100%' }}
          zoomControl={false}
          attributionControl={true}
        >
          <TileLayer
            url="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}"
            attribution='&copy; <a href="https://maps.google.com">Google Maps</a>'
          />
          <ZoomControl position="bottomright" />
          <RouteFinder 
            isModal={isModal} 
            cameras={cameras}
          />
          {cameras.length > 0 && <CameraMarkers cameras={cameras} />}
          {showHeatmap && cameras.length > 0 && <HeatmapLayer data={cameras} />}
          <MapInvalidator /> {/* Add the invalidator component here */}
          {/* {cameras.length > 0 && <CameraMarkers cameras={cameras} />} */}
        </MapContainer>
      )}

      <div style={{
        position: 'absolute',
        top: '20px',
        right: '20px',
        zIndex: 1000
      }}>
        <Tooltip title="Find Route">
          <Button
            type={showRouteFinder ? "primary" : "default"}
            shape="circle"
            icon={<CarOutlined />}
            onClick={showModal}
            style={{ marginBottom: '10px' }}
          />
        </Tooltip>
        <Tooltip title={showHeatmap ? "Hide Heatmap" : "Show Heatmap"}>
            <Button
                type={showHeatmap ? "primary" : "default"}
                shape="circle"
                icon={<AppstoreOutlined />}
                onClick={toggleHeatmap}
                style={{ marginBottom: '10px' }}
            />
        </Tooltip>
      </div>
    </div>
  );
};

export default TrafficMap;