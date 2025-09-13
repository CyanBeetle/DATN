'use client';

import { Alert, Button, Card, Select } from 'antd';
import L from 'leaflet';
import 'leaflet-routing-machine';
import debounce from 'lodash.debounce'; // cần cài: yarn add lodash.debounce
import { useCallback, useRef, useState, useEffect } from 'react';
import { useMap } from 'react-leaflet';
import { message as antdMessage } from 'antd'; // Import message API from antd

const { Option } = Select;

// Helper to calculate distance between two lat/lng points in meters
function haversineDistance(coords1, coords2) {
  function toRad(x) {
    return x * Math.PI / 180;
  }
  const R = 6371e3; // Earth radius in meters
  const dLat = toRad(coords2.lat - coords1.lat);
  const dLon = toRad(coords2.lng - coords1.lng);
  const lat1 = toRad(coords1.lat);
  const lat2 = toRad(coords2.lat);

  const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.sin(dLon / 2) * Math.sin(dLon / 2) * Math.cos(lat1) * Math.cos(lat2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

// Helper to determine effect radius based on congestion level
function getEffectRadius(congestionLevel) {
  if (congestionLevel >= 5) return 800; // meters
  if (congestionLevel >= 4) return 500;
  if (congestionLevel >= 3) return 300;
  return 100; // Level 0-2 or undefined
}

async function geocode(address) {
  try {
    const res = await fetch(
      `https://nominatim.openstreetmap.org/search?format=json&limit=5&countrycodes=vn&q=${encodeURIComponent(address)}`
    );
    const data = await res.json();
    return data.map((item) => ({
      value: item.display_name,
      label: item.display_name,
      coordinates: { lat: parseFloat(item.lat), lng: parseFloat(item.lon) }
    }));
  } catch (error) {
    console.error('Geocoding error:', error);
    return [];
  }
}

const RouteFinder = ({ isModal, cameras }) => {
  const [startPoint, setStartPoint] = useState('');
  const [endPoint, setEndPoint] = useState('');
  const [startOptions, setStartOptions] = useState([]);
  const [endOptions, setEndOptions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const map = useMap();
  const routingControlRef = useRef(null);
  const [isSettingStart, setIsSettingStart] = useState(false);
  const [isSettingEnd, setIsSettingEnd] = useState(false);
  const [startMarker, setStartMarker] = useState(null);
  const [endMarker, setEndMarker] = useState(null);

  const geocodePoint = async (latlng) => {
    try {
      const res = await fetch(
        `https://nominatim.openstreetmap.org/reverse?format=json&lat=${latlng.lat}&lon=${latlng.lng}`
      );
      const data = await res.json();
      return data.display_name || `Lat: ${latlng.lat.toFixed(4)}, Lng: ${latlng.lng.toFixed(4)}`;
    } catch (error) {
      console.error('Reverse geocoding error:', error);
      return `Lat: ${latlng.lat.toFixed(4)}, Lng: ${latlng.lng.toFixed(4)}`;
    }
  };

  useEffect(() => {
    if (!map || (!isSettingStart && !isSettingEnd)) {
      map.getContainer().style.cursor = 'grab';
      return;
    }

    map.getContainer().style.cursor = 'crosshair';

    const handleMapClick = async (e) => {
      const { lat, lng } = e.latlng;
      const displayName = await geocodePoint(e.latlng);

      if (isSettingStart) {
        setStartPoint(displayName);
        setStartOptions([{ value: displayName, label: displayName, coordinates: { lat, lng } }]);
        if (startMarker) map.removeLayer(startMarker);
        const newStartMarker = L.marker([lat, lng]).addTo(map).bindPopup("Start Point");
        setStartMarker(newStartMarker);
        setIsSettingStart(false);
      } else if (isSettingEnd) {
        setEndPoint(displayName);
        setEndOptions([{ value: displayName, label: displayName, coordinates: { lat, lng } }]);
        if (endMarker) map.removeLayer(endMarker);
        const newEndMarker = L.marker([lat, lng]).addTo(map).bindPopup("End Point");
        setEndMarker(newEndMarker);
        setIsSettingEnd(false);
      }
      map.getContainer().style.cursor = 'grab'; // Reset cursor
    };

    map.on('click', handleMapClick);

    return () => {
      map.off('click', handleMapClick);
      map.getContainer().style.cursor = 'grab'; // Ensure cursor is reset on cleanup
    };
  }, [map, isSettingStart, isSettingEnd, geocodePoint, startMarker, endMarker]);

  const handleSearch = useCallback(
    debounce(async (value, isStart) => {
      if (!value) return;
      const results = await geocode(value);
      if (isStart) setStartOptions(results);
      else setEndOptions(results);
    }, 500),
    []
  );

  const handleFindRoute = () => {
    if (!startPoint || !endPoint) {
      setError('Please select both start and end points.');
      return;
    }
    setLoading(true);
    setError('');

    const startSelection = startOptions.find((opt) => opt.value === startPoint);
    const endSelection = endOptions.find((opt) => opt.value === endPoint);
    if (!startSelection || !endSelection) {
      setError('Unable to geocode the selected locations.');
      setLoading(false);
      return;
    }

    const startCoords = startSelection.coordinates;
    const endCoords = endSelection.coordinates;

    if (routingControlRef.current) {
      map.removeControl(routingControlRef.current);
      routingControlRef.current = null;
    }

    const control = L.Routing.control({
      waypoints: [
        L.latLng(startCoords.lat, startCoords.lng),
        L.latLng(endCoords.lat, endCoords.lng)
      ],
      routeWhileDragging: true,
      showAlternatives: false, 
      fitSelectedRoutes: true,
      lineOptions: { styles: [{ color: 'blue', weight: 4 }] }
    }).addTo(map);

    control.on('routesfound', function (e) {
      const routes = e.routes;
      if (routes.length > 0) {
        const route = routes[0];
        console.log('Tổng chiều dài (m):', route.summary.totalDistance);
        console.log('Tổng thời gian (s):', route.summary.totalTime);
        // console.log('Danh sách tọa độ:', route.coordinates);

        // Check for congestion along the route
        let congestionWarningShown = false;
        if (cameras && cameras.length > 0) {
          for (const camera of cameras) {
            if (congestionWarningShown) break;
            if (camera.congestion_level >= 3) {
              const cameraEffectRadius = getEffectRadius(camera.congestion_level);
              for (const routePoint of route.coordinates) {
                const distanceToCamera = haversineDistance(
                  { lat: routePoint.lat, lng: routePoint.lng },
                  { lat: camera.lat, lng: camera.lng }
                );
                if (distanceToCamera <= cameraEffectRadius) {
                  antdMessage.warning(`Route passes near a congested area (Level ${camera.congestion_level}) around camera: ${camera.name || camera.id}.`, 5);
                  congestionWarningShown = true;
                  break; // Stop checking this camera's points
                }
              }
            }
          }
        }
      }
    });

    routingControlRef.current = control;
    setLoading(false);
  };

  const clearRoute = () => {
    if (routingControlRef.current) {
      map.removeControl(routingControlRef.current);
      routingControlRef.current = null;
    }
    setStartPoint('');
    setEndPoint('');
    setStartOptions([]);
    setEndOptions([]);
    if (startMarker) map.removeLayer(startMarker);
    if (endMarker) map.removeLayer(endMarker);
    setStartMarker(null);
    setEndMarker(null);
    setError('');
  };

  return (
    <>
      {
        isModal &&
        <Card
          title="Find Route"
          style={{ position: 'absolute', top: 20, left: 20, width: 360, zIndex: 1000 }
          }
        >
          {error && (
            <Alert
              message={error}
              type="error"
              showIcon
              closable
              onClose={() => setError('')}
              style={{ marginBottom: 12 }}
            />
          )}

          <div style={{ marginBottom: 12 }}>
            <label>Start Point</label>
            <Select
              showSearch
              placeholder="Enter start location"
              onSearch={(val) => handleSearch(val, true)}
              onChange={(val) => setStartPoint(val)}
              value={startPoint || undefined}
              options={startOptions}
              filterOption={false}
              allowClear
              style={{ width: '100%' }}
            />
            <Button onClick={() => { setIsSettingStart(true); setIsSettingEnd(false); }} style={{marginLeft: '8px'}}>
              Set on Map
            </Button>
          </div>

          <div style={{ marginBottom: 12 }}>
            <label>End Point</label>
            <Select
              showSearch
              placeholder="Enter destination"
              onSearch={(val) => handleSearch(val, false)}
              onChange={(val) => setEndPoint(val)}
              value={endPoint || undefined}
              options={endOptions}
              filterOption={false}
              allowClear
              style={{ width: '100%' }}
            />
            <Button onClick={() => { setIsSettingEnd(true); setIsSettingStart(false);}} style={{marginLeft: '8px'}}>
              Set on Map
            </Button>
          </div>

          <div style={{ display: 'flex', gap: 8 }}>
            <Button type="primary" block loading={loading} onClick={handleFindRoute}>
              Find Route
            </Button>
            <Button danger onClick={clearRoute}>
              Clear
            </Button>
          </div>
        </Card >
      }
    </>

  );
};

export default RouteFinder;