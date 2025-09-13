"use client";

import { useState, useEffect } from 'react';
import 'leaflet/dist/leaflet.css';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import { HeatmapLayer } from 'react-leaflet-heatmap-layer-v3';
import L from 'leaflet';

// Fix Leaflet icons issue in Next.js
import DefaultIcon from 'leaflet/dist/images/marker-icon.png';
import DefaultIconRetina from 'leaflet/dist/images/marker-icon-2x.png';
import DefaultShadow from 'leaflet/dist/images/marker-shadow.png';

// Default Vietnam location (center of Ho Chi Minh City)
const DEFAULT_CENTER = [10.775, 106.702]; 
const DEFAULT_ZOOM = 13;

const ForecastMap = ({ points = [], center = DEFAULT_CENTER, zoom = DEFAULT_ZOOM }) => {
  const [mapReady, setMapReady] = useState(false);

  // Fix Leaflet icon issue when running in Next.js
  useEffect(() => {
    // This effect only needs to run once on component mount
    delete L.Icon.Default.prototype._getIconUrl;
    L.Icon.Default.mergeOptions({
      iconUrl: DefaultIcon.src,
      iconRetinaUrl: DefaultIconRetina.src,
      shadowUrl: DefaultShadow.src,
    });
    
    // Mark map as ready
    setMapReady(true);
  }, []);

  // Format points for heatmap layer
  const heatmapPoints = points.map(point => [
    point.lat,
    point.lng,
    point.value
  ]);

  return (
    <div style={{ height: '100%', width: '100%' }}>
      <MapContainer 
        center={center} 
        zoom={zoom} 
        style={{ height: '100%', width: '100%' }}
        scrollWheelZoom={true}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        
        {/* Heatmap layer for traffic density */}
        {mapReady && heatmapPoints.length > 0 && (
          <HeatmapLayer
            points={heatmapPoints}
            longitudeExtractor={p => p[1]}
            latitudeExtractor={p => p[0]}
            intensityExtractor={p => p[2]}
            radius={20}
            max={10}
            minOpacity={0.3}
            gradient={{
              0.2: '#4caf50', // Low traffic (green)
              0.4: '#8bc34a', // Light traffic (light green)
              0.6: '#ffeb3b', // Moderate traffic (yellow)
              0.8: '#ff9800', // Heavy traffic (orange)
              1.0: '#f44336'  // Severe traffic (red)
            }}
          />
        )}
        
        {/* Center marker */}
        <Marker position={center}>
          <Popup>
            Selected Area
          </Popup>
        </Marker>
      </MapContainer>
    </div>
  );
};

export default ForecastMap;