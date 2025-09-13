"use client";

import { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { Tooltip } from 'antd';

const ForecastMap = ({ center, zoom, points = [] }) => {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const heatmapLayerRef = useRef(null);
  const markersLayerRef = useRef(null);

  // Initialize map on component mount
  useEffect(() => {
    // Fix for marker icon issues in Next.js
    delete L.Icon.Default.prototype._getIconUrl;
    L.Icon.Default.mergeOptions({
      iconRetinaUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png',
      iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
      shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
    });

    if (!mapInstanceRef.current && mapRef.current) {
      // Create map instance
      mapInstanceRef.current = L.map(mapRef.current).setView(center, zoom);
      
      // Add tile layer (OpenStreetMap)
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19,
      }).addTo(mapInstanceRef.current);
      
      // Create layers for organization
      markersLayerRef.current = L.layerGroup().addTo(mapInstanceRef.current);
    }
    
    // Clean up on unmount
    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);
  
  // Update map center and zoom when props change
  useEffect(() => {
    if (mapInstanceRef.current && center && zoom) {
      mapInstanceRef.current.setView(center, zoom);
    }
  }, [center, zoom]);
  
  // Update points when data changes
  useEffect(() => {
    if (mapInstanceRef.current && points && points.length > 0) {
      // Clear existing markers
      if (markersLayerRef.current) {
        markersLayerRef.current.clearLayers();
      }
      
      // Add new markers
      points.forEach(point => {
        const marker = createCongestionMarker(point);
        if (marker) {
          markersLayerRef.current.addLayer(marker);
        }
      });
      
      // Try to load and update heatmap if it's available
      updateHeatmapLayer(points);
    }
  }, [points]);
  
  // Create a congestion marker for a point
  const createCongestionMarker = (point) => {
    if (!point.lat || !point.lng || !point.level) return null;
    
    // Get color based on congestion level
    const color = getCongestionColor(point.level);
    
    // Create custom circular marker
    const marker = L.circleMarker([point.lat, point.lng], {
      radius: 6,
      fillColor: color,
      color: '#fff',
      weight: 1,
      opacity: 1,
      fillOpacity: 0.8
    });
    
    // Add popup with info
    if (point.location) {
      marker.bindTooltip(`
        <b>${point.location}</b><br>
        Congestion: ${getCongestionText(point.level)}
      `);
    }
    
    return marker;
  };
  
  // Update the heatmap layer
  const updateHeatmapLayer = (points) => {
    // If leaflet.heat is available, use it
    if (window.L.heatLayer) {
      // Convert points to heatmap format
      const heatmapData = points.map(point => [
        point.lat,
        point.lng,
        point.level / 5  // Normalize level to 0-1 range
      ]);
      
      // Clear existing heatmap layer
      if (heatmapLayerRef.current) {
        mapInstanceRef.current.removeLayer(heatmapLayerRef.current);
      }
      
      // Create new heatmap layer
      heatmapLayerRef.current = L.heatLayer(heatmapData, {
        radius: 20,
        blur: 15,
        maxZoom: 17,
        gradient: {
          0.0: '#4caf50', // Free flowing
          0.25: '#8bc34a', // Light
          0.5: '#ffeb3b',  // Moderate
          0.75: '#ff9800', // Heavy
          1.0: '#f44336'   // Very heavy
        }
      }).addTo(mapInstanceRef.current);
      
    } else {
      // Attempt to load the plugin dynamically
      console.log('Leaflet heatmap plugin not available, trying to load it...');
      
      const script = document.createElement('script');
      script.src = 'https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js';
      script.async = true;
      
      script.onload = () => {
        console.log('Heatmap plugin loaded, updating map...');
        updateHeatmapLayer(points);
      };
      
      document.head.appendChild(script);
    }
  };
  
  // Get congestion color based on level
  const getCongestionColor = (level) => {
    switch (Math.round(level)) {
      case 1: return '#4caf50'; // Free flowing
      case 2: return '#8bc34a'; // Light
      case 3: return '#ffeb3b'; // Moderate
      case 4: return '#ff9800'; // Heavy
      case 5: return '#f44336'; // Very heavy
      default: return '#757575'; // Unknown
    }
  };
  
  // Get congestion text based on level
  const getCongestionText = (level) => {
    switch (Math.round(level)) {
      case 1: return 'Free flowing';
      case 2: return 'Light traffic';
      case 3: return 'Moderate traffic';
      case 4: return 'Heavy traffic';
      case 5: return 'Severe congestion';
      default: return 'Unknown';
    }
  };

  return <div ref={mapRef} style={{ width: '100%', height: '100%' }}></div>;
};

export default ForecastMap;