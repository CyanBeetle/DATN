"use client";

import L from 'leaflet';
// import 'leaflet.heat';

import { useEffect } from 'react';
import { useMap } from 'react-leaflet';
import 'leaflet.heat';

const HeatmapLayer = ({ data }) => {
  const map = useMap();

  useEffect(() => {
    if (!data || data.length === 0) return;

    // Configure heatmap options
    const heatmapOptions = {
      radius: 25,
      blur: 15,
      maxZoom: 17,
      gradient: {
        0.0: 'green',    // Low congestion (good)
        0.3: 'yellow',   // Moderate congestion
        0.6: 'orange',   // High congestion
        0.8: 'red'       // Severe congestion
      }
      // minOpacity: 0.05,
      // maxZoom: 17,
      // radius: 25,
      // blur: 15,
      // max: 1.0,
      // gradient: {
      //   0.0: 'green',    // Low congestion (good)
      //   0.3: 'yellow',   // Moderate congestion
      //   0.6: 'orange',   // High congestion
      //   0.8: 'red'       // Severe congestion
      // }
    };

    const heatPoints = data.map(point => [
      point.lat,
      point.lng,
      point.intensity / 100 || 0  // Normalize intensity to 0-1 range
    ]);

    // Create heatmap layer
    const heatLayer = L.heatLayer(heatPoints, heatmapOptions).addTo(map);

    // Cleanup function to remove layer when component unmounts
    return () => {
      map.removeLayer(heatLayer);
    };
  }, [map, data]);

  // This component doesn't render anything directly
  return null;
};

export default HeatmapLayer;