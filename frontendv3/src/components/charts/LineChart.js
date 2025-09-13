'use client';

import { useEffect, useRef } from 'react';
import Chart from 'chart.js/auto';

/**
 * Line Chart component using Chart.js
 * 
 * @param {Object} props - Component props
 * @param {Object} props.data - Chart.js data object
 * @param {Object} props.options - Chart.js options object
 * @returns {JSX.Element} Line chart component
 */
export default function LineChart({ data, options = {} }) {
  const chartRef = useRef(null);
  const chartInstance = useRef(null);

  useEffect(() => {
    // If we already have a chart instance, destroy it
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }

    // If we have data and a ref to the canvas
    if (data && chartRef.current) {
      // Create a new chart instance
      const ctx = chartRef.current.getContext('2d');
      chartInstance.current = new Chart(ctx, {
        type: 'line',
        data: data,
        options: {
          responsive: true,
          maintainAspectRatio: false,
          ...options
        }
      });
    }

    // Clean up function
    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, [data, options]);

  return (
    <canvas ref={chartRef} />
  );
}