"use client";

import { useEffect, useRef, useState } from 'react';
import { Spin, Empty } from 'antd';

const AverageSpeedChart = ({ labels = [], data = [] }) => {
  const chartRef = useRef(null);
  const chartInstanceRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Load Chart.js dynamically to avoid SSR issues
  useEffect(() => {
    const loadChartJS = async () => {
      try {
        setLoading(true);
        
        // Try to load Chart.js dynamically
        if (!window.Chart) {
          const Chart = await import('chart.js/auto');
          window.Chart = Chart;
        }
        
        setLoading(false);
      } catch (err) {
        console.error('Error loading Chart.js:', err);
        setError('Failed to load chart library. Please try again later.');
        setLoading(false);
      }
    };
    
    loadChartJS();
  }, []);
  
  // Create or update chart when data changes
  useEffect(() => {
    if (loading || error || !labels.length || !data.length || !window.Chart) return;
    
    try {
      // Destroy existing chart if it exists
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
      }
      
      // Set up colors
      const primaryColor = 'rgba(24, 144, 255, 0.7)';
      const secondaryColor = 'rgba(114, 46, 209, 0.6)';
      
      // Create chart configuration
      const config = {
        type: 'line',
        data: {
          labels: labels,
          datasets: [
            {
              label: data[0].label,
              data: data[0].data,
              backgroundColor: 'rgba(24, 144, 255, 0.1)',
              borderColor: primaryColor,
              borderWidth: 2,
              pointBackgroundColor: primaryColor,
              pointBorderColor: '#fff',
              pointRadius: 4,
              pointHoverRadius: 6,
              tension: 0.3
            },
            {
              label: data[1]?.label || 'Secondary',
              data: data[1]?.data || [],
              backgroundColor: 'rgba(114, 46, 209, 0.1)',
              borderColor: secondaryColor,
              borderWidth: 2,
              pointBackgroundColor: secondaryColor,
              pointBorderColor: '#fff',
              pointRadius: 4,
              pointHoverRadius: 6,
              tension: 0.3
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: 'top',
              labels: {
                boxWidth: 12,
                usePointStyle: true,
                pointStyle: 'circle'
              }
            },
            tooltip: {
              mode: 'index',
              intersect: false,
              callbacks: {
                label: function(context) {
                  let label = context.dataset.label || '';
                  if (label) {
                    label += ': ';
                  }
                  
                  const value = context.parsed.y;
                  
                  // Add speed classification
                  let speedText = '';
                  if (value < 15) speedText = ' (Very slow)';
                  else if (value < 25) speedText = ' (Slow)';
                  else if (value < 40) speedText = ' (Moderate)';
                  else if (value < 55) speedText = ' (Fast)';
                  else speedText = ' (Very fast)';
                  
                  label += value + ' km/h' + speedText;
                  return label;
                }
              }
            }
          },
          scales: {
            x: {
              grid: {
                display: false,
                drawBorder: false
              }
            },
            y: {
              beginAtZero: true,
              suggestedMin: 0,
              suggestedMax: 60,
              title: {
                display: true,
                text: 'Average Speed (km/h)'
              },
              grid: {
                color: 'rgba(0, 0, 0, 0.05)'
              }
            }
          }
        }
      };
      
      // Create the chart
      const ctx = chartRef.current.getContext('2d');
      chartInstanceRef.current = new window.Chart(ctx, config);
      
      // Add event listeners for animation/interactivity if needed
      
    } catch (err) {
      console.error('Error creating average speed chart:', err);
      setError('Failed to render average speed chart.');
    }
  }, [labels, data, loading, error]);
  
  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
      }
    };
  }, []);
  
  // If loading, show spinner
  if (loading) {
    return (
      <div style={{ height: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <Spin size="large" />
      </div>
    );
  }
  
  // If error, show error message
  if (error) {
    return (
      <div style={{ height: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <div>{error}</div>
      </div>
    );
  }
  
  // If no data, show empty state
  if (!labels.length || !data.length) {
    return (
      <div style={{ height: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <Empty description="No speed data available" />
      </div>
    );
  }
  
  return (
    <div style={{ height: '100%', width: '100%', position: 'relative' }}>
      <canvas ref={chartRef} />
    </div>
  );
};

export default AverageSpeedChart; 