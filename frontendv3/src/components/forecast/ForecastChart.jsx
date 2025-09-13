"use client";

import { useEffect, useRef, useState } from 'react';
import { Card, Spin, Segmented, Radio, Empty } from 'antd';
import { LineChartOutlined, BarChartOutlined } from '@ant-design/icons';

const ForecastChart = ({ times = [], congestion = [], speed = [] }) => {
  const chartRef = useRef(null);
  const chartInstanceRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [chartType, setChartType] = useState('line');
  const [dataType, setDataType] = useState('congestion');
  
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
  
  // Create or update chart when data or chart type changes
  useEffect(() => {
    if (loading || error || !times.length || !window.Chart) return;
    
    try {
      // Destroy any existing chart
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
      }
      
      // Get data based on selected type
      const values = dataType === 'congestion' ? congestion : speed;
      
      // Set up the chart data
      const data = {
        labels: times,
        datasets: [
          {
            label: dataType === 'congestion' ? 'Congestion Level' : 'Average Speed (km/h)',
            data: values,
            fill: dataType === 'congestion',
            backgroundColor: dataType === 'congestion' 
              ? 'rgba(255, 99, 132, 0.2)' 
              : 'rgba(54, 162, 235, 0.2)',
            borderColor: dataType === 'congestion' 
              ? 'rgb(255, 99, 132)' 
              : 'rgb(54, 162, 235)',
            tension: 0.3,
            borderWidth: 2,
          }
        ]
      };
      
      // Configure chart options
      const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                let label = context.dataset.label || '';
                if (label) {
                  label += ': ';
                }
                
                let value = context.parsed.y;
                if (dataType === 'congestion') {
                  label += value.toFixed(1);
                  label += ` (${getCongestionText(value)})`;
                } else {
                  label += value.toFixed(1) + ' km/h';
                }
                
                return label;
              }
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            max: dataType === 'congestion' ? 5 : undefined,
            title: {
              display: true,
              text: dataType === 'congestion' ? 'Congestion Level' : 'Speed (km/h)'
            },
            grid: {
              color: 'rgba(0, 0, 0, 0.05)'
            }
          },
          x: {
            title: {
              display: true,
              text: 'Time'
            },
            grid: {
              color: 'rgba(0, 0, 0, 0.05)'
            }
          }
        }
      };
      
      // Create the chart
      const ctx = chartRef.current.getContext('2d');
      chartInstanceRef.current = new window.Chart(ctx, {
        type: chartType,
        data: data,
        options: options
      });
      
    } catch (err) {
      console.error('Error creating chart:', err);
      setError('Failed to render traffic forecast chart.');
    }
    
  }, [times, congestion, speed, loading, error, chartType, dataType]);
  
  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
      }
    };
  }, []);
  
  // Get congestion text based on level
  const getCongestionText = (level) => {
    const roundedLevel = Math.round(level);
    switch (roundedLevel) {
      case 1: return 'Free flowing';
      case 2: return 'Light';
      case 3: return 'Moderate';
      case 4: return 'Heavy';
      case 5: return 'Severe';
      default: return 'Unknown';
    }
  };
  
  // If data is missing, show empty state
  if (!loading && (!times.length || !congestion.length)) {
    return (
      <div style={{ height: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <Empty description="No forecast data available" />
      </div>
    );
  }
  
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
  
  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ marginBottom: '16px', display: 'flex', justifyContent: 'space-between' }}>
        <Segmented
          options={[
            { 
              label: 'Congestion', 
              value: 'congestion',
            },
            { 
              label: 'Speed', 
              value: 'speed',
            }
          ]}
          value={dataType}
          onChange={setDataType}
        />
        
        <Segmented
          options={[
            { 
              label: 'Line', 
              value: 'line',
              icon: <LineChartOutlined />
            },
            { 
              label: 'Bar', 
              value: 'bar',
              icon: <BarChartOutlined />
            }
          ]}
          value={chartType}
          onChange={setChartType}
        />
      </div>
      
      <div style={{ flex: 1, position: 'relative' }}>
        <canvas ref={chartRef} />
      </div>
    </div>
  );
};

export default ForecastChart; 