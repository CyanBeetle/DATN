"use client";

import React from 'react';
import { Column } from '@ant-design/plots';

const BarChart = ({ data }) => {
  if (!data || data.length === 0) {
    return <div className="flex items-center justify-center h-full">No data available for chart</div>;
  }

  const config = {
    data,
    xField: 'type',
    yField: 'value',
    label: {
      position: 'middle',
      style: {
        fill: '#FFFFFF',
        opacity: 0.6,
      },
    },
    color: ({ type }) => {
      const colorMap = {
        'car': '#1890ff',
        'truck': '#f04864',
        'bus': '#9254de',
        'motorcycle': '#faad14',
        'bicycle': '#52c41a',
        'person': '#13c2c2'
      };
      return colorMap[type] || '#5B8FF9';
    },
    xAxis: {
      label: {
        autoHide: true,
        autoRotate: false,
      },
    },
    meta: {
      type: {
        alias: 'Object Type',
      },
      value: {
        alias: 'Count',
      },
    },
    animation: {
      appear: {
        animation: 'wave-in',
        duration: 1500,
      },
    },
  };

  return <Column {...config} />;
};

export default BarChart;