"use client";

import React from 'react';
import { Area } from '@ant-design/plots';

const AreaChart = ({ data }) => {
  if (!data || data.length === 0) {
    return <div className="flex items-center justify-center h-full">No data available for chart</div>;
  }

  const config = {
    data,
    xField: 'time',
    yField: ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person'],
    seriesField: '',
    color: ['#1890ff', '#f04864', '#9254de', '#faad14', '#52c41a', '#13c2c2'],
    xAxis: {
      range: [0, 1],
      tickCount: 5,
    },
    areaStyle: { fillOpacity: 0.6 },
    legend: {
      position: 'top',
    },
    smooth: true,
    animation: {
      appear: {
        animation: 'path-in',
        duration: 1000,
      },
    },
  };

  return <Area {...config} />;
};

export default AreaChart;