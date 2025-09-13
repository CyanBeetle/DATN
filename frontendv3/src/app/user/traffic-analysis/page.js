'use client';

import React, { useState, useEffect, useCallback } from 'react';
import {
  Typography,
  Select,
  Button,
  Spin,
  Alert,
  Card,
  Row,
  Col,
  Statistic,
  Empty,
  Form,
  TimePicker,
  Checkbox,
  Image // For displaying base64 image
} from 'antd';
import { LineChartOutlined, ClockCircleOutlined, InfoCircleOutlined, SendOutlined } from '@ant-design/icons';
import { Line } from '@ant-design/charts';
import axiosInstance from '@/utils/axiosInstance';
import { useAuth } from '@/context/authContext';
import dayjs from 'dayjs'; // For TimePicker default value

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;

// Helper function to determine congestion level and color based on A-F scale
const getCongestionInfo = (speed) => {
  if (speed < 10) return { level: 'F', color: '#8B0000', description: 'Gridlock conditions. Stop and go traffic, if moving at all.' };
  if (speed < 20) return { level: 'E', color: '#FF0000', description: 'Severe congestion. Very slow speeds, frequent stops.' };
  if (speed < 30) return { level: 'D', color: '#FFA500', description: 'Heavy congestion. Significantly reduced speeds.' };
  if (speed < 40) return { level: 'C', color: '#FFFF00', description: 'Moderate congestion. Somewhat reduced speeds.' };
  if (speed < 50) return { level: 'B', color: '#90EE90', description: 'Light congestion. Slightly reduced speeds.' };
  return { level: 'A', color: '#008000', description: 'Free flow conditions. Traffic flows at or above speed limit.' };
};

const TrafficAnalysisPage = () => {
  // Removed horizon state, as model selection implies horizon
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [forecastData, setForecastData] = useState(null);  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState(null);
  const [loadingModels, setLoadingModels] = useState(true);

  const [form] = Form.useForm();
  const { user } = useAuth(); // For auth checks if necessary

  const fetchAvailableModels = useCallback(async () => {
    setLoadingModels(true);
    try {
      const response = await axiosInstance.get('/api/ml/models');
      setAvailableModels(response.data || []);
      if (response.data && response.data.length > 0) {
        setSelectedModelId(response.data[0].id); // Select the first model by default
        form.setFieldsValue({ modelId: response.data[0].id });
      }
    } catch (err) {
      console.error('Error fetching available models:', err);
      setError('Failed to load available prediction models.');
      setAvailableModels([]);
    } finally {
      setLoadingModels(false);
    }
  }, [form]);

  useEffect(() => {
    fetchAvailableModels();
  }, [fetchAvailableModels]);

  const handleFetchForecast = async (values) => {
    if (!values.modelId) {
      setError('Please select a model.');
      return;
    }    setLoading(true);
    setError(null);
    setForecastData(null);

    const payload = {
      model_identifier: values.modelId,
      day_of_week: parseInt(values.dayOfWeek, 10),
      time_of_day: values.timeOfDay.format('HH:mm'),
      weather_harsh: values.weatherHarsh || false,
    };

    try {
      const response = await axiosInstance.post('/api/ml/predict', payload);
      // Add congestion info to forecast data
      if (response.data && response.data.predicted_speeds) {
        const enrichedPredictedSpeeds = response.data.predicted_speeds.map(speed => ({
          speed: parseFloat(speed.toFixed(2)),
          congestion: getCongestionInfo(speed)
        }));
        setForecastData({...response.data, enrichedPredictedSpeeds});      } else {
        setForecastData(response.data);
      }
    } catch (err) {
      console.error('Error fetching forecast:', err);
      setError(
        err.response?.data?.detail ||
        'Failed to fetch traffic forecast. Please try again.'
      );
    } finally {
      setLoading(false);
    }
  };

  const chartConfig = forecastData?.enrichedPredictedSpeeds
    ? {
        data: forecastData.predicted_timestamps.map((ts, index) => ({
          timestamp: ts, // Already formatted HH:MM by backend
          value: forecastData.enrichedPredictedSpeeds[index].speed,
          congestionLevel: forecastData.enrichedPredictedSpeeds[index].congestion.level,
          color: forecastData.enrichedPredictedSpeeds[index].congestion.color,
          type: forecastData.model_info?.name || 'Prediction',
        })),        xField: 'timestamp',
        yField: 'value',
        seriesField: 'congestionLevel', // Group by congestion level for color
        color: ({ congestionLevel }) => { // Custom color mapping based on A-F scale
          switch (congestionLevel) {
            case 'F': return '#8B0000';  // Dark red
            case 'E': return '#FF0000';  // Red
            case 'D': return '#FFA500';  // Orange
            case 'C': return '#FFFF00';  // Yellow
            case 'B': return '#90EE90';  // Light green
            case 'A': return '#008000';  // Green
            default: return '#008000';   // Default to green
          }
        },
        xAxis: {
          title: { text: 'Time' },
          label: { rotate: 0.3 },
        },        yAxis: {
          title: { text: 'Predicted Speed (km/h)' },
          min: 0,
          max: 70,  // Set maximum to 70 km/h to better show fluctuations
          grid: {
            alternateColor: 'rgba(0, 0, 0, 0.03)', 
          },
        },
        smooth: true,
        tooltip: {
          formatter: (datum) => {
            return { 
              name: `${datum.congestionLevel} Congestion (${datum.type})`, 
              value: `${datum.value} km/h at ${datum.timestamp}` 
            };
          },
        },        legend: { 
          position: 'top-right',
          items: [ // Custom legend items for A-F congestion levels
            { id: 'A', name: 'Level A - Free Flow', marker: { symbol: 'circle', style: { fill: '#008000' } } },
            { id: 'B', name: 'Level B - Light Congestion', marker: { symbol: 'circle', style: { fill: '#90EE90' } } },
            { id: 'C', name: 'Level C - Moderate Congestion', marker: { symbol: 'circle', style: { fill: '#FFFF00' } } },
            { id: 'D', name: 'Level D - Heavy Congestion', marker: { symbol: 'circle', style: { fill: '#FFA500' } } },
            { id: 'E', name: 'Level E - Severe Congestion', marker: { symbol: 'circle', style: { fill: '#FF0000' } } },
            { id: 'F', name: 'Level F - Gridlock', marker: { symbol: 'circle', style: { fill: '#8B0000' } } },
          ]
        },        point: { // Style points on the line
          shape: 'circle',
          style: ({ congestionLevel }) => {
            let fill;
            switch (congestionLevel) {
              case 'F': fill = '#8B0000'; break;  // Dark red
              case 'E': fill = '#FF0000'; break;  // Red
              case 'D': fill = '#FFA500'; break;  // Orange
              case 'C': fill = '#FFFF00'; break;  // Yellow
              case 'B': fill = '#90EE90'; break;  // Light green
              case 'A': fill = '#008000'; break;  // Green
              default: fill = '#008000'; break;   // Default to green
            }
            return { fill, stroke: fill, lineWidth: 1 };
          }
        },
        height: 350,
      }
    : null;

  // Calculate overall/current congestion for display
  const getCurrentCongestionSummary = () => {
    if (!forecastData || !forecastData.enrichedPredictedSpeeds || forecastData.enrichedPredictedSpeeds.length === 0) {
      return null;
    }
    // Consider the first prediction point as "current" or average if preferred
    const firstPrediction = forecastData.enrichedPredictedSpeeds[0];
    return (
      <Col xs={24} md={12}>
        <Title level={4}>Current Forecasted Condition</Title>
        <Statistic 
          title="Initial Predicted Speed" 
          value={`${firstPrediction.speed} km/h`} 
        />
        <Statistic 
          title="Congestion Level" 
          value={firstPrediction.congestion.level}
          valueStyle={{ color: firstPrediction.congestion.color }}
        />
        <Paragraph style={{ color: firstPrediction.congestion.color, marginTop: '8px' }}>
          {firstPrediction.congestion.description}
        </Paragraph>
      </Col>
    );
  };

  return (
    <div style={{ padding: '20px' }}>
      <Title level={2}><LineChartOutlined /> Traffic Forecast</Title>
      <Paragraph>
        Select a prediction model and provide input conditions to generate a traffic speed forecast.
      </Paragraph>

      <Form
        form={form}
        layout="vertical"
        onFinish={handleFetchForecast}
        initialValues={{
          // dayjs().day() gives 0 for Sunday, 1 for Monday, ..., 6 for Saturday.
          // The Select options are value="0" for Monday, ..., value="6" for Sunday.
          // Backend expects 0 for Monday, ..., 6 for Sunday.
          dayOfWeek: (dayjs().day() === 0 ? 6 : dayjs().day() - 1).toString(),
          timeOfDay: dayjs(), // Defaults to current time
          weatherHarsh: false,
        }}
      >
        <Row gutter={[16, 16]} align="bottom" style={{ marginBottom: '24px' }}>
          <Col xs={24} sm={12} md={8} lg={6}>
            <Form.Item
              name="modelId"
              label="Select Prediction Model"
              rules={[{ required: true, message: 'Please select a model!' }]}
            >
              <Select
                placeholder="Select a model"
                loading={loadingModels}
                disabled={loading || loadingModels}
                onChange={(value) => setSelectedModelId(value)}
              >
                {availableModels.map(model => (
                  <Option key={model.id} value={model.id}>
                    {model.display_name} ({model.inferred_horizon})
                  </Option>
                ))}
              </Select>
            </Form.Item>
          </Col>
          <Col xs={24} sm={12} md={5} lg={4}>
            <Form.Item
              name="dayOfWeek"
              label="Day of Week"
              rules={[{ required: true, message: 'Please select a day!' }]}
            >
              <Select disabled={loading}>
                <Option value="0">Monday</Option>
                <Option value="1">Tuesday</Option>
                <Option value="2">Wednesday</Option>
                <Option value="3">Thursday</Option>
                <Option value="4">Friday</Option>
                <Option value="5">Saturday</Option>
                <Option value="6">Sunday</Option>
              </Select>
            </Form.Item>
          </Col>
          <Col xs={24} sm={12} md={5} lg={4}>
            <Form.Item
              name="timeOfDay"
              label="Time of Day"
              rules={[{ required: true, message: 'Please select a time!' }]}
            >
              <TimePicker format="HH:mm" style={{ width: '100%' }} disabled={loading} />
            </Form.Item>
          </Col>
          <Col xs={24} sm={12} md={4} lg={3}>
            <Form.Item name="weatherHarsh" valuePropName="checked">
              <Checkbox disabled={loading}>Harsh Weather</Checkbox>
            </Form.Item>
          </Col>
          <Col xs={24} sm={12} md={2} lg={2}>
            <Form.Item>
              <Button
                type="primary"
                htmlType="submit"
                icon={<SendOutlined />}
                loading={loading}
                style={{ width: '100%' }}
              >
                Predict
              </Button>
            </Form.Item>
          </Col>
        </Row>
      </Form>

      {loading && !forecastData && <Spin tip="Generating forecast..." style={{ display: 'block', margin: '20px auto' }} />}
      {error && <Alert message="Error" description={error} type="error" showIcon closable onClose={() => setError(null)} />}

      {forecastData && (
        <Card title="Forecast Results" style={{ marginTop: '20px' }}>
          <Row gutter={[16, 24]}>
            <Col xs={24} md={12}>
              <Title level={4}>Model & Input</Title>
              <Statistic title="Model Used" value={forecastData.model_file_name} />
              <Statistic 
                title="Inferred Horizon" 
                value={availableModels.find(m => m.id === forecastData.user_input.model_identifier)?.inferred_horizon || 'N/A'} 
              /> 
              <Statistic title="Granularity" value={forecastData.granularity_display} />
              <Text strong style={{ marginTop: '10px', display: 'block' }}>Input Parameters:</Text>
              <Text>Day: {forecastData.day_name}, Time: {forecastData.user_input.time_of_day}, Weather: {forecastData.user_input.weather_harsh ? 'Harsh' : 'Normal'}</Text>
            </Col>
            {/* Display current congestion summary */}
            {getCurrentCongestionSummary()}
          </Row>
            <Row style={{marginTop: '24px'}}>
            <Col span={24}>
              <Title level={4}>Detailed Predictions Chart (km/h)</Title>
              {chartConfig ? (
                <Line {...chartConfig} />
              ) : (
                <Empty description="No forecast points to display in chart." />
              )}
            </Col>
          </Row>

        </Card>
      )}
    </div>
  );
};

export default TrafficAnalysisPage;