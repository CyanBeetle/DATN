"use client";

import {
  Card,
  Col,
  Row,
  Typography
} from 'antd';
import axios from 'axios';
import { useEffect, useState } from 'react';

const { Text, Paragraph } = Typography;

export default function NewsPage() {
  const [weather, setWeather] = useState(null);
  const [news, setNews] = useState(null);
  const [loading, setLoading] = useState(true); // Quản lý trạng thái loading

  useEffect(() => {
    const fetchWeather = async () => {
      try {
        const apiKey = process.env.NEXT_PUBLIC_WEATHER_API_KEY || '1b254b21faa04d06aca175101251005'; // Use env variable
        const res = await axios.get('https://api.weatherapi.com/v1/forecast.json', {
          params: {
            key: apiKey, // Use the apiKey variable
            q: 'Ho Chi Minh',
            days: 1,
            aqi: 'no',
          },
          withCredentials: false
        });

        setWeather(res.data);
      } catch (error) {
        console.error('Error fetching weather data:', error);
      }
    };
    fetchWeather();
    const fetchNews = async () => {
      try {
        const res = await axios.get('/api/news');
        setNews(res.data);
        console.log("News: ", res.data);
      } catch (error) {
        console.error('Error fetching news data:', error);
      }
    };

    fetchNews();
  }, []);

  const current = weather?.current;

  return (
    <div style={{ padding: '16px' }}>
      <Row gutter={8}>
        <Col span={12}>
          <Card title={`Thời tiết ${weather?.location?.name}`}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <img
                  src={weather?.current?.condition?.icon}
                  alt={weather?.current?.condition?.text}
                  width={64}
                  height={64}
                />
                <div>
                  <div style={{ fontSize: 18 }}>{weather?.current?.condition?.text}</div>
                  <div style={{ color: '#888' }}>RealFeel® {weather?.current?.feelslike_c}°</div>
                </div>
              </div>

              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: '36px', fontWeight: 'bold', color: '#1890ff' }}>
                  {weather?.current?.temp_c}°C
                </div>
                <div style={{ fontSize: 12 }}>{weather?.location?.localtime?.split(' ')[1]}</div>
              </div>
            </div>

            <div style={{ marginTop: 16, display: 'flex', justifyContent: 'space-between' }}>
              <div>
                <p>Gió: {weather?.current?.wind_dir} {weather?.current?.wind_kph} km/h</p>
                <p>Chất lượng không khí: Vừa phải</p>
                <p>Điểm sương: {weather?.current?.dewpoint_c}°C</p>
                <p>Mặt độ mây: {weather?.current?.cloud}%</p>
              </div>
              <div style={{ textAlign: 'right' }}>
                <p>Gió giật: {weather?.current?.gust_kph} km/h</p>
                <p>Độ ẩm: {weather?.current?.humidity}%</p>
                <p>Tầm nhìn: {weather?.current?.vis_km} km</p>
                <p>Trần mây: {weather?.current?.cloud * 100} m</p>
              </div>
            </div>
          </Card>
        </Col>

        <Col span={12}>
          <Card title="Tin tức" style={{ height: 624, overflowY: 'scroll' }}>
            {news?.length > 0 ? (
              news.slice(0, 5).map((item, index) => (
                <div key={index} style={{ marginBottom: 16, borderBottom: '1px solid #f0f0f0', paddingBottom: 12 }}>
                  <Row gutter={8}>
                    <Col span={8}>
                      <img
                        src={item.enclosure?.[0]?.$.url || '/fallback.jpg'}
                        alt={item.title?.[0]}
                        style={{ width: '100%', height: 100, objectFit: 'cover', borderRadius: 4 }}
                      />
                    </Col>
                    <Col span={16}>
                      <a href={item.link?.[0]} target="_blank" rel="noopener noreferrer" style={{ fontWeight: 'bold', fontSize: 15 }}>
                        {item.title?.[0]}
                      </a>
                      <Paragraph ellipsis={{ rows: 2, expandable: false }} style={{ margin: '4px 0', fontSize: 13 }}>
                        {item.description?.[0]?.replace(/<[^>]*>/g, '')}
                      </Paragraph>
                    </Col>
                  </Row>
                </div>
              ))
            ) : (
              <Text>Đang tải tin tức...</Text>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
}
