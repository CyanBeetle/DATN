"use client";

import { useAuth } from '@/context/authContext';
import axiosInstance from '@/utils/axiosInstance';
import {
  QuestionCircleOutlined, // Ensure icons are imported
  RobotOutlined, // Ensure icons are imported
  SendOutlined, // Ensure icons are imported
  UserOutlined // Ensure icons are imported
} from '@ant-design/icons';
import {
  App, // Added App
  Avatar,
  Button,
  Card,
  Divider,
  Input,
  List,
  Space,
  Spin,
  Typography,
} from 'antd';
import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown'; // Added import

const { Title, Text, Paragraph } = Typography;
const { Search } = Input;

import { SUGGESTED_QUESTIONS } from './data';

const ChatbotPage = () => {
  const { user } = useAuth();
  const { message: antdAppMessage } = App.useApp(); // Use App.useApp() for message context
  const [messages, setMessages] = useState([
    {
      content: "Hello! I'm your AI traffic assistant, powered by Deepseek. How can I help you today?",
      sender: "bot",
      timestamp: new Date()
    },
    {
      content: "You can ask me about traffic conditions, cameras, routes, account settings, and more.",
      sender: "bot",
      timestamp: new Date(Date.now() + 100)
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async (content) => {
    if (!content.trim()) return;

    const userMessage = {
      content,
      sender: 'user',
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);

    const historyForAPI = messages.slice(-5).map(msg => ({
      role: msg.sender === 'user' ? 'user' : 'assistant',
      content: msg.content
    }));

    try {
      const response = await axiosInstance.post(
        '/api/chatbot/deepseek',
        {
          message: content,
          history: historyForAPI
        },
        { baseURL: '/' } // Override baseURL for this Next.js API route call
      );

      if (response.data && response.data.reply) {
        const botMessage = {
          content: response.data.reply,
          sender: 'bot',
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, botMessage]);
      } else {
        antdAppMessage.error('Received an empty or invalid response from the chatbot.'); // Use antdAppMessage
        const botMessage = {
          content: "Sorry, I couldn't get a response. Please try again.",
          sender: "bot",
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, botMessage]);
      }
    } catch (error) {
      console.error('Error sending message to chatbot API:', error);
      let errorMessage = 'Sorry, something went wrong. Please try again.';
      if (error.response && error.response.data && error.response.data.error) {
        errorMessage = error.response.data.error;
      }
      antdAppMessage.error(errorMessage); // Use antdAppMessage
      const botMessage = {
        content: "Sorry, I encountered an error. Please try again later.",
        sender: "bot",
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, botMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleSuggestedQuestion = (question) => {
    handleSendMessage(question);
  };

  const getTimeString = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div style={{ maxWidth: '100%', margin: 20 }}>
      <Card styles={{ body: { padding: 12, display: 'flex', height: 624 } }}> {/* Changed bodyStyle to styles.body */}
        {/* Main Chat Area */}
        <div style={{ width: '75%', flexDirection: 'column', display: 'flex', justifyContent: 'flex-end' }}>
          {/* Messages Container */}
          <div
            style={{
              flexGrow: 1,
              overflowY: 'scroll',
              backgroundColor: '#fafafa',
              borderRadius: '4px',
              marginBottom: '16px',
              padding: '8px'
            }}
          >
            <List
              style={{ width: '100%' }}
              itemLayout="horizontal"
              dataSource={messages}
              renderItem={(message) => (
                <List.Item
                  style={{
                    flexDirection: message.sender === 'user' ? 'row-reverse' : 'row',
                    border: 'none', display: 'flex', justifyContent: 'flex-start',
                    alignItems: 'flex-start'
                  }}
                >
                  <Avatar
                    icon={message.sender === 'user' ? <UserOutlined /> : <RobotOutlined />}
                    style={{
                      backgroundColor: message.sender === 'user' ? '#1890ff' : '#52c41a',
                      flexShrink: 0
                    }}
                  />
                  <div
                    style={{
                      backgroundColor: message.sender === 'user' ? '#e6f7ff' : '#f6ffed',
                      borderRadius: '6px',
                      padding: '8px 12px',
                      maxWidth: '80%',
                      marginRight: message.sender === 'user' ? '8px' : 0,
                      marginLeft: message.sender === 'user' ? 0 : '8px',
                      wordBreak: 'break-word'
                    }}
                  >
                    <div style={{ fontSize: '14px' }}>
                      {message.sender === 'bot' ? (
                        <ReactMarkdown
                          components={{
                            p: ({ node, ...props }) => <Paragraph style={{ marginBottom: 0 }} {...props} />,
                            strong: ({ node, ...props }) => <Text strong {...props} />,
                            em: ({ node, ...props }) => <Text italic {...props} />,
                          }}
                        >
                          {message.content}
                        </ReactMarkdown>
                      ) : (
                        message.content
                      )}
                    </div>
                    <div style={{ fontSize: '11px', color: '#999', marginTop: '4px', textAlign: message.sender === 'user' ? 'left' : 'right' }}>
                      {getTimeString(message.timestamp)}
                    </div>
                  </div>
                </List.Item>
              )}
            />

            {isTyping && (
              <div style={{ display: 'flex', alignItems: 'center', padding: '4px 0', marginLeft: '8px' }}>
                <Avatar icon={<RobotOutlined />} style={{ backgroundColor: '#52c41a', marginRight: '8px' }} />
                <div
                  style={{
                    backgroundColor: '#f6ffed',
                    padding: '8px 12px',
                    borderRadius: '6px',
                  }}
                >
                  <Spin size="small" /> <Text style={{ marginLeft: '8px' }}>Typing...</Text>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          {/* Input Area */}
          <div>
            <Search
              placeholder="Type your message here..."
              enterButton={<SendOutlined />}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onSearch={handleSendMessage}
              disabled={isTyping}
              loading={isTyping}
            />
          </div>
        </div>
        {/* Suggested Questions Sidebar */}
        <div style={{ width: '25%', borderLeft: '1px solid #f0f0f0', padding: '16px', backgroundColor: '#fafafa', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <Text strong>Suggested Questions</Text>
            <Divider style={{ margin: '8px 0' }} />
            <Space direction="vertical" style={{ width: '100%' }}>
              {SUGGESTED_QUESTIONS.map((question, index) => (
                <Button
                  key={index}
                  type="text"
                  icon={<QuestionCircleOutlined />}
                  size="small"
                  block
                  style={{ display: 'flex', justifyContent: 'left', padding: '4px 8px', textAlign: 'left', height: 'auto', whiteSpace: 'normal' }}
                  onClick={() => handleSuggestedQuestion(question)}
                  disabled={isTyping}
                >
                  {question}
                </Button>
              ))}
            </Space>
          </div>

          <Text type="secondary" style={{ fontSize: '12px' }}>
            <Divider style={{ margin: '16px 0 8px 0' }} />
            Ask about traffic conditions, cameras, routes, account settings, and more. Powered by Deepseek.
          </Text>
        </div>
      </Card >
    </div >
  );
};

export default ChatbotPage;