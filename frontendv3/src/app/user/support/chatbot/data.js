// Predefined chatbot responses based on keywords
export const CHATBOT_RESPONSES = {
    GREETING: [
        "Hello! I'm your traffic assistant. How can I help you today?",
        "Hi there! I'm here to answer questions about traffic and our application. What would you like to know?",
        "Welcome to the traffic support chatbot! Ask me about traffic conditions, features, or how to use the app."
    ],
    TRAFFIC: [
        "Traffic information is available on our map view. You can see real-time congestion levels, incidents, and cameras.",
        "You can check current traffic conditions by going to the Map section. The color coding indicates congestion levels.",
        "For traffic forecasts, visit the Traffic Analysis > Predictions section to see future traffic patterns."
    ],
    CAMERA: [
        "To view traffic cameras, navigate to the Camera section. You can add favorites for easy access.",
        "Traffic cameras can be accessed from the map by clicking on camera icons, or directly from the Camera menu.",
        "You can receive notifications from cameras by enabling alerts in the camera details page."
    ],
    ACCOUNT: [
        "You can manage your account in the Settings section. There you can update your display name and password.",
        "Account settings allow you to change your password and update your profile information.",
        "To access account settings, click on your profile icon in the top-right corner and select 'Settings'."
    ],
    REPORT: [
        "You can submit traffic incident reports or infrastructure issues in the Reports section.",
        "To report a traffic incident, go to Reports and fill out the form with details about what you observed.",
        "Your submitted reports can be tracked in the Reports section, where you can also see their status."
    ],
    LOGIN: [
        "To log in, click the 'Login' button in the top right and enter your credentials.",
        "If you're having trouble logging in, make sure you're using the correct username and password.",
        "For security reasons, your account will be temporarily locked after multiple failed login attempts."
    ],
    ROUTE: [
        "Find the best route by going to Traffic Analysis > Routes and entering your start and end locations.",
        "Our route finder takes into account current traffic conditions to suggest the optimal path.",
        "You can save your frequently used routes for quick access in the future."
    ],
    FORECAST: [
        "Traffic forecasts are available in the Traffic Analysis > Predictions section.",
        "Our AI-powered prediction system uses historical data and current patterns to forecast future traffic conditions.",
        "Forecasts can be viewed for different time frames: 30 minutes, 1 hour, or 3 hours ahead."
    ],
    NEWS: [
        "Latest traffic news and updates can be found in the News section.",
        "The News page also includes weather forecasts that might affect traffic conditions.",
        "Traffic-related announcements from authorities are highlighted in the News feed."
    ],
    DEFAULT: [
        "I'm not sure I understand. Could you rephrase or ask a different question?",
        "I don't have information about that. Try asking about traffic conditions, cameras, routes, or app features.",
        "Sorry, I didn't quite catch that. I'm best at answering questions about traffic and how to use our application."
    ]
};

// Initial welcome messages
export const INITIAL_MESSAGES = [
    {
        content: "Hello! I'm your traffic assistant. How can I help you today?",
        sender: "bot",
        timestamp: new Date()
    },
    {
        content: "You can ask me about traffic conditions, cameras, routes, account settings, and more.",
        sender: "bot",
        timestamp: new Date(Date.now() + 100)
    }
];

// Suggested questions
export const SUGGESTED_QUESTIONS = [
    "How do I check traffic?",
    "Where can I view cameras?",
    "How do I find the best route?",
    "How to report a traffic incident?",
    "How do I change my password?",
    "Where are traffic forecasts?"
];