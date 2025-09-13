"use client"

import React, { createContext, useContext, useState, useEffect } from "react";
import { ConfigProvider, theme } from "antd";
import { lightTheme, darkTheme } from "./colors";

// Create Theme Context
const ThemeContext = createContext(null);

export function ThemeProvider({ children }) {
    const [themeMode, setThemeMode] = useState("light");
    const [themeColor, setThemeColor] = useState(lightTheme);

    // Load theme preference from localStorage
    useEffect(() => {
        const savedTheme = localStorage.getItem("theme");
        if (savedTheme === "dark" || savedTheme === "light") {
        setThemeMode(savedTheme);
        setThemeColor(savedTheme === "dark" ? darkTheme : lightTheme);
        }
    }, []);

  // Toggle theme
    const toggleTheme = () => {
        setThemeMode((prev) => {
            const newTheme = prev === "light" ? "dark" : "light";
            setThemeColor(prev === "light" ? darkTheme : lightTheme);
            localStorage.setItem("theme", newTheme);
            return newTheme;
        });
    };

    return (
    <ThemeContext.Provider value={{ themeMode, toggleTheme }}>
        <ConfigProvider
            theme={{
                algorithm: themeMode === "dark" ? theme.darkAlgorithm : theme.defaultAlgorithm,
                token: {
                    // ✅ Define your custom colors
                    borderRadius: 4,
                    colorBgBase: themeColor.surface,
                    colorError: themeColor.error,
                    colorInfo: themeColor.primary,
                    colorLink: themeColor.primary,
                    colorPrimary: themeColor.primary,
                    colorSuccess: themeColor.primary,
                    colorTextBase: themeColor.onSurface,
                    colorWarning: themeColor.tertiary,
                    fontSize: 18, // Background color
                },
                components: {
                    // ✅ Global customization for specific components
                    Button: {
                        colorPrimary: themeColor.primary,
                        colorText: themeColor.onSurface,
                        colorBg: themeColor.surface,                        
                        borderRadius: 8,
                        controlHeight: 40,
                    },
                    Input: {
                        colorBgContainer: themeColor.surface,
                        borderRadius: 8,
                        colorText: themeColor.onSurface,
                    },
                    Card: {
                        colorBgContainer: themeColor.background,
                        borderRadius: 8,
                        colorText: themeColor.onSurface,
                        boxShadow: "0 0 10px rgba(0, 0, 0, 0.1)",
                    },
                    Splitter: {
                        colorBgContainer: themeColor.surface,
                        borderRadius: 8,
                        boxShadow: "0 0 10px rgba(0, 0, 0, 0.1)",
                    },
                    Typography: {
                        colorText: themeColor.onSurface,
                        fontSize: 18,
                        fontSizeHeading1: 32,
                        fontSizeHeading2: 24,
                        fontSizeHeading3: 20,
                    },
                    Modal: {
                        colorBgContainer: themeColor.surface,
                        borderRadius: 8,
                        colorText: themeColor.onSurface,
                    },
                    Menu: {
                        colorBgContainer: themeColor.primaryContainer,
                        borderRadius: 8,
                        colorText: themeColor.onPrimaryContainer,
                    },
                    Layout: {
                        colorBgContainer: themeColor.primaryContainer,
                        borderRadius: 8,
                        colorText: themeColor.onSurface,
                    },
                    Select: {
                        borderRadius: 8,
                        colorText: themeColor.onSurface,
                        optionFontSize: 14,
                        optionHeight: 32,
                    },
                },
            }}
        >
            {children}
        </ConfigProvider>
    </ThemeContext.Provider>
  );
}

// Custom hook to use theme context
export const useTheme = () => useContext(ThemeContext);