# Capstone Project

This project is a full-stack application featuring a frontend, a backend API, and machine learning model building capabilities.

## Project Structure

The project is organized into the following main directories:

-   `backendv3/`: Contains the FastAPI backend application.
    -   `run.py`: Main script to run the backend server.
    -   `api/`: Defines API endpoints.
    -   `app/`: Core application logic, configuration, and main FastAPI app.
    -   `db/`: Database models (`models.py`) and session management.
    -   `auth/`: Authentication and authorization logic.
    -   `cameras/`: Camera-related functionalities.
    -   `reports/`: User report handling.
    -   `ml/`: Machine learning integration points within the backend.
    -   `requirements.txt`: Python dependencies for the backend.
-   `frontendv3/`: Contains the Next.js frontend application.
    -   `src/app/`: Main application pages and components.
    -   `public/`: Static assets.
    -   `package.json`: Node.js dependencies and scripts.
-   `modelbuilding/`: Contains scripts and resources for training and managing machine learning models.
    -   `app.py`: Potentially a Flask/Dash app for model interaction or visualization.
    -   `run_pipeline.py`: Script to run the model training pipeline.
    -   `requirements.txt`: Python dependencies for model building.

## Getting Started

### Prerequisites

-   Python (version specified in backend `requirements.txt` or system specs)
-   Node.js and npm/pnpm (version specified in frontend `package.json` or system specs)
-   MongoDB (or other database as per `backendv3/db/session.py` configuration)

### Backend Setup

1.  Navigate to the `backendv3` directory:
    ```bash
    cd backendv3
    ```
2.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Configure environment variables (e.g., database connection string, JWT secrets). This might involve creating a `.env` file based on `app/config.py`.
4.  Run the backend server:
    ```bash
    python run.py
    ```
    The API documentation will be available at `http://localhost:8000/docs`.

### Frontend Setup

1.  Navigate to the `frontendv3` directory:
    ```bash
    cd frontendv3
    ```
2.  Install Node.js dependencies:
    ```bash
    npm install # or pnpm install
    ```
3.  Run the frontend development server:
    ```bash
    npm run dev # or pnpm dev
    ```
    The frontend will be accessible at `http://localhost:3000`.

### Model Building

1.  Navigate to the `modelbuilding` directory:
    ```bash
    cd modelbuilding
    ```
2.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Follow instructions in `modelbuilding/README.md` or relevant scripts (`run_pipeline.py`) to train or use models.

## Key Features

*(Please add a brief description of the key features and functionalities of your application here.)*

## API Endpoints

A list of primary API endpoints can be found in `API_Endpoints_List.txt` or by exploring the auto-generated documentation at `/docs` when the backend is running.

## Database Design

Details about the database schema and relationships are described in `Database Design.txt`.

## System Specifications

Further system specifications and requirements are outlined in `System Specification.txt`.

## Use Cases

The main use cases for this application are detailed in `Use Case Description.txt`.

---

