# Architecture Overview

## 1. Overview

This repository contains a data analysis platform built with Streamlit. The platform allows users to import, process, visualize, and analyze data through a web interface. It incorporates advanced data processing techniques, prediction models, and risk assessment capabilities.

The application follows a modular architecture with clear separation of concerns between data handling, processing, visualization, and storage components. The system is designed to be both interactive and capable of handling sophisticated data analysis tasks.

## 2. System Architecture

The application follows a layered architecture:

- **Presentation Layer**: Streamlit web interface (app.py)
- **Business Logic Layer**: Specialized modules for data processing, prediction, and risk assessment
- **Data Access Layer**: Utilities for data handling and database operations
- **Data Storage Layer**: PostgreSQL database for persistent storage

```
┌─────────────────────────────────────┐
│            Presentation             │
│         (Streamlit Frontend)        │
└───────────────────┬─────────────────┘
                    │
┌───────────────────▼─────────────────┐
│           Business Logic            │
│  ┌────────────┐  ┌────────────────┐ │
│  │    Data    │  │   Advanced     │ │
│  │ Processing │  │Data Processing │ │
│  └────────────┘  └────────────────┘ │
│  ┌────────────┐  ┌────────────────┐ │
│  │ Prediction │  │Risk Assessment │ │
│  └────────────┘  └────────────────┘ │
└───────────────────┬─────────────────┘
                    │
┌───────────────────▼─────────────────┐
│           Data Access Layer         │
│  ┌────────────┐  ┌────────────────┐ │
│  │    Data    │  │  Visualization │ │
│  │  Handlers  │  │     Tools      │ │
│  └────────────┘  └────────────────┘ │
│  ┌────────────┐                     │
│  │  Database  │                     │
│  │  Handlers  │                     │
│  └────────────┘                     │
└───────────────────┬─────────────────┘
                    │
┌───────────────────▼─────────────────┐
│           Data Storage              │
│           (PostgreSQL)              │
└─────────────────────────────────────┘
```

## 3. Key Components

### 3.1 Frontend (Presentation Layer)

**Technology**: Streamlit
**Main File**: `app.py`

The frontend is built using Streamlit, which provides an interactive web interface for data analysis. The main app.py file orchestrates the user interface and connects to the various backend modules.

Key features of the frontend:
- Session state management for preserving data across interactions
- Wide layout configuration for better data visualization
- Integration with all backend modules

### 3.2 Business Logic Modules

The business logic is organized into specialized modules:

#### 3.2.1 Data Processing (`modules/data_processing.py`)

Handles basic data processing operations:
- Handling missing values (via multiple imputation methods)
- Removing duplicates
- Normalizing data (Min-Max or Standard Scaling)
- Feature selection

#### 3.2.2 Advanced Data Processing (`modules/advanced_data_processing.py`)

Implements sophisticated data analysis techniques:
- MCMC-based interpolation using PyMC
- Conditional Generative Adversarial Network (CGAN) for synthetic data
- Outlier detection using Isolation Forest
- Statistical testing (K-S distribution, Spearman correlation)
- Permutation testing

#### 3.2.3 Prediction (`modules/prediction.py`)

Manages predictive modeling:
- Model training (Linear Regression, Decision Tree, Random Forest, Gradient Boosting)
- Performance evaluation
- Forecasting
- Model persistence

#### 3.2.4 Risk Assessment (`modules/risk_assessment.py`)

Evaluates prediction reliability:
- Calculation of prediction intervals
- Confidence level adjustments
- Error analysis

### 3.3 Data Access Layer

#### 3.3.1 Data Handler (`utils/data_handler.py`)

Manages data import and export:
- Reading from different file formats (CSV, Excel)
- Exporting data in various formats

#### 3.3.2 Visualization (`utils/visualization.py`)

Generates data visualizations:
- Histograms with KDE
- Other visualization types (suggested by implementation but not fully visible in snippets)
- Based on Matplotlib and Seaborn

#### 3.3.3 Database Handler (`utils/database.py`)

Provides database interaction:
- Connection management through SQLAlchemy
- ORM models for data persistence
- Session management

### 3.4 Data Storage

**Technology**: PostgreSQL
**ORM**: SQLAlchemy

The application uses PostgreSQL for persistent storage, with tables defined in the database.py file:
- `datasets`: Stores uploaded and processed datasets
- `AnalysisResult`: Stores analysis results (only partially visible in the code)

## 4. Data Flow

The typical data flow through the system follows this pattern:

1. **Data Import**: User uploads data files through the Streamlit interface
2. **Initial Processing**: Basic data cleaning and normalization through DataProcessor
3. **Advanced Processing**: Optional advanced analysis through AdvancedDataProcessor
4. **Model Training**: Users can train predictive models via the Predictor module
5. **Risk Assessment**: Predictions can be evaluated for reliability through the RiskAssessor
6. **Visualization**: Results are visualized through the Visualizer
7. **Data Storage**: Processed data and results can be persisted to the database

## 5. External Dependencies

The application relies on several key external libraries:

### 5.1 Core Framework
- **Streamlit**: Web application framework
- **SQLAlchemy**: ORM for database operations

### 5.2 Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **SciPy**: Scientific computing

### 5.3 Machine Learning
- **scikit-learn**: Traditional machine learning algorithms
- **PyMC**: Probabilistic programming and MCMC
- **TensorFlow/PyTorch**: Deep learning frameworks for advanced models

### 5.4 Visualization
- **Matplotlib**: Basic plotting
- **Seaborn**: Statistical data visualization

## 6. Deployment Strategy

The application is configured for deployment in a containerized environment:

### 6.1 Local Development
The `.replit` configuration specifies dependencies and runtime settings for local development:
- Python 3.11
- PostgreSQL 16
- Various system packages (cairo, ffmpeg, etc.)

### 6.2 Production Deployment
The application is designed for autoscaling deployment:
- Streamlit server running on port 5000
- External port mapped to port 80
- Headless server configuration

### 6.3 Database Configuration
Database connection is established via environment variables:
- `DATABASE_URL` environment variable expected for PostgreSQL connection
- Connection pooling configured for optimal performance
- Error handling for connection failures

## 7. Known Issues and Considerations

- The traceback in `attached_assets` suggests a database initialization issue, likely due to missing environment variables or configuration
- The application dependencies are substantial and require careful management in deployment
- The PostgreSQL integration appears to be a critical component requiring proper setup

## 8. Future Architecture Considerations

Potential areas for architectural enhancement:

1. **Authentication System**: Currently not visible in the codebase
2. **API Layer**: For programmatic access to the platform's capabilities
3. **Caching Strategy**: For improved performance with large datasets
4. **Job Queue**: For handling long-running tasks asynchronously
5. **Containerization**: Docker configuration for simpler deployment