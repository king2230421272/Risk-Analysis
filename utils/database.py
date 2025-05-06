import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, MetaData, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json
from io import StringIO  # Import StringIO from io module

# Get PostgreSQL connection details from environment variables
DATABASE_URL = os.environ.get('DATABASE_URL')

# Create the engine with explicit connection pool settings and timeout handling
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,  # Recycle connections after 30 minutes
    connect_args={
        'connect_timeout': 10,  # Connection timeout in seconds
        'application_name': 'DataAnalysisPlatform',  # Identify the application in pg_stat_activity
    }
)

# Create base class for declarative models
Base = declarative_base()

# Define metadata
metadata = MetaData()

# Define models
class Dataset(Base):
    """Model for storing datasets in the database."""
    __tablename__ = 'datasets'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    data_type = Column(String(50), nullable=False)  # 'original' or 'interpolated'
    created_at = Column(DateTime, default=datetime.now)
    modified_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    columns_info = Column(Text, nullable=True)  # JSON string with column information
    row_count = Column(Integer, nullable=False)
    column_count = Column(Integer, nullable=False)
    csv_data = Column(Text, nullable=False)  # CSV string of the entire dataset

# Define models for analysis results
class AnalysisResult(Base):
    """Model for storing analysis results in the database."""
    __tablename__ = 'analysis_results'
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, nullable=False)
    analysis_type = Column(String(50), nullable=False)  # 'prediction', 'risk_assessment', etc.
    analysis_params = Column(Text, nullable=True)  # JSON string with parameters used
    created_at = Column(DateTime, default=datetime.now)
    result_data = Column(Text, nullable=False)  # JSON string of analysis results
    
# Create all tables
def initialize_database():
    """Create all database tables if they don't exist."""
    import time
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Test connection first
            with engine.connect() as connection:
                from sqlalchemy import text
                connection.execute(text("SELECT 1"))  # Simple test query with proper SQLAlchemy text construct
            
            # If connection successful, create tables
            Base.metadata.create_all(engine)
            print("Database connection successful, tables created.")
            return True
        except Exception as e:
            print(f"Database connection attempt {attempt+1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print("Maximum connection attempts reached. Using application without database features.")
                return False
    
# Session factory
Session = sessionmaker(bind=engine)

class DatabaseHandler:
    """Utility for handling database operations for the data analysis platform."""
    
    def __init__(self):
        """Initialize database handler and ensure tables exist."""
        self.db_available = initialize_database()
        
        # If database is not available, set a flag and print a warning
        if not self.db_available:
            print("WARNING: Database features are not available. The application will work but without database functionality.")
            
    def _check_db_available(self):
        """Check if database is available and raise an appropriate exception if not."""
        if not self.db_available:
            raise Exception("Database is not available. Please check your database connection.")
            
    def _db_operation_with_retry(self, operation_func, *args, **kwargs):
        """
        Execute a database operation with retry logic for resilience against connection issues.
        
        Parameters:
        -----------
        operation_func : function
            The database operation function to execute
        *args, **kwargs : 
            Arguments to pass to the operation function
            
        Returns:
        --------
        Any
            The result of the operation function
        """
        max_retries = 3
        retry_delay = 1  # seconds
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return operation_func(*args, **kwargs)
            except Exception as e:
                last_error = e
                # Check if it's a connection error that we should retry
                error_str = str(e).lower()
                if "connection" in error_str or "ssl" in error_str or "timeout" in error_str:
                    if attempt < max_retries - 1:
                        print(f"Database operation failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                        print(f"Retrying in {retry_delay} seconds...")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        # Re-initialize database connection
                        self.db_available = initialize_database()
                    else:
                        break
                else:
                    # Not a connection error, don't retry
                    break
                    
        # If we get here, all retries failed
        raise last_error
        
    def save_dataset(self, dataset_df, name, description=None, data_type='original'):
        """
        Save a pandas DataFrame to the database.
        
        Parameters:
        -----------
        dataset_df : pandas.DataFrame
            The DataFrame to save
        name : str
            Name of the dataset
        description : str, optional
            Description of the dataset
        data_type : str
            Type of dataset ('original' or 'interpolated')
            
        Returns:
        --------
        int
            ID of the saved dataset
        """
        # Check if database is available
        self._check_db_available()
        
        if dataset_df is None or dataset_df.empty:
            raise ValueError("Cannot save empty dataset")
        
        # Define the actual operation function to allow for retry
        def _save_dataset_operation():
            session = Session()
            try:
                # Convert DataFrame to CSV string
                csv_data = dataset_df.to_csv(index=False)
                
                # Create columns info as JSON
                columns_info = {}
                for col in dataset_df.columns:
                    dtype = str(dataset_df[col].dtype)
                    stats = {}
                    if pd.api.types.is_numeric_dtype(dataset_df[col]):
                        stats = {
                            'min': float(dataset_df[col].min()) if not dataset_df[col].isna().all() else None,
                            'max': float(dataset_df[col].max()) if not dataset_df[col].isna().all() else None,
                            'mean': float(dataset_df[col].mean()) if not dataset_df[col].isna().all() else None,
                            'std': float(dataset_df[col].std()) if not dataset_df[col].isna().all() else None,
                            'null_count': int(dataset_df[col].isna().sum()),
                        }
                    else:
                        stats = {
                            'unique_count': int(dataset_df[col].nunique()),
                            'null_count': int(dataset_df[col].isna().sum()),
                        }
                    
                    columns_info[col] = {
                        'dtype': dtype,
                        'stats': stats
                    }
                
                # Create new dataset object
                new_dataset = Dataset(
                    name=name,
                    description=description,
                    data_type=data_type,
                    row_count=len(dataset_df),
                    column_count=len(dataset_df.columns),
                    columns_info=json.dumps(columns_info),
                    csv_data=csv_data
                )
                
                # Add to session and commit
                session.add(new_dataset)
                session.commit()
                
                # Get the ID of the new dataset
                dataset_id = new_dataset.id
                
                return dataset_id
                
            except Exception as e:
                session.rollback()
                raise e
            finally:
                session.close()
                
        # Execute the operation with retry logic
        return self._db_operation_with_retry(_save_dataset_operation)
    
    def load_dataset(self, dataset_id=None, name=None, data_type=None):
        """
        Load a dataset from the database.
        
        Parameters:
        -----------
        dataset_id : int, optional
            ID of the dataset to load
        name : str, optional
            Name of the dataset to load
        data_type : str, optional
            Type of dataset to load ('original' or 'interpolated')
            
        Returns:
        --------
        pandas.DataFrame
            The loaded dataset as a DataFrame
        """
        # Check if database is available
        self._check_db_available()
        
        # Define the actual operation function to allow for retry
        def _load_dataset_operation():
            session = Session()
            try:
                # Query the dataset
                query = session.query(Dataset)
                
                if dataset_id is not None:
                    query = query.filter(Dataset.id == dataset_id)
                
                if name is not None:
                    query = query.filter(Dataset.name == name)
                    
                if data_type is not None:
                    query = query.filter(Dataset.data_type == data_type)
                
                dataset = query.first()
                
                if dataset is None:
                    raise ValueError("Dataset not found")
                
                # Convert CSV string back to DataFrame
                df = pd.read_csv(StringIO(dataset.csv_data))
                
                return df
                
            except Exception as e:
                raise e
            finally:
                session.close()
                
        # Execute the operation with retry logic
        return self._db_operation_with_retry(_load_dataset_operation)
    
    def save_analysis_result(self, dataset_id, analysis_type, analysis_params, result_data):
        """
        Save analysis results to the database.
        
        Parameters:
        -----------
        dataset_id : int
            ID of the dataset used for analysis
        analysis_type : str
            Type of analysis ('prediction', 'risk_assessment', etc.)
        analysis_params : dict
            Parameters used for the analysis
        result_data : dict or pandas.DataFrame
            Results of the analysis
            
        Returns:
        --------
        int
            ID of the saved analysis result
        """
        # Check if database is available
        self._check_db_available()
        
        # Create a session
        session = Session()
        
        try:
            # Convert result data to JSON if it's a DataFrame
            if isinstance(result_data, pd.DataFrame):
                result_data_json = result_data.to_json(orient='split')
            else:
                result_data_json = json.dumps(result_data)
            
            # Convert analysis params to JSON
            if analysis_params is not None:
                analysis_params_json = json.dumps(analysis_params)
            else:
                analysis_params_json = None
            
            # Create new analysis result object
            new_result = AnalysisResult(
                dataset_id=dataset_id,
                analysis_type=analysis_type,
                analysis_params=analysis_params_json,
                result_data=result_data_json
            )
            
            # Add to session and commit
            session.add(new_result)
            session.commit()
            
            # Get the ID of the new analysis result
            result_id = new_result.id
            
            return result_id
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def load_analysis_result(self, result_id):
        """
        Load analysis results from the database.
        
        Parameters:
        -----------
        result_id : int
            ID of the analysis result to load
            
        Returns:
        --------
        tuple
            (dataset_id, analysis_type, analysis_params, result_data)
        """
        # Check if database is available
        self._check_db_available()
        
        # Create a session
        session = Session()
        
        try:
            # Query the analysis result
            result = session.query(AnalysisResult).filter(AnalysisResult.id == result_id).first()
            
            if result is None:
                raise ValueError("Analysis result not found")
            
            # Parse JSON strings
            analysis_params = json.loads(result.analysis_params) if result.analysis_params else None
            
            # Check if result data is a DataFrame
            try:
                result_data = pd.read_json(result.result_data, orient='split')
            except:
                result_data = json.loads(result.result_data)
            
            return (result.dataset_id, result.analysis_type, analysis_params, result_data)
            
        except Exception as e:
            raise e
        finally:
            session.close()
    
    def list_datasets(self, data_type=None):
        """
        List all datasets in the database.
        
        Parameters:
        -----------
        data_type : str, optional
            Filter by dataset type ('original' or 'interpolated')
            
        Returns:
        --------
        list
            List of dictionaries with dataset information
        """
        # Check if database is available
        self._check_db_available()
        
        # Define the actual operation function to allow for retry
        def _list_datasets_operation():
            session = Session()
            try:
                # Query all datasets
                query = session.query(Dataset)
                
                if data_type is not None:
                    query = query.filter(Dataset.data_type == data_type)
                
                datasets = query.all()
                
                # Convert to list of dictionaries
                dataset_list = []
                for ds in datasets:
                    dataset_list.append({
                        'id': ds.id,
                        'name': ds.name,
                        'description': ds.description,
                        'data_type': ds.data_type,
                        'created_at': ds.created_at,
                        'modified_at': ds.modified_at,
                        'row_count': ds.row_count,
                        'column_count': ds.column_count
                    })
                
                return dataset_list
                
            except Exception as e:
                raise e
            finally:
                session.close()
                
        # Execute the operation with retry logic
        return self._db_operation_with_retry(_list_datasets_operation)
            
    def list_analysis_results(self, dataset_id=None, analysis_type=None):
        """
        List analysis results in the database.
        
        Parameters:
        -----------
        dataset_id : int, optional
            Filter by dataset ID
        analysis_type : str, optional
            Filter by analysis type
            
        Returns:
        --------
        list
            List of dictionaries with analysis result information
        """
        # Check if database is available
        self._check_db_available()
        
        # Define the actual operation function to allow for retry
        def _list_analysis_results_operation():
            session = Session()
            try:
                # Query analysis results
                query = session.query(AnalysisResult)
                
                if dataset_id is not None:
                    query = query.filter(AnalysisResult.dataset_id == dataset_id)
                
                if analysis_type is not None:
                    query = query.filter(AnalysisResult.analysis_type == analysis_type)
                
                results = query.all()
                
                # Convert to list of dictionaries
                result_list = []
                for res in results:
                    result_list.append({
                        'id': res.id,
                        'dataset_id': res.dataset_id,
                        'analysis_type': res.analysis_type,
                        'created_at': res.created_at,
                    })
                
                return result_list
                
            except Exception as e:
                raise e
            finally:
                session.close()
                
        # Execute the operation with retry logic
        return self._db_operation_with_retry(_list_analysis_results_operation)
            
    def delete_dataset(self, dataset_id):
        """
        Delete a dataset from the database.
        
        Parameters:
        -----------
        dataset_id : int
            ID of the dataset to delete
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        # Check if database is available
        self._check_db_available()
        
        # Define the actual operation function to allow for retry
        def _delete_dataset_operation():
            session = Session()
            try:
                # Delete associated analysis results first
                session.query(AnalysisResult).filter(AnalysisResult.dataset_id == dataset_id).delete()
                
                # Delete the dataset
                result = session.query(Dataset).filter(Dataset.id == dataset_id).delete()
                
                session.commit()
                
                return result > 0
                
            except Exception as e:
                session.rollback()
                raise e
            finally:
                session.close()
                
        # Execute the operation with retry logic
        return self._db_operation_with_retry(_delete_dataset_operation)