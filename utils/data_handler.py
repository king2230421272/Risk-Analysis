import pandas as pd
import numpy as np
import base64
import io

class DataHandler:
    """
    Utility for handling data import and export operations.
    """
    
    def __init__(self):
        """Initialize data handler with default settings."""
        pass
    
    def import_data(self, file_obj):
        """
        Import data from a file.
        
        Parameters:
        -----------
        file_obj : file object
            File object from file uploader
            
        Returns:
        --------
        pandas.DataFrame
            Imported data
        """
        # Get the file extension
        filename = file_obj.name
        file_ext = filename.split('.')[-1].lower()
        
        # Import data based on file extension
        if file_ext == 'csv':
            return pd.read_csv(file_obj)
        elif file_ext in ['xlsx', 'xls']:
            return pd.read_excel(file_obj)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def export_data(self, data, format='csv'):
        """
        Export data to a file format suitable for download.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data to export
        format : str
            Export format ('csv' or 'excel')
            
        Returns:
        --------
        bytes
            Data in the requested format ready for download
        """
        try:
            # Create a buffer for the data
            buffer = io.BytesIO()
            
            # Write data to the buffer based on format
            if format == 'csv':
                data.to_csv(buffer, index=False)
            elif format == 'excel':
                data.to_excel(buffer, index=False, engine='openpyxl')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Reset buffer position
            buffer.seek(0)
            
            # Return the buffer value
            return buffer.getvalue()
        
        except Exception as e:
            print(f"Error exporting data: {e}")
            raise
