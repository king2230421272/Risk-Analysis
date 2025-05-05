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
        Export data to a file and generate a download link.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data to export
        format : str
            Export format ('csv' or 'excel')
            
        Returns:
        --------
        tuple
            (success, download_link)
        """
        try:
            # Create a buffer for the data
            buffer = io.BytesIO()
            
            # Write data to the buffer based on format
            if format == 'csv':
                data.to_csv(buffer, index=False)
                mime_type = 'text/csv'
                file_ext = 'csv'
            elif format == 'excel':
                data.to_excel(buffer, index=False)
                mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                file_ext = 'xlsx'
            else:
                return False, None
            
            # Reset buffer position
            buffer.seek(0)
            
            # Convert buffer to base64
            b64 = base64.b64encode(buffer.read()).decode()
            
            # Generate download link
            href = f'<a href="data:{mime_type};base64,{b64}" download="exported_data.{file_ext}">Download {file_ext.upper()} File</a>'
            
            return True, href
        
        except Exception as e:
            print(f"Error exporting data: {e}")
            return False, None
