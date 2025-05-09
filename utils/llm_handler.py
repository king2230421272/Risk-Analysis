"""
LLM Handler Module

This module handles interactions with large language models (LLMs) for natural language
processing tasks within the Data Analysis Platform.
"""

import os
import json
import openai
from anthropic import Anthropic
import traceback
from deepseek_ai import DeepSeekAI
from deepseek_ai.chat import DeepSeekChat

class LlmHandler:
    """
    Handler for large language model interactions.
    Supports OpenAI, Anthropic, and Deepseek models for natural language processing tasks.
    """
    
    def __init__(self):
        """Initialize the LLM handler with available API keys."""
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        self.anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')
        self.deepseek_api_key = os.environ.get('DEEPSEEK_API_KEY')
        
        # Check available services
        self.openai_available = self.openai_api_key is not None and len(self.openai_api_key) > 0
        self.anthropic_available = self.anthropic_api_key is not None and len(self.anthropic_api_key) > 0
        self.deepseek_available = self.deepseek_api_key is not None and len(self.deepseek_api_key) > 0
        
        # Initialize clients if keys are available
        self.openai_client = None
        self.anthropic_client = None
        self.deepseek_client = None
        self.deepseek_chat = None
        
        if self.openai_available:
            try:
                self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
                print("OpenAI client successfully initialized")
            except Exception as e:
                print(f"Error initializing OpenAI client: {e}")
                self.openai_available = False
        
        if self.anthropic_available:
            try:
                self.anthropic_client = Anthropic(api_key=self.anthropic_api_key)
                print("Anthropic client successfully initialized")
            except Exception as e:
                print(f"Error initializing Anthropic client: {e}")
                self.anthropic_available = False
                
        if self.deepseek_available:
            try:
                import os
                os.environ["DEEPSEEK_API_KEY"] = self.deepseek_api_key
                self.deepseek_client = DeepSeekAI()
                self.deepseek_chat = DeepSeekChat()
                print("Deepseek client successfully initialized")
            except Exception as e:
                print(f"Error initializing Deepseek client: {e}")
                self.deepseek_available = False
    
    def is_any_service_available(self):
        """Check if any LLM service is available."""
        return self.openai_available or self.anthropic_available or self.deepseek_available
    
    def get_available_services(self):
        """Get a list of available LLM services."""
        services = []
        if self.openai_available:
            services.append("OpenAI (GPT-4o)")
        if self.anthropic_available:
            services.append("Anthropic (Claude-3.5-Sonnet)")
        if self.deepseek_available:
            services.append("Deepseek (Deepseek-Chat)")
        return services
    
    def parse_condition_text_with_openai(self, natural_language_text, column_info, example_data=None):
        """
        Parse natural language text into structured condition data using OpenAI's GPT model.
        
        Parameters:
        -----------
        natural_language_text : str
            The natural language description of condition values
        column_info : dict
            Dictionary with column statistics information (name, min, max, mean, etc.)
        example_data : pandas.DataFrame, optional
            Example data to provide context
            
        Returns:
        --------
        dict
            Structured condition data as a dictionary
        """
        if not self.openai_available:
            return {"error": "OpenAI API key is not available or invalid. Please add your API key in the settings."}
        
        try:
            # Prepare system message with context about the columns
            system_message = f"""
            You are a data analysis assistant helping convert natural language descriptions into 
            structured data for a CGAN (Conditional Generative Adversarial Network) model.
            
            Given the following columns with their statistics:
            {json.dumps(column_info, indent=2)}
            
            Convert the user's natural language description into a JSON object where:
            1. Keys are the column names listed above
            2. Values are numeric values derived from the description that are within the min/max range for each column
            
            Return ONLY a valid JSON object with no explanation or markdown formatting.
            """
            
            # Add example data context if provided
            if example_data is not None:
                example_desc = f"""
                Here are a few example rows from the dataset to help you understand the data:
                {example_data.head(3).to_string()}
                """
                system_message += example_desc
            
            # Send request to OpenAI
            # Note that the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": natural_language_text}
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            content = response.choices[0].message.content
            if not content:
                return {"error": "Empty content received from OpenAI API"}
                
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Raw content from OpenAI: {content}")
                # Fall back to regex-based extraction
                extracted = self._extract_json_from_text(content)
                if extracted:
                    return extracted
                return {"error": f"Failed to parse JSON response from OpenAI: {str(e)}"}
            
        except Exception as e:
            return {
                "error": f"Error processing with OpenAI: {str(e)}",
                "traceback": traceback.format_exc()
            }
    
    def parse_condition_text_with_anthropic(self, natural_language_text, column_info, example_data=None):
        """
        Parse natural language text into structured condition data using Anthropic's Claude model.
        
        Parameters:
        -----------
        natural_language_text : str
            The natural language description of condition values
        column_info : dict
            Dictionary with column statistics information (name, min, max, mean, etc.)
        example_data : pandas.DataFrame, optional
            Example data to provide context
            
        Returns:
        --------
        dict
            Structured condition data as a dictionary
        """
        if not self.anthropic_available:
            return {"error": "Anthropic API key is not available or invalid. Please add your API key in the settings."}
        
        try:
            # Prepare prompt with context about the columns
            prompt = f"""
            Human: You are a data analysis assistant helping convert natural language descriptions into 
            structured data for a CGAN (Conditional Generative Adversarial Network) model.
            
            Given the following columns with their statistics:
            {json.dumps(column_info, indent=2)}
            
            Convert my natural language description into a JSON object where:
            1. Keys are the column names listed above
            2. Values are numeric values derived from the description that are within the min/max range for each column
            
            Return ONLY a valid JSON object with no explanation or markdown formatting.
            
            Here's my description: {natural_language_text}
            
            """
            
            # Add example data context if provided
            if example_data is not None:
                example_desc = f"""
                Here are a few example rows from the dataset to help you understand the data:
                {example_data.head(3).to_string()}
                """
                prompt += example_desc
            
            prompt += "\n\nAssistant:"
            
            # Send request to Anthropic
            # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract and parse the JSON response
            if hasattr(response, 'content') and response.content and len(response.content) > 0:
                content = response.content[0].text
                if not content:
                    return {"error": "Empty content received from Anthropic API"}
                    
                # Strip any potential markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                try:
                    result = json.loads(content)
                    return result
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    print(f"Raw content from Anthropic: {content}")
                    # Fall back to regex-based extraction
                    extracted = self._extract_json_from_text(content)
                    if extracted:
                        return extracted
                    return {"error": f"Failed to parse JSON response from Anthropic: {str(e)}"}
            else:
                return {"error": "Invalid response format from Anthropic API"}
            
        except Exception as e:
            return {
                "error": f"Error processing with Anthropic: {str(e)}",
                "traceback": traceback.format_exc()
            }
    
    def parse_condition_text_with_deepseek(self, natural_language_text, column_info, example_data=None):
        """
        Parse natural language text into structured condition data using Deepseek Chat model.
        
        Parameters:
        -----------
        natural_language_text : str
            The natural language description of condition values
        column_info : dict
            Dictionary with column statistics information (name, min, max, mean, etc.)
        example_data : pandas.DataFrame, optional
            Example data to provide context
            
        Returns:
        --------
        dict
            Structured condition data as a dictionary
        """
        if not self.deepseek_available:
            return {"error": "Deepseek API key is not available or invalid. Please add your API key in the settings."}
        
        try:
            # Prepare system message with context about the columns
            system_message = f"""You are a data analysis assistant helping convert natural language descriptions into 
            structured data for a CGAN (Conditional Generative Adversarial Network) model.
            
            Given the following columns with their statistics:
            {json.dumps(column_info, indent=2)}
            
            Convert the user's natural language description into a JSON object where:
            1. Keys are the column names listed above
            2. Values are numeric values derived from the description that are within the min/max range for each column
            
            Return ONLY a valid JSON object with no explanation or markdown formatting."""
            
            # Add example data context if provided
            if example_data is not None:
                example_desc = f"""
                Here are a few example rows from the dataset to help you understand the data:
                {example_data.head(3).to_string()}
                """
                system_message += example_desc
            
            # Send request to Deepseek
            response = self.deepseek_chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": natural_language_text}
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            if hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if not content:
                    return {"error": "Empty content received from Deepseek API"}
                    
                try:
                    result = json.loads(content)
                    return result
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    print(f"Raw content from Deepseek: {content}")
                    # Fall back to regex-based extraction
                    extracted = self._extract_json_from_text(content)
                    if extracted:
                        return extracted
                    return {"error": f"Failed to parse JSON response from Deepseek: {str(e)}"}
            else:
                return {"error": "Invalid response format from Deepseek API"}
            
        except Exception as e:
            return {
                "error": f"Error processing with Deepseek: {str(e)}",
                "traceback": traceback.format_exc()
            }
    
    def parse_condition_text(self, natural_language_text, column_info, service="auto", example_data=None):
        """
        Parse natural language text into structured condition data using the specified service.
        
        Parameters:
        -----------
        natural_language_text : str
            The natural language description of condition values
        column_info : dict
            Dictionary with column statistics information
        service : str
            Which service to use ('auto', 'openai', 'anthropic', or 'deepseek')
        example_data : pandas.DataFrame, optional
            Example data to provide context
            
        Returns:
        --------
        dict
            Structured condition data as a dictionary
        """
        # Automatically select service if set to auto
        if service == "auto":
            if self.anthropic_available:
                service = "anthropic"
            elif self.openai_available:
                service = "openai"
            elif self.deepseek_available:
                service = "deepseek"
            else:
                return {"error": "No LLM service available. Please add either OpenAI, Anthropic, or Deepseek API key in the settings."}
        
        # Use the specified service
        if service == "openai" and self.openai_available:
            return self.parse_condition_text_with_openai(natural_language_text, column_info, example_data)
        elif service == "anthropic" and self.anthropic_available:
            return self.parse_condition_text_with_anthropic(natural_language_text, column_info, example_data)
        elif service == "deepseek" and self.deepseek_available:
            return self.parse_condition_text_with_deepseek(natural_language_text, column_info, example_data)
        else:
            return {"error": f"Selected service '{service}' is not available. Please check your API key settings."}
    
    def parse_condition_text_with_code(self, natural_language_text, column_info):
        """
        Parse natural language text using code-based rules without LLM.
        This is a fallback method when no LLM service is available.
        
        Parameters:
        -----------
        natural_language_text : str
            The natural language description of condition values
        column_info : dict
            Dictionary with column statistics information
            
        Returns:
        --------
        dict
            Structured condition data as a dictionary
        """
        import re
        
        result = {}
        
        # Default to mean values
        for col in column_info:
            result[col["Column"]] = col["Mean"]
        
        # Try to extract numeric values associated with column names
        for col in column_info:
            col_name = col["Column"].lower()
            
            # Search for pattern like "column_name is X" or "column_name should be X"
            patterns = [
                rf"{col_name}\s+is\s+([-+]?\d*\.?\d+)",
                rf"{col_name}\s+should\s+be\s+([-+]?\d*\.?\d+)",
                rf"{col_name}\s+=\s*([-+]?\d*\.?\d+)",
                rf"{col_name}[：:]\s*([-+]?\d*\.?\d+)",
            ]
            
            for pattern in patterns:
                matches = re.search(pattern, natural_language_text.lower())
                if matches:
                    try:
                        value = float(matches.group(1))
                        # Ensure value is within min/max range
                        value = max(col["Min"], min(col["Max"], value))
                        result[col["Column"]] = value
                        break
                    except (ValueError, IndexError):
                        pass
            
            # Look for relative terms like "high", "low", "medium"
            if col_name in natural_language_text.lower():
                text_segment = natural_language_text.lower().split(col_name)[1][:20]
                
                if "high" in text_segment or "large" in text_segment or "big" in text_segment:
                    # Use value at 75% between mean and max
                    result[col["Column"]] = col["Mean"] + 0.75 * (col["Max"] - col["Mean"])
                elif "low" in text_segment or "small" in text_segment:
                    # Use value at 75% between min and mean
                    result[col["Column"]] = col["Min"] + 0.25 * (col["Mean"] - col["Min"])
                elif "medium" in text_segment or "average" in text_segment:
                    # Use the mean
                    result[col["Column"]] = col["Mean"]
        
        return result
        
    def process_prediction_conditions(self, natural_language_text, data, feature_columns=None, service="auto"):
        """
        Process natural language text into structured condition parameters for prediction models.
        
        Parameters:
        -----------
        natural_language_text : str
            The natural language description of condition values and scenarios
        data : pandas.DataFrame
            Training data to provide statistical context
        feature_columns : list, optional
            Specific feature columns to focus on (if None, all columns except the target are used)
        service : str
            Which service to use ('auto', 'openai', 'anthropic', or 'deepseek')
            
        Returns:
        --------
        dict
            Structured condition data as a dictionary for model training
        """
        if not natural_language_text or natural_language_text.strip() == "":
            return {}
            
        try:
            # Prepare column information
            column_info = []
            
            # Use specified columns or all columns
            cols_to_process = feature_columns if feature_columns is not None else data.columns
            
            for col in cols_to_process:
                if col in data.columns and data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    col_stats = {
                        "Column": col,
                        "Min": float(data[col].min()),
                        "Max": float(data[col].max()),
                        "Mean": float(data[col].mean()),
                        "Median": float(data[col].median()),
                        "Std": float(data[col].std()),
                        "Type": str(data[col].dtype)
                    }
                    column_info.append(col_stats)
            
            # Automatically select service if set to auto
            if service == "auto":
                if self.anthropic_available:
                    service = "anthropic"
                elif self.openai_available:
                    service = "openai"
                elif self.deepseek_available:
                    service = "deepseek"
            
            # Prepare the system prompt differently for prediction vs other tasks
            if self.openai_available and (service == "openai"):
                system_message = f"""
                You are an AI assistant helping with machine learning prediction models. 
                Your task is to convert a natural language description into structured condition data
                that can be used for training prediction models.
                
                Based on the dataset with these column statistics:
                {json.dumps(column_info, indent=2)}
                
                Convert the user's description into a JSON object where:
                1. Keys are the feature column names that are relevant to the described conditions
                2. Values are appropriate numeric values that represent the condition described
                
                For example, if the user says "predict results when temperature is high and pressure is low", 
                you would return something like {{"temperature": 85, "pressure": 0.2}} (with actual values
                that make sense for the given column ranges).
                
                Only include columns that are explicitly mentioned or strongly implied in the description.
                Return ONLY a valid JSON object with no explanation or markdown formatting.
                """
                
                # Send request to OpenAI
                # Note that the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
                # do not change this unless explicitly requested by the user
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": natural_language_text}
                    ],
                    response_format={"type": "json_object"}
                )
                
                # Parse the response
                result = json.loads(response.choices[0].message.content)
                return result
                
            elif self.anthropic_available and (service == "anthropic"):
                prompt = f"""
                Human: You are an AI assistant helping with machine learning prediction models. 
                Your task is to convert a natural language description into structured condition data
                that can be used for training prediction models.
                
                Based on the dataset with these column statistics:
                {json.dumps(column_info, indent=2)}
                
                Convert my description into a JSON object where:
                1. Keys are the feature column names that are relevant to the described conditions
                2. Values are appropriate numeric values that represent the condition described
                
                For example, if I say "predict results when temperature is high and pressure is low", 
                you would return something like {{"temperature": 85, "pressure": 0.2}} (with actual values
                that make sense for the given column ranges).
                
                Only include columns that are explicitly mentioned or strongly implied in the description.
                Return ONLY a valid JSON object with no explanation or markdown formatting.

                Here's my description: {natural_language_text}
                
                Assistant:
                """
                
                # Send request to Anthropic
                # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                # Extract and parse the JSON response
                content = response.content[0].text
                # Strip any potential markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                result = json.loads(content)
                return result
                
            elif self.deepseek_available and (service == "deepseek"):
                system_message = f"""
                You are an AI assistant helping with machine learning prediction models. 
                Your task is to convert a natural language description into structured condition data
                that can be used for training prediction models.
                
                Based on the dataset with these column statistics:
                {json.dumps(column_info, indent=2)}
                
                Convert the user's description into a JSON object where:
                1. Keys are the feature column names that are relevant to the described conditions
                2. Values are appropriate numeric values that represent the condition described
                
                For example, if the user says "predict results when temperature is high and pressure is low", 
                you would return something like {{"temperature": 85, "pressure": 0.2}} (with actual values
                that make sense for the given column ranges).
                
                Only include columns that are explicitly mentioned or strongly implied in the description.
                Return ONLY a valid JSON object with no explanation or markdown formatting.
                """
                
                # Send request to Deepseek
                response = self.deepseek_chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": natural_language_text}
                    ],
                    response_format={"type": "json_object"}
                )
                
                # Parse the response
                result = json.loads(response.choices[0].message.content)
                return result
            else:
                # Use code-based fallback method
                return self.parse_condition_text_with_code(natural_language_text, column_info)
                
        except Exception as e:
            print(f"Error processing prediction conditions: {str(e)}")
            traceback.print_exc()
            return {"error": str(e)}