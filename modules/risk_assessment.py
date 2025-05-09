import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import io
import base64
from PIL import Image
import cv2

class RiskAssessor:
    """
    Module for assessing risks in prediction models and results,
    calculating unit loss from land use maps, and determining overall risk levels.
    """
    
    def __init__(self):
        """Initialize risk assessor with default settings."""
        # 土地利用类型损失系数(示例值)
        self.land_use_types = {
            'Urban': {'color': [0, 0, 255], 'loss_factor': 1.0}, # 城市区域，蓝色
            'Residential': {'color': [0, 255, 255], 'loss_factor': 0.8}, # 居民区，青色
            'Industrial': {'color': [255, 0, 0], 'loss_factor': 0.9}, # 工业区，红色
            'Agricultural': {'color': [0, 255, 0], 'loss_factor': 0.5}, # 农业区，绿色
            'Forest': {'color': [0, 128, 0], 'loss_factor': 0.2}, # 森林，深绿色
            'Water': {'color': [255, 255, 0], 'loss_factor': 0.1}, # 水域，黄色
            'Other': {'color': [128, 128, 128], 'loss_factor': 0.3}, # 其他，灰色
        }
        
        # 风险等级定义
        self.risk_levels = {
            'Low': {'max_value': 0.2, 'color': '#4CAF50'}, # 绿色
            'Medium': {'max_value': 0.5, 'color': '#FFEB3B'}, # 黄色
            'High': {'max_value': 0.8, 'color': '#FF9800'}, # 橙色
            'Critical': {'max_value': float('inf'), 'color': '#F44336'}, # 红色
        }
    
    def prediction_intervals(self, predictions_df, confidence_level=95):
        """
        Calculate prediction intervals for the predictions.
        
        Parameters:
        -----------
        predictions_df : pandas.DataFrame
            DataFrame containing actual values, predictions, and errors
        confidence_level : int
            Confidence level for prediction intervals (default: 95)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with prediction intervals
        """
        # Make a copy of the predictions dataframe
        risk_df = predictions_df.copy()
        
        # Calculate the standard error of the predictions
        std_error = np.std(risk_df['error'])
        
        # Calculate the confidence interval factor based on the confidence level
        alpha = (100 - confidence_level) / 100
        z_value = stats.norm.ppf(1 - alpha/2)
        
        # Calculate prediction intervals
        risk_df['lower_bound'] = risk_df['predicted'] - z_value * std_error
        risk_df['upper_bound'] = risk_df['predicted'] + z_value * std_error
        
        # Calculate interval width
        risk_df['interval_width'] = risk_df['upper_bound'] - risk_df['lower_bound']
        
        # Check if actual value is within prediction interval
        if 'predicted' in risk_df.columns:
            target_col = [col for col in risk_df.columns if col not in 
                         ['predicted', 'error', 'lower_bound', 'upper_bound', 'interval_width']][0]
            risk_df['within_interval'] = (
                (risk_df[target_col] >= risk_df['lower_bound']) & 
                (risk_df[target_col] <= risk_df['upper_bound'])
            )
        
        return risk_df
    
    def error_distribution(self, predictions_df):
        """
        Analyze the distribution of prediction errors.
        
        Parameters:
        -----------
        predictions_df : pandas.DataFrame
            DataFrame containing predictions and errors
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with error distribution analysis
        """
        # Make a copy of the predictions dataframe
        risk_df = predictions_df.copy()
        
        # Calculate absolute error
        risk_df['abs_error'] = np.abs(risk_df['error'])
        
        # Calculate relative error (percentage)
        target_col = [col for col in risk_df.columns if col not in 
                     ['predicted', 'error', 'abs_error']][0]
        risk_df['rel_error'] = (risk_df['abs_error'] / np.abs(risk_df[target_col])) * 100
        
        # Calculate z-score of errors
        risk_df['error_zscore'] = stats.zscore(risk_df['error'])
        
        # Classify errors based on z-score
        risk_df['error_severity'] = pd.cut(
            np.abs(risk_df['error_zscore']),
            bins=[0, 1, 2, 3, np.inf],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        return risk_df
    
    def outlier_detection(self, predictions_df, threshold=3.0):
        """
        Detect outliers in predictions and errors.
        
        Parameters:
        -----------
        predictions_df : pandas.DataFrame
            DataFrame containing predictions and errors
        threshold : float
            Z-score threshold for outlier detection (default: 3.0)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with outlier detection results
        """
        # Make a copy of the predictions dataframe
        risk_df = predictions_df.copy()
        
        # Calculate z-scores for predictions and actual values
        target_col = [col for col in risk_df.columns if col not in 
                     ['predicted', 'error']][0]
        
        risk_df['actual_zscore'] = stats.zscore(risk_df[target_col])
        risk_df['predicted_zscore'] = stats.zscore(risk_df['predicted'])
        risk_df['error_zscore'] = stats.zscore(risk_df['error'])
        
        # Flag outliers based on z-scores
        risk_df['actual_outlier'] = np.abs(risk_df['actual_zscore']) > threshold
        risk_df['predicted_outlier'] = np.abs(risk_df['predicted_zscore']) > threshold
        risk_df['error_outlier'] = np.abs(risk_df['error_zscore']) > threshold
        
        # Combine outlier flags
        risk_df['is_outlier'] = (
            risk_df['actual_outlier'] | 
            risk_df['predicted_outlier'] | 
            risk_df['error_outlier']
        )
        
        # Calculate outlier severity
        risk_df['outlier_severity'] = pd.cut(
            np.maximum.reduce([
                np.abs(risk_df['actual_zscore']),
                np.abs(risk_df['predicted_zscore']),
                np.abs(risk_df['error_zscore'])
            ]),
            bins=[0, threshold, threshold*1.5, threshold*2, np.inf],
            labels=['Normal', 'Mild', 'Moderate', 'Severe']
        )
        
        return risk_df
    
    def generate_risk_summary(self, risk_assessment, assessment_method):
        """
        Generate a summary of risk assessment results.
        
        Parameters:
        -----------
        risk_assessment : pandas.DataFrame
            DataFrame with risk assessment results
        assessment_method : str
            Method used for risk assessment
            
        Returns:
        --------
        dict
            Dictionary with risk summary statistics
        """
        summary = {}
        
        if assessment_method == "Prediction Intervals":
            # Calculate percentage of actual values within prediction intervals
            within_interval_pct = risk_assessment['within_interval'].mean() * 100
            summary['within_interval_pct'] = f"{within_interval_pct:.2f}%"
            
            # Calculate average interval width
            avg_interval_width = risk_assessment['interval_width'].mean()
            summary['avg_interval_width'] = f"{avg_interval_width:.2f}"
            
            return f"""
            **Prediction Intervals Summary:**
            - {summary['within_interval_pct']} of actual values fall within the prediction intervals
            - Average interval width: {summary['avg_interval_width']}
            """
            
        elif assessment_method == "Error Distribution":
            # Calculate error statistics
            mean_abs_error = risk_assessment['abs_error'].mean()
            median_abs_error = risk_assessment['abs_error'].median()
            mean_rel_error = risk_assessment['rel_error'].mean()
            
            # Count errors by severity
            error_counts = risk_assessment['error_severity'].value_counts()
            
            return f"""
            **Error Distribution Summary:**
            - Mean absolute error: {mean_abs_error:.2f}
            - Median absolute error: {median_abs_error:.2f}
            - Mean relative error: {mean_rel_error:.2f}%
            - Error severity counts:
              - Low: {error_counts.get('Low', 0)}
              - Medium: {error_counts.get('Medium', 0)}
              - High: {error_counts.get('High', 0)}
              - Critical: {error_counts.get('Critical', 0)}
            """
            
        elif assessment_method == "Outlier Detection":
            # Calculate outlier statistics
            outlier_count = risk_assessment['is_outlier'].sum()
            outlier_percentage = (outlier_count / len(risk_assessment)) * 100
            
            # Count outliers by severity
            outlier_severity = risk_assessment['outlier_severity'].value_counts()
            
            return f"""
            **Outlier Detection Summary:**
            - Total outliers detected: {outlier_count} ({outlier_percentage:.2f}%)
            - Outlier severity counts:
              - Mild: {outlier_severity.get('Mild', 0)}
              - Moderate: {outlier_severity.get('Moderate', 0)}
              - Severe: {outlier_severity.get('Severe', 0)}
            """
            
        return "No summary available for the selected assessment method."
    
    def analyze_land_use_image(self, image_file):
        """
        分析土地利用图并计算单位损失。
        
        Parameters:
        -----------
        image_file : UploadedFile
            上传的土地利用图文件
            
        Returns:
        --------
        dict
            包含分析结果的字典
        """
        try:
            # 读取图像
            image_bytes = image_file.read()
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            # 图像尺寸
            height, width, _ = image.shape
            total_pixels = height * width
            
            # 分析每种土地利用类型的面积
            land_use_areas = {}
            unit_loss = 0.0
            
            # 创建分析结果图像
            result_image = image.copy()
            
            for land_type, info in self.land_use_types.items():
                # 创建颜色掩码（允许一定的颜色偏差）
                color = np.array(info['color'])
                lower_bound = np.maximum(0, color - 15)
                upper_bound = np.minimum(255, color + 15)
                
                # 找到该颜色范围内的像素
                mask = cv2.inRange(image, lower_bound, upper_bound)
                matching_pixels = cv2.countNonZero(mask)
                
                # 计算面积百分比
                area_percentage = (matching_pixels / total_pixels) * 100
                land_use_areas[land_type] = {
                    'pixels': matching_pixels,
                    'percentage': area_percentage,
                    'loss_factor': info['loss_factor']
                }
                
                # 计算该类型的单位损失贡献
                type_loss = area_percentage * info['loss_factor'] / 100
                unit_loss += type_loss
                
                # 在结果图像中标记该区域
                result_image[mask > 0] = color
            
            # 转换结果图像为base64
            _, buffer = cv2.imencode('.png', result_image)
            result_image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # 返回分析结果
            return {
                'land_use_areas': land_use_areas,
                'unit_loss': unit_loss,
                'result_image': result_image_b64,
                'total_area': total_pixels,
                'dimensions': f"{width}x{height}"
            }
            
        except Exception as e:
            raise Exception(f"Error analyzing land use image: {str(e)}")
    
    def predict_parameter(self, model, input_data):
        """
        使用训练好的模型预测目标参数。
        
        Parameters:
        -----------
        model : object
            训练好的模型对象
        input_data : dict
            输入参数的字典
            
        Returns:
        --------
        float
            预测的参数值
        """
        try:
            # 转换输入数据为模型可接受的格式
            input_df = pd.DataFrame([input_data])
            
            # 使用模型进行预测
            prediction = model.predict(input_df)
            
            # 返回预测值
            return float(prediction[0])
        except Exception as e:
            raise Exception(f"Error predicting parameter: {str(e)}")
    
    def calculate_risk_level(self, predicted_value, unit_loss):
        """
        根据预测值和单位损失计算风险等级。
        
        Parameters:
        -----------
        predicted_value : float
            预测的参数值
        unit_loss : float
            单位损失值
            
        Returns:
        --------
        dict
            包含风险评估结果的字典
        """
        try:
            # 计算总风险值
            total_risk = predicted_value * unit_loss
            
            # 确定风险等级
            risk_level = None
            for level, info in sorted(self.risk_levels.items(), key=lambda x: x[1]['max_value']):
                if total_risk <= info['max_value']:
                    risk_level = level
                    color = info['color']
                    break
            
            # 如果没有找到匹配的等级，使用最高等级
            if not risk_level:
                risk_level = 'Critical'
                color = self.risk_levels['Critical']['color']
            
            # 返回风险评估结果
            return {
                'predicted_value': predicted_value,
                'unit_loss': unit_loss,
                'total_risk': total_risk,
                'risk_level': risk_level,
                'color': color
            }
        except Exception as e:
            raise Exception(f"Error calculating risk level: {str(e)}")
    
    def visualize_risk_assessment(self, risk_result):
        """
        可视化风险评估结果。
        
        Parameters:
        -----------
        risk_result : dict
            风险评估结果字典
            
        Returns:
        --------
        str
            图像的base64编码
        """
        try:
            # 创建图表
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # 风险数据
            risk_level = risk_result['risk_level']
            total_risk = risk_result['total_risk']
            color = risk_result['color']
            
            # 创建仪表盘样式的图表
            max_val = 1.0
            sns.set_style("whitegrid")
            
            # 标签和注解
            ax.text(0.5, 0.3, f"Risk Level: {risk_level}", 
                    horizontalalignment='center', fontsize=18, fontweight='bold')
            ax.text(0.5, 0.2, f"Risk Value: {total_risk:.4f}", 
                    horizontalalignment='center', fontsize=14)
            
            # 绘制仪表盘
            theta = np.linspace(0, np.pi, 100)
            radius = 0.8
            x = radius * np.cos(theta) + 0.5
            y = radius * np.sin(theta) + 0.5
            
            # 绘制仪表盘背景
            ax.plot(x, y, color='lightgray', linewidth=20, alpha=0.5, solid_capstyle='round')
            
            # 计算风险指示器的位置
            risk_ratio = min(total_risk / max_val, 1.0)
            risk_theta = np.pi * (1 - risk_ratio)
            risk_x = radius * np.cos(risk_theta) + 0.5
            risk_y = radius * np.sin(risk_theta) + 0.5
            
            # 绘制风险指示器
            ax.plot([0.5, risk_x], [0.5, risk_y], color=color, linewidth=5, solid_capstyle='round')
            
            # 调整图表属性
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            # 添加标题
            ax.set_title('Risk Assessment', fontsize=16, pad=20)
            
            # 保存图像到内存缓冲区
            buf = io.BytesIO()
            fig.tight_layout()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # 转换为base64
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            return img_str
            
        except Exception as e:
            raise Exception(f"Error visualizing risk assessment: {str(e)}")
