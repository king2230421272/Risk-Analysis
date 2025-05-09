import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import io
import base64
from PIL import Image
import cv2
import importlib
import shutil
import os
import subprocess
import rasterio
import logging
import concurrent.futures

logger = logging.getLogger(__name__)

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
        risk_assessment : pandas.DataFrame or dict
            DataFrame with risk assessment results or dictionary with results
        assessment_method : str
            Method used for risk assessment
            
        Returns:
        --------
        dict or str
            Dictionary with risk summary statistics or formatted string summary
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
        
        elif assessment_method == "prob_loss":
            # 处理概率损失法的结果汇总
            summary = {}
            
            if isinstance(risk_assessment, pd.DataFrame):
                # 如果是DataFrame，计算平均风险值
                if 'prob' in risk_assessment.columns and 'loss' in risk_assessment.columns:
                    risk_assessment['risk'] = risk_assessment['prob'] * risk_assessment['loss']
                    avg_risk = risk_assessment['risk'].mean()
                else:
                    # 查找可能包含风险信息的列
                    risk_cols = [col for col in risk_assessment.columns if 'risk' in col.lower()]
                    if risk_cols:
                        avg_risk = risk_assessment[risk_cols[0]].mean()
                    else:
                        avg_risk = 0
                
                # 根据风险值确定风险等级
                if avg_risk < 0.3:
                    risk_level = "低风险"
                elif avg_risk < 0.6:
                    risk_level = "中风险"
                elif avg_risk < 0.8:
                    risk_level = "高风险"
                else:
                    risk_level = "极高风险"
                
                summary["risk_value"] = avg_risk
                summary["risk_level"] = risk_level
                
            elif isinstance(risk_assessment, dict):
                # 如果是字典，直接使用字典中的值
                if "risk_value" in risk_assessment:
                    avg_risk = risk_assessment["risk_value"]
                elif "risk" in risk_assessment:
                    avg_risk = risk_assessment["risk"]
                else:
                    avg_risk = 0
                    
                if "risk_level" in risk_assessment:
                    risk_level = risk_assessment["risk_level"]
                else:
                    # 根据风险值确定风险等级
                    if avg_risk < 0.3:
                        risk_level = "低风险"
                    elif avg_risk < 0.6:
                        risk_level = "中风险"
                    elif avg_risk < 0.8:
                        risk_level = "高风险"
                    else:
                        risk_level = "极高风险"
                
                summary["risk_value"] = avg_risk
                summary["risk_level"] = risk_level
            else:
                # 如果是单一数值，直接计算风险等级
                avg_risk = float(risk_assessment) if isinstance(risk_assessment, (int, float, str)) else 0
                
                # 根据风险值确定风险等级
                if avg_risk < 0.3:
                    risk_level = "低风险"
                elif avg_risk < 0.6:
                    risk_level = "中风险"
                elif avg_risk < 0.8:
                    risk_level = "高风险"
                else:
                    risk_level = "极高风险"
                
                summary["risk_value"] = avg_risk
                summary["risk_level"] = risk_level
            
            return summary
            
        elif assessment_method == "iahp_critic_gt":
            # 处理IAHP-CRITIC-GT法的结果汇总
            summary = {}
            
            if isinstance(risk_assessment, dict):
                # 提取风险得分
                if "risk_scores" in risk_assessment:
                    avg_score = sum(risk_assessment["risk_scores"]) / len(risk_assessment["risk_scores"])
                    max_score = max(risk_assessment["risk_scores"])
                    min_score = min(risk_assessment["risk_scores"])
                    
                    summary["avg_risk_score"] = avg_score
                    summary["max_risk_score"] = max_score
                    summary["min_risk_score"] = min_score
                    
                    # 根据平均风险得分确定风险等级
                    if avg_score < 0.3:
                        risk_level = "低风险"
                    elif avg_score < 0.6:
                        risk_level = "中风险"
                    elif avg_score < 0.8:
                        risk_level = "高风险"
                    else:
                        risk_level = "极高风险"
                    
                    summary["risk_level"] = risk_level
                    summary["risk_value"] = avg_score
                
                # 提取风险分类
                if "risk_categories" in risk_assessment:
                    category_counts = {}
                    for category in risk_assessment["risk_categories"]:
                        category_counts[category] = category_counts.get(category, 0) + 1
                    
                    summary["risk_category_counts"] = category_counts
                    
                # 提取权重信息
                if "weights" in risk_assessment:
                    summary["weights"] = risk_assessment["weights"]
            
            return summary
            
        elif assessment_method == "dynamic_bayes":
            # 处理动态贝叶斯网络法的结果汇总
            summary = {}
            
            if isinstance(risk_assessment, dict):
                # 提取预测概率
                if "risk_probabilities" in risk_assessment:
                    if isinstance(risk_assessment["risk_probabilities"], dict):
                        # 提取最高风险状态及其概率
                        max_state = max(risk_assessment["risk_probabilities"].items(), 
                                       key=lambda x: x[1])
                        summary["max_risk_state"] = max_state[0]
                        summary["max_risk_prob"] = max_state[1]
                        
                        # 如果有高风险状态，计算综合风险值
                        high_risk_states = [s for s, p in risk_assessment["risk_probabilities"].items() 
                                         if "高" in str(s) or "危险" in str(s) or "严重" in str(s)]
                        if high_risk_states:
                            high_risk_prob = sum(risk_assessment["risk_probabilities"][s] for s in high_risk_states)
                            summary["high_risk_prob"] = high_risk_prob
                            
                            # 根据高风险概率确定风险等级
                            if high_risk_prob < 0.3:
                                risk_level = "低风险"
                            elif high_risk_prob < 0.6:
                                risk_level = "中风险"
                            elif high_risk_prob < 0.8:
                                risk_level = "高风险"
                            else:
                                risk_level = "极高风险"
                                
                            summary["risk_level"] = risk_level
                            summary["risk_value"] = high_risk_prob
                        else:
                            # 如果没有明确的高风险状态，使用最高概率状态
                            if summary["max_risk_prob"] < 0.3:
                                risk_level = "低风险"
                            elif summary["max_risk_prob"] < 0.6:
                                risk_level = "中风险"
                            elif summary["max_risk_prob"] < 0.8:
                                risk_level = "高风险"
                            else:
                                risk_level = "极高风险"
                                
                            summary["risk_level"] = risk_level
                            summary["risk_value"] = summary["max_risk_prob"]
                    else:
                        # 如果风险概率是列表或数组
                        max_prob_index = np.argmax(risk_assessment["risk_probabilities"])
                        max_prob = risk_assessment["risk_probabilities"][max_prob_index]
                        
                        summary["max_risk_state"] = f"状态 {max_prob_index}"
                        summary["max_risk_prob"] = max_prob
                        
                        # 根据最大概率确定风险等级
                        if max_prob < 0.3:
                            risk_level = "低风险"
                        elif max_prob < 0.6:
                            risk_level = "中风险"
                        elif max_prob < 0.8:
                            risk_level = "高风险"
                        else:
                            risk_level = "极高风险"
                            
                        summary["risk_level"] = risk_level
                        summary["risk_value"] = max_prob
            
            return summary
            
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

    def assess_risk(self, data, method='prob_loss', **kwargs):
        """
        统一风险评估入口
        method: 'prob_loss' | 'iahp_critic_gt' | 'dynamic_bayes'
        data: pandas.DataFrame，包含所需字段
        其他参数通过kwargs传递
        """
        if method == 'prob_loss':
            return self._risk_prob_loss(data, **kwargs)
        elif method == 'iahp_critic_gt':
            return self._risk_iahp_critic_gt(data, **kwargs)
        elif method == 'dynamic_bayes':
            return self._risk_dynamic_bayes(data, **kwargs)
        else:
            raise ValueError("未知的风险评估方法")

    # HEC-RAS关键信息配置区块
    HEC_RAS_CONFIG = {
        'hec_ras_exe_path': r'C:/Program Files/HEC/HEC-RAS/6.0/HEC-RAS.exe',  # HEC-RAS主程序路径
        'template_dir': r'./hec_ras_template',  # 模板工程目录
        'work_base_dir': r'./hec_ras_work',     # 工作目录基路径
    }

    def prepare_hec_ras_input(self, template_dir, work_dir, qp_value, other_params=None):
        """
        复制模板工程到工作目录，并替换QP等参数
        """
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        for fname in os.listdir(template_dir):
            shutil.copy(os.path.join(template_dir, fname), work_dir)
        # 修改关键参数（如流量、边界条件等）
        plan_file = os.path.join(work_dir, 'Project.p01')
        if os.path.exists(plan_file):
            with open(plan_file, 'r') as f:
                content = f.read()
            content = content.replace('QP_PLACEHOLDER', str(qp_value))
            # 其他参数同理
            with open(plan_file, 'w') as f:
                f.write(content)
        return work_dir

    def run_hec_ras_batch(self, work_dir, hec_ras_exe_path=None):
        """
        调用HEC-RAS命令行批处理，返回输出洪水淹没图路径
        """
        if hec_ras_exe_path is None:
            hec_ras_exe_path = self.HEC_RAS_CONFIG['hec_ras_exe_path']
        prj_file = os.path.join(work_dir, 'Project.prj')
        cmd = f'"{hec_ras_exe_path}" "{prj_file}" /COM'
        result = subprocess.run(cmd, shell=True, cwd=work_dir)
        if result.returncode != 0:
            raise RuntimeError('HEC-RAS运行失败')
        output_flood_map = os.path.join(work_dir, 'max_flood_depth.tif')
        return output_flood_map

    def read_flood_map(self, flood_map_path):
        """
        读取洪水淹没图（GeoTIFF）
        """
        with rasterio.open(flood_map_path) as src:
            flood_data = src.read(1)
            profile = src.profile
        return flood_data, profile

    def hec_ras_flood_simulation(self, satellite_image_path, qp_value, hec_ras_params=None):
        """
        细化hec-ras洪水演进自动化流程
        """
        # 1. 路径配置
        template_dir = self.HEC_RAS_CONFIG['template_dir']
        work_base_dir = self.HEC_RAS_CONFIG['work_base_dir']
        hec_ras_exe_path = self.HEC_RAS_CONFIG['hec_ras_exe_path']
        # 2. 创建唯一工作目录
        import uuid
        work_dir = os.path.join(work_base_dir, str(uuid.uuid4()))
        # 3. 生成输入文件
        self.prepare_hec_ras_input(template_dir, work_dir, qp_value, hec_ras_params)
        # 4. 调用HEC-RAS
        output_flood_map = self.run_hec_ras_batch(work_dir, hec_ras_exe_path)
        # 5. 可选：读取输出数据
        # flood_data, profile = self.read_flood_map(output_flood_map)
        return output_flood_map

    def calculate_losses(self, flood_map_path, landuse_map_path, population_density, qp_value, w_df, v_total, warning_time, understanding_level, property_unit_values, property_loss_rates, year_gap=0, loss_growth_rate=0.03):
        """
        损失计算：输入洪水淹没图和土地利用图，输出生命损失、经济损失、社会环境影响。
        """
        # 1. 读取flood_map和landuse_map，统计h≥0.3m的面积A（伪代码，需实际遥感/GIS处理）
        A = 1.0  # km^2，示例
        # 2. 生命损失
        P_AR = population_density * A
        S_D = qp_value / w_df
        # f死亡率、a修正系数、警报时间、理解程度等需查表或业务逻辑
        f = 0.01  # 示例
        if v_total > 1e8:
            a = 1.5
        elif v_total > 1e7:
            a = 1.3
        elif v_total >= 1e5:
            a = 1.1
        else:
            a = 1.0
        L_OL = P_AR * f * a
        # 3. 经济损失
        # 统计各类财产的淹没面积Ai（伪代码）
        L_IE2 = sum(property_loss_rates.get(k, 1.0) * property_unit_values.get(k, 1.0) * 1.0 for k in property_unit_values)
        L_IE1 = 0  # 工程损毁损失，需外部输入
        L_IE = L_IE1 + L_IE2
        k = 0.63
        L_DE = k * L_IE
        n = year_gap
        L_E = (L_IE + L_DE) * (1 + loss_growth_rate) ** n
        # 4. 社会环境影响
        C_list = [1.0 for _ in range(9)]  # 9项系数，需实际赋值
        I_SE = np.prod(C_list)
        return {
            'life_loss': L_OL,
            'economic_loss': L_E,
            'social_env_index': I_SE
        }

    def _check_required_columns(self, data, required_cols):
        """检查DataFrame是否包含所需列"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("输入必须为pandas.DataFrame")
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            raise ValueError(f"数据中缺少列: {missing}")

    def _risk_prob_loss(self, data, prob_col='prob', loss_col='loss', **kwargs):
        """
        方法一：风险 = 溃决概率 × 损失，支持批量和单值输入，结构化返回。
        性能优化：支持向量化批量处理。
        """
        try:
            # 前置：自动化流程
            if isinstance(data, dict) and 'satellite_image_path' in data and 'predictor' in data:
                satellite_image_path = data['satellite_image_path']
                predictor = data['predictor']
                input_features = data['input_features']
                qp_value = predictor.predict_qp(input_features)
                flood_map = self.hec_ras_flood_simulation(satellite_image_path, qp_value)
                landuse_map = data.get('landuse_map_path')
                losses = self.calculate_losses(
                    flood_map, landuse_map, data.get('population_density',1), qp_value, data.get('w_df',1),
                    data.get('v_total',1e8), data.get('warning_time',1), data.get('understanding_level','明确'),
                    data.get('property_unit_values',{}), data.get('property_loss_rates',{}),
                    data.get('year_gap',0), data.get('loss_growth_rate',0.03))
                prob = predictor.predict_breach_probability(input_features)
                risk = prob * losses['economic_loss']
                return {
                    'input': data,
                    'prob': prob,
                    'loss': losses['economic_loss'],
                    'risk': risk,
                    'losses': losses,
                    'flood_map': flood_map
                }
            # 批量/单值兼容
            if isinstance(data, pd.DataFrame):
                self._check_required_columns(data, [prob_col, loss_col])
                # 向量化处理
                risk = data[prob_col].values * data[loss_col].values
                result = data[[prob_col, loss_col]].copy()
                result['risk'] = risk
                return result
            elif isinstance(data, dict):
                prob = data.get(prob_col)
                loss = data.get(loss_col)
                if prob is None or loss is None:
                    raise ValueError("字典输入需包含prob和loss")
                return {'input': data, 'prob': prob, 'loss': loss, 'risk': prob * loss}
            elif isinstance(data, (list, tuple)) and len(data) == 2:
                prob, loss = data
                return {'input': data, 'prob': prob, 'loss': loss, 'risk': prob * loss}
            else:
                raise ValueError("输入格式不支持")
        except Exception as e:
            logger.error(f"风险评估方法一出错: {e}")
            return {'error': str(e)}

    def _calc_iahp_weights(self, indicator_cols, expert_input=None, llm_service='auto', matrix=None):
        """
        IAHP主观权重计算，支持载入专家意见（文字/表格），并用大语言模型解析。
        参数：
            indicator_cols: 指标列名列表
            expert_input: 专家意见（str/表格/None）
            llm_service: LLM服务类型
            matrix: 专家成对比较矩阵（二维ndarray或list），若无则等权
        返回：
            ndarray，主观权重
        """
        # 优先用专家成对比较矩阵
        n = len(indicator_cols)
        if matrix is not None:
            matrix = np.array(matrix)
            eigvals, eigvecs = np.linalg.eig(matrix)
            max_index = np.argmax(eigvals)
            weights = np.real(eigvecs[:, max_index])
            weights = weights / weights.sum()
            return weights
        # 其次用专家意见文本/表格+LLM解析
        if expert_input is not None:
            try:
                llm_handler = None
                try:
                    llm_module = importlib.import_module('utils.llm_handler')
                    llm_handler = llm_module.LlmHandler()
                except Exception as e:
                    print(f"无法导入LLM Handler: {e}")
                if llm_handler and llm_handler.is_any_service_available():
                    # 构造prompt，要求输出与indicator_cols顺序一致的权重list
                    prompt = f"请根据以下专家意见，给出与下列指标顺序一致的主观权重（归一化，和为1），只返回Python list，不要解释。\n指标：{indicator_cols}\n专家意见：{expert_input}"
                    # 这里用OpenAI为例
                    result = llm_handler.parse_condition_text(prompt, [{"Column": col} for col in indicator_cols], service=llm_service)
                    # 解析list
                    if isinstance(result, list) and len(result) == n:
                        weights = np.array(result, dtype=float)
                        weights = weights / weights.sum()
                        return weights
                    elif isinstance(result, dict):
                        # 支持dict格式，如{"A":0.2,"B":0.3,...}
                        weights = np.array([result.get(col, 1.0/n) for col in indicator_cols], dtype=float)
                        weights = weights / weights.sum()
                        return weights
                    else:
                        print("LLM返回格式无法解析，使用等权")
                else:
                    print("无可用LLM服务，使用等权")
            except Exception as e:
                print(f"LLM解析专家意见失败：{e}，使用等权")
        # 默认等权
        return np.ones(n) / n

    def _calc_critic_weights(self, data, indicator_cols, model_details=None):
        """
        CRITIC客观权重计算，优先用预测模型训练后的feature_importance。
        参数：
            data: DataFrame
            indicator_cols: 指标列名列表
            model_details: 预测模型训练后的model_details（含feature_importance）
        返回：
            ndarray，客观权重
        """
        n = len(indicator_cols)
        # 优先用模型训练结果
        if model_details is not None and 'feature_importance' in model_details:
            fi = model_details['feature_importance']
            if hasattr(fi, 'tolist'):
                fi = fi.tolist()
            weights = np.array(fi, dtype=float)
            if len(weights) == n:
                weights = weights / weights.sum()
                return weights
        # 否则用CRITIC算法
        norm_data = (data[indicator_cols] - data[indicator_cols].min()) / (data[indicator_cols].max() - data[indicator_cols].min() + 1e-12)
        stds = norm_data.std()
        corr = norm_data.corr()
        C = 1 - corr.abs()
        info = stds * C.sum()
        weights = info / info.sum()
        return weights.values

    def _gt_fusion(self, iahp_weights, critic_weights, alpha=0.5):
        """
        GT博弈融合主客观权重
        参数：
            iahp_weights: ndarray，主观权重
            critic_weights: ndarray，客观权重
            alpha: 融合系数，默认0.5
        返回：
            ndarray，融合权重
        """
        return alpha * np.array(iahp_weights) + (1 - alpha) * np.array(critic_weights)

    def _risk_iahp_critic_gt(self, data, indicator_cols=None, expert_input=None, llm_service='auto', matrix=None, alpha=0.5, model_details=None, **kwargs):
        """
        方法二：IAHP-CRITIC-GT主客观博弈融合，结构化返回。
        """
        try:
            if indicator_cols is None:
                raise ValueError("请提供 indicator_cols（指标列名列表）")
            self._check_required_columns(data, indicator_cols)
            iahp_weights = self._calc_iahp_weights(indicator_cols, expert_input, llm_service, matrix)
            critic_weights = self._calc_critic_weights(data, indicator_cols, model_details)
            fusion_weights = self._gt_fusion(iahp_weights, critic_weights, alpha)
            data = data.copy()
            data['综合得分'] = (data[indicator_cols] * fusion_weights).sum(axis=1)
            for i, col in enumerate(indicator_cols):
                data[f'{col}_权重'] = fusion_weights[i]
            return {
                'input': data[indicator_cols],
                'weights': {col: fusion_weights[i] for i, col in enumerate(indicator_cols)},
                '综合得分': data['综合得分'],
                'result_df': data[indicator_cols + [f'{col}_权重' for col in indicator_cols] + ['综合得分']]
            }
        except Exception as e:
            logger.error(f"风险评估方法二出错: {e}")
            return {'error': str(e)}

    def _risk_dynamic_bayes(self, data, sequence_cols=None, n_states=3, predict_col=None, predict_index=-1, custom_structure=None, batch=False, **kwargs):
        """
        细化版：动态贝叶斯网络风险评估
        性能优化：支持批量推断。
        输入说明：
        - data: pandas.DataFrame，包含时序特征和溃决标签。每一行为一个样本，列为各时刻的特征和标签。
        - sequence_cols: list，时序特征和目标列名。
        - n_states: int，离散化级数。
        - predict_col: str，要预测的目标列名。
        - batch: bool，是否批量推断（True时对所有行做推断，False只对最后一行）。
        返回：dict，包括预测概率分布、风险等级、模型对象、结构等。
        """
        try:
            from pomegranate import BayesianNetwork
        except ImportError:
            logger.error("请先安装pomegranate库：pip install pomegranate")
            return {'error': '缺少pomegranate库'}
        if sequence_cols is None:
            sequence_cols = list(data.columns)
        self._check_required_columns(data, sequence_cols)
        # 离散化
        data_discrete = data[sequence_cols].apply(lambda x: pd.qcut(x, n_states, labels=False, duplicates='drop'))
        # 网络结构
        if custom_structure:
            model = BayesianNetwork.from_structure(data_discrete.values, custom_structure)
        else:
            model = BayesianNetwork.from_samples(data_discrete.values, algorithm='exact')
        if predict_col is None:
            predict_col = sequence_cols[predict_index]
        col_idx = sequence_cols.index(predict_col)
        results = []
        def single_predict(evidence):
            observed = {k: v for k, v in evidence.items() if k != predict_col}
            dist = model.predict_proba(observed)
            target_dist = dist[col_idx]
            if hasattr(target_dist, 'parameters'):
                probs = target_dist.parameters[0]
                max_state = max(probs, key=probs.get)
                risk_level = '高' if probs[max_state] > 0.5 and int(max_state) == n_states-1 else '中' if int(max_state) == n_states-2 else '低'
                return {'probs': probs, 'risk_level': risk_level}
            else:
                return {'error': '无法获取概率分布'}
        try:
            if batch:
                # 性能优化：多线程批量推断
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    evidences = [row.to_dict() for _, row in data_discrete.iterrows()]
                    batch_results = list(executor.map(single_predict, evidences))
                return {
                    'predict_col': predict_col,
                    'results': batch_results,
                    'model': model,
                    'structure': model.structure
                }
            else:
                evidence = data_discrete.iloc[-1].to_dict()
                single_result = single_predict(evidence)
                return {
                    'predict_col': predict_col,
                    'result': single_result,
                    'model': model,
                    'structure': model.structure
                }
        except Exception as e:
            logger.error(f"动态贝叶斯推断出错: {e}")
            return {'error': str(e)}

    # 示例用法
    # result = risk_assessor._risk_dynamic_bayes(
    #     data,
    #     sequence_cols=['X_t-2', 'X_t-1', 'X_t', 'is_breach'],
    #     n_states=3,
    #     predict_col='is_breach',
    #     batch=True
    # )
    # print(result['results'])
