import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AlertInfo:
    """云层过境预警信息"""
    station_id: str  # 受影响电站标识
    start_time: float  # 预计开始时刻 t_s
    end_time: float  # 预计结束时刻 t_e
    intensity: float  # 影响强度 I (正值代表遮蔽增强)
    confidence: float  # 置信度 C (0-1)
    
    def __post_init__(self):
        """验证输入参数"""
        assert 0 <= self.confidence <= 1, "置信度必须在[0,1]范围内"
        assert self.start_time < self.end_time, "开始时刻必须早于结束时刻"
        assert self.intensity >= 0, "影响强度必须为非负值"


class CorrectionConfig:
    """修正模块配置参数"""
    def __init__(self):
        # 时间移位参数
        self.beta = 1.0  # 影响信号到时间移位的映射系数
        self.delta_min = -100.0  # 时间移位下限(秒)
        self.delta_max = 100.0  # 时间移位上限(秒)
        
        # 幅值缩放参数
        self.eta = 0.5  # 影响信号到缩放的灵敏系数
        self.A_min = 0.1  # 幅值下限
        self.A_max = 1.0  # 幅值上限
        
        # 功率约束参数
        self.P_max = 1000.0  # 最大有功功率(kW)
        self.S_up = 50.0  # 上升斜率上限(kW/s)
        self.S_down = 50.0  # 下降斜率上限(kW/s)
        
        # 融合权重参数
        self.c0 = 0.3  # 融合权重基准项
        self.c1 = 0.7  # 置信度到融合权重的线性增益
        self.lambda_min = 0.0  # 融合权重下限
        self.lambda_max = 1.0  # 融合权重上限
        
        # 其他参数
        self.min_interval = 30.0  # 预警最小间隔(秒)
        self.min_duration = 60.0  # 预警最小持续时间(秒)
        self.signal_threshold = 0.01  # 影响信号阈值，低于此值冻结修正
        
    def update_from_dict(self, config_dict: Dict):
        """从字典更新配置"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"未知配置参数: {key}")


class PowerCorrectionModule:
    """光伏发电量预测修正模块"""
    
    def __init__(self, config: Optional[CorrectionConfig] = None):
        self.config = config if config else CorrectionConfig()
        self.last_output = None  # 保存上一时刻的输出
        self.degraded_mode = False  # 降级模式标志
        
    def clip(self, x: float, x_min: float, x_max: float) -> float:
        """截断函数"""
        return max(x_min, min(x_max, x))
    
    def indicator_function(self, t: float, t_start: float, t_end: float) -> float:
        """区间指示函数 χ"""
        return 1.0 if t_start <= t <= t_end else 0.0
    
    def preprocess_alerts(self, alerts: List[AlertInfo]) -> List[AlertInfo]:
        """预处理预警序列：合并过近事件、过滤过短事件"""
        if not alerts:
            return []
        
        # 按开始时间排序
        sorted_alerts = sorted(alerts, key=lambda a: a.start_time)
        processed = []
        
        i = 0
        while i < len(sorted_alerts):
            current = sorted_alerts[i]
            
            # 检查持续时间
            duration = current.end_time - current.start_time
            if duration < self.config.min_duration:
                # 扩展至最小持续时间
                current.end_time = current.start_time + self.config.min_duration
                logger.info(f"预警持续时间过短，已扩展至{self.config.min_duration}秒")
            
            # 尝试合并连续的近距离事件
            j = i + 1
            while j < len(sorted_alerts):
                next_alert = sorted_alerts[j]
                if next_alert.start_time - current.end_time < self.config.min_interval:
                    # 合并事件
                    current.end_time = max(current.end_time, next_alert.end_time)
                    # 加权平均强度和置信度
                    weight1 = current.end_time - current.start_time
                    weight2 = next_alert.end_time - next_alert.start_time
                    total_weight = weight1 + weight2
                    current.intensity = (current.intensity * weight1 + 
                                       next_alert.intensity * weight2) / total_weight
                    current.confidence = (current.confidence * weight1 + 
                                        next_alert.confidence * weight2) / total_weight
                    logger.info(f"合并间隔过近的预警事件")
                    j += 1
                else:
                    break
            
            processed.append(current)
            i = j
        
        return processed
    
    def compute_impact_signal(self, t: float, alerts: List[AlertInfo]) -> float:
        """计算综合影响信号 r(t)"""
        r_t = 0.0
        for alert in alerts:
            chi = self.indicator_function(t, alert.start_time, alert.end_time)
            r_t += alert.confidence * alert.intensity * chi
        return r_t
    
    def compute_confidence_sum(self, t: float, alerts: List[AlertInfo]) -> float:
        """计算综合置信度 C(t)"""
        c_t = 0.0
        for alert in alerts:
            chi = self.indicator_function(t, alert.start_time, alert.end_time)
            c_t += alert.confidence * chi
        return c_t
    
    def time_shift_correction(self, 
                            base_prediction: np.ndarray,
                            time_axis: np.ndarray,
                            impact_signal: np.ndarray) -> np.ndarray:
        """时间移位修正"""
        n = len(time_axis)
        shifted_prediction = np.zeros_like(base_prediction)
        
        for k in range(n):
            # 计算时间移位量
            delta_k = self.clip(
                self.config.beta * impact_signal[k],
                self.config.delta_min,
                self.config.delta_max
            )
            
            # 冻结小信号的移位
            if abs(impact_signal[k]) < self.config.signal_threshold:
                delta_k = 0.0
            
            # 计算移位后的时间点
            shifted_time = time_axis[k] + delta_k
            
            # 插值获取移位后的预测值
            if shifted_time < time_axis[0]:
                # 镜像延拓
                shifted_prediction[k] = base_prediction[0]
            elif shifted_time > time_axis[-1]:
                # 边界保持
                shifted_prediction[k] = base_prediction[-1]
            else:
                # 线性插值
                shifted_prediction[k] = np.interp(shifted_time, time_axis, base_prediction)
        
        return shifted_prediction
    
    def amplitude_correction(self,
                           shifted_prediction: np.ndarray,
                           impact_signal: np.ndarray,
                           dt: float) -> np.ndarray:
        """幅值修正与约束"""
        n = len(shifted_prediction)
        corrected = np.zeros_like(shifted_prediction)
        
        for k in range(n):
            # 计算幅值缩放因子
            A_k = self.clip(
                1.0 - self.config.eta * impact_signal[k],
                self.config.A_min,
                self.config.A_max
            )
            
            # 应用缩放并施加功率饱和
            Y_p = self.clip(
                A_k * shifted_prediction[k],
                0.0,
                self.config.P_max
            )
            
            # 施加斜率约束
            if k == 0:
                if self.last_output is not None:
                    Y_prev = self.last_output
                else:
                    Y_prev = Y_p  # 首次运行，无历史数据
            else:
                Y_prev = corrected[k-1]
            
            # 上下行斜率限制
            max_increase = Y_prev + self.config.S_up * dt
            max_decrease = Y_prev - self.config.S_down * dt
            
            corrected[k] = self.clip(Y_p, max_decrease, max_increase)
        
        return corrected
    
    def fusion_correction(self,
                        base_prediction: np.ndarray,
                        corrected_prediction: np.ndarray,
                        confidence_sum: np.ndarray) -> np.ndarray:
        """融合修正序列与基础序列"""
        n = len(base_prediction)
        final_prediction = np.zeros_like(base_prediction)
        
        for k in range(n):
            # 计算融合权重
            lambda_k = self.clip(
                self.config.c0 + self.config.c1 * confidence_sum[k],
                self.config.lambda_min,
                self.config.lambda_max
            )
            
            # 加权融合
            final_prediction[k] = (lambda_k * corrected_prediction[k] + 
                                  (1 - lambda_k) * base_prediction[k])
        
        return final_prediction
    
    def correct(self,
               base_prediction: np.ndarray,
               time_axis: np.ndarray,
               alerts: List[AlertInfo],
               sampling_period: float = 1.0) -> Dict[str, np.ndarray]:
        """
        执行完整的修正流程
        
        参数:
            base_prediction: 基础发电量预测序列 B(t_k)
            time_axis: 时间轴 (秒)
            alerts: 云层过境预警序列
            sampling_period: 采样周期 Δt (秒)
            
        返回:
            包含中间结果和最终预测的字典
        """
        try:
            n = len(base_prediction)
            
            # 验证输入
            assert len(time_axis) == n, "时间轴长度必须与预测序列一致"
            assert sampling_period > 0, "采样周期必须为正值"
            
            # 预处理预警序列
            processed_alerts = self.preprocess_alerts(alerts)
            
            if not processed_alerts:
                logger.info("无有效预警，直接返回基础预测")
                self.last_output = base_prediction[-1] if len(base_prediction) > 0 else None
                return {
                    'final_prediction': base_prediction.copy(),
                    'base_prediction': base_prediction.copy(),
                    'impact_signal': np.zeros(n),
                    'confidence_sum': np.zeros(n),
                    'fusion_weight': np.zeros(n)
                }
            
            # 步骤1: 计算影响信号和置信度
            impact_signal = np.array([
                self.compute_impact_signal(t, processed_alerts) 
                for t in time_axis
            ])
            
            confidence_sum = np.array([
                self.compute_confidence_sum(t, processed_alerts)
                for t in time_axis
            ])
            
            # 步骤2: 时间移位修正
            shifted_prediction = self.time_shift_correction(
                base_prediction, time_axis, impact_signal
            )
            
            # 步骤3: 幅值修正与约束
            corrected_prediction = self.amplitude_correction(
                shifted_prediction, impact_signal, sampling_period
            )
            
            # 步骤4: 融合基础序列与修正序列
            final_prediction = self.fusion_correction(
                base_prediction, corrected_prediction, confidence_sum
            )
            
            # 更新历史输出
            self.last_output = final_prediction[-1]
            self.degraded_mode = False
            
            # 计算融合权重用于分析
            fusion_weight = np.array([
                self.clip(
                    self.config.c0 + self.config.c1 * c,
                    self.config.lambda_min,
                    self.config.lambda_max
                ) for c in confidence_sum
            ])
            
            logger.info(f"修正完成，处理了{len(processed_alerts)}条预警")
            
            return {
                'final_prediction': final_prediction,
                'base_prediction': base_prediction,
                'shifted_prediction': shifted_prediction,
                'corrected_prediction': corrected_prediction,
                'impact_signal': impact_signal,
                'confidence_sum': confidence_sum,
                'fusion_weight': fusion_weight
            }
            
        except Exception as e:
            logger.error(f"修正过程出错: {str(e)}")
            # 降级模式：返回基础预测
            self.degraded_mode = True
            return {
                'final_prediction': base_prediction.copy(),
                'base_prediction': base_prediction.copy(),
                'impact_signal': np.zeros(n),
                'confidence_sum': np.zeros(n),
                'fusion_weight': np.zeros(n),
                'error': str(e)
            }
    
    def reset(self):
        """重置模块状态"""
        self.last_output = None
        self.degraded_mode = False
        logger.info("修正模块已重置")


# 使用示例
if __name__ == "__main__":
    # 创建配置
    config = CorrectionConfig()
    config.P_max = 500.0  # 500kW装机容量
    config.S_up = 30.0
    config.S_down = 40.0
    
    # 创建修正模块
    corrector = PowerCorrectionModule(config)
    
    # 生成模拟数据
    time_axis = np.arange(0, 3600, 10)  # 1小时，10秒采样
    base_prediction = 300 + 100 * np.sin(time_axis / 1800 * np.pi)  # 模拟基础预测
    
    # 创建预警序列
    alerts = [
        AlertInfo(
            station_id="station_001",
            start_time=600,
            end_time=900,
            intensity=0.6,
            confidence=0.8
        ),
        AlertInfo(
            station_id="station_001",
            start_time=1800,
            end_time=2100,
            intensity=0.4,
            confidence=0.7
        )
    ]
    
    # 执行修正
    results = corrector.correct(
        base_prediction=base_prediction,
        time_axis=time_axis,
        alerts=alerts,
        sampling_period=10.0
    )
    
    # 输出结果统计
    print(f"基础预测均值: {results['base_prediction'].mean():.2f} kW")
    print(f"修正后预测均值: {results['final_prediction'].mean():.2f} kW")
    print(f"最大修正幅度: {np.abs(results['final_prediction'] - results['base_prediction']).max():.2f} kW")
    print(f"平均融合权重: {results['fusion_weight'].mean():.3f}")
