import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TCNConfig:
    """时序卷积网络配置"""
    # 输入特征配置
    meteorological_features: int = 8  # 气象特征数量（辐照度、风速、温度、湿度等）
    historical_power_features: int = 1  # 历史发电量特征数量
    additional_features: int = 4  # 附加特征（时间编码等）
    
    # 网络结构配置
    num_channels: List[int] = None  # 每层通道数 [32, 64, 128, 128]
    kernel_size: int = 3  # 卷积核大小
    dropout: float = 0.2  # Dropout率
    
    # TCN特定参数
    dilation_base: int = 2  # 膨胀基数
    use_skip_connections: bool = True  # 是否使用跳跃连接
    use_batch_norm: bool = True  # 是否使用批归一化
    use_weight_norm: bool = True  # 是否使用权重归一化
    
    # 序列长度配置
    input_length: int = 288  # 输入序列长度（如48小时，5分钟采样）
    output_length: int = 144  # 输出序列长度（如24小时预测）
    
    # 训练配置
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 10
    
    # 数据预处理配置
    normalize: bool = True
    fill_missing: bool = True
    missing_threshold: float = 0.3  # 缺失值比例阈值
    
    def __post_init__(self):
        if self.num_channels is None:
            self.num_channels = [32, 64, 128, 128]
        
        self.total_input_features = (
            self.meteorological_features + 
            self.historical_power_features + 
            self.additional_features
        )
    
    def save(self, path: str):
        """保存配置到JSON文件"""
        config_dict = {
            'meteorological_features': self.meteorological_features,
            'historical_power_features': self.historical_power_features,
            'additional_features': self.additional_features,
            'num_channels': self.num_channels,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout,
            'dilation_base': self.dilation_base,
            'use_skip_connections': self.use_skip_connections,
            'use_batch_norm': self.use_batch_norm,
            'use_weight_norm': self.use_weight_norm,
            'input_length': self.input_length,
            'output_length': self.output_length,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'normalize': self.normalize,
            'fill_missing': self.fill_missing,
            'missing_threshold': self.missing_threshold
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        logger.info(f"配置已保存到 {path}")
    
    @classmethod
    def load(cls, path: str):
        """从JSON文件加载配置"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class Chomp1d(nn.Module):
    """裁剪模块，用于因果卷积"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """时序卷积块"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2,
                 use_batch_norm=True, use_weight_norm=True):
        super(TemporalBlock, self).__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        if use_weight_norm:
            self.conv1 = nn.utils.weight_norm(self.conv1)
        
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # 第二个卷积层
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        if use_weight_norm:
            self.conv2 = nn.utils.weight_norm(self.conv2)
        
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # 批归一化
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(n_outputs)
            self.bn2 = nn.BatchNorm1d(n_outputs)
        
        # 残差连接
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        # 第一个卷积路径
        out = self.conv1(x)
        out = self.chomp1(out)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # 第二个卷积路径
        out = self.conv2(out)
        out = self.chomp2(out)
        if self.use_batch_norm:
            out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # 残差连接
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """时序卷积网络"""
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2,
                 use_batch_norm=True, use_weight_norm=True, dilation_base=2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = dilation_base ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                padding=padding, dropout=dropout, use_batch_norm=use_batch_norm,
                use_weight_norm=use_weight_norm
            ))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class AttentionModule(nn.Module):
    """注意力机制模块"""
    def __init__(self, hidden_dim):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, channels)
        attention_weights = F.softmax(self.attention(x), dim=1)
        attended = torch.sum(attention_weights * x, dim=1)  # (batch, channels)
        return attended, attention_weights


class PVPowerPredictionTCN(nn.Module):
    """光伏发电量预测时序卷积网络"""
    def __init__(self, config: TCNConfig):
        super(PVPowerPredictionTCN, self).__init__()
        self.config = config
        
        # TCN主干网络
        self.tcn = TemporalConvNet(
            num_inputs=config.total_input_features,
            num_channels=config.num_channels,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            use_batch_norm=config.use_batch_norm,
            use_weight_norm=config.use_weight_norm,
            dilation_base=config.dilation_base
        )
        
        # 注意力机制（可选）
        self.use_attention = True
        if self.use_attention:
            self.attention = AttentionModule(config.num_channels[-1])
        
        # 跳跃连接（可选）
        if config.use_skip_connections:
            self.skip_conv = nn.Conv1d(
                config.total_input_features,
                config.num_channels[-1],
                kernel_size=1
            )
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(config.num_channels[-1], config.num_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.num_channels[-1] // 2, config.output_length)
        )
        
        # 用于保存归一化参数
        self.register_buffer('mean', torch.zeros(config.total_input_features))
        self.register_buffer('std', torch.ones(config.total_input_features))
        self.register_buffer('target_mean', torch.zeros(1))
        self.register_buffer('target_std', torch.ones(1))
        
        logger.info(f"TCN模型初始化完成: {self.count_parameters()} 参数")
    
    def count_parameters(self):
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        """
        前向传播
        x shape: (batch, seq_len, features) -> (batch, features, seq_len)
        """
        # 转换维度: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # TCN特征提取
        tcn_out = self.tcn(x)  # (batch, channels, seq_len)
        
        # 跳跃连接
        if self.config.use_skip_connections:
            skip_out = self.skip_conv(x)
            tcn_out = tcn_out + skip_out
        
        # 注意力机制或全局池化
        if self.use_attention:
            features, _ = self.attention(tcn_out)
        else:
            # 全局平均池化
            features = torch.mean(tcn_out, dim=2)  # (batch, channels)
        
        # 输出预测
        output = self.fc(features)  # (batch, output_length)
        
        return output


class PVDataset(Dataset):
    """光伏数据集"""
    def __init__(self, meteorological_data: np.ndarray, 
                 historical_power: np.ndarray,
                 target_power: Optional[np.ndarray] = None,
                 additional_features: Optional[np.ndarray] = None):
        """
        参数:
            meteorological_data: 气象数据 (N, seq_len, met_features)
            historical_power: 历史发电量 (N, seq_len, 1)
            target_power: 目标发电量 (N, output_len)
            additional_features: 附加特征 (N, seq_len, add_features)
        """
        self.meteorological_data = torch.FloatTensor(meteorological_data)
        self.historical_power = torch.FloatTensor(historical_power)
        self.target_power = torch.FloatTensor(target_power) if target_power is not None else None
        self.additional_features = torch.FloatTensor(additional_features) if additional_features is not None else None
        
        # 合并所有输入特征
        features_list = [self.meteorological_data, self.historical_power]
        if self.additional_features is not None:
            features_list.append(self.additional_features)
        
        self.input_features = torch.cat(features_list, dim=-1)
    
    def __len__(self):
        return len(self.input_features)
    
    def __getitem__(self, idx):
        if self.target_power is not None:
            return self.input_features[idx], self.target_power[idx]
        return self.input_features[idx]


class DataPreprocessor:
    """数据预处理器"""
    def __init__(self, config: TCNConfig):
        self.config = config
        self.mean = None
        self.std = None
        self.target_mean = None
        self.target_std = None
    
    def add_time_features(self, timestamps: np.ndarray) -> np.ndarray:
        """添加时间编码特征"""
        # 假设timestamps是Unix时间戳或datetime对象
        hour = np.array([ts.hour for ts in timestamps])
        day_of_week = np.array([ts.dayofweek for ts in timestamps])
        day_of_year = np.array([ts.dayofyear for ts in timestamps])
        month = np.array([ts.month for ts in timestamps])
        
        # 周期性编码
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day_of_year / 365)
        day_cos = np.cos(2 * np.pi * day_of_year / 365)
        
        time_features = np.stack([hour_sin, hour_cos, day_sin, day_cos], axis=-1)
        return time_features
    
    def handle_missing_values(self, data: np.ndarray) -> np.ndarray:
        """处理缺失值"""
        if not self.config.fill_missing:
            return data
        
        # 检查缺失值比例
        missing_ratio = np.isnan(data).sum() / data.size
        if missing_ratio > self.config.missing_threshold:
            logger.warning(f"缺失值比例 {missing_ratio:.2%} 超过阈值 {self.config.missing_threshold:.2%}")
        
        # 使用线性插值填充
        for i in range(data.shape[-1]):
            feature_data = data[..., i]
            mask = np.isnan(feature_data)
            if mask.any():
                # 前向填充 + 后向填充
                feature_data[mask] = np.interp(
                    np.flatnonzero(mask),
                    np.flatnonzero(~mask),
                    feature_data[~mask]
                )
                data[..., i] = feature_data
        
        return data
    
    def normalize(self, data: np.ndarray, is_target: bool = False, fit: bool = True):
        """归一化数据"""
        if not self.config.normalize:
            return data
        
        if is_target:
            if fit:
                self.target_mean = np.mean(data)
                self.target_std = np.std(data) + 1e-8
            return (data - self.target_mean) / self.target_std
        else:
            if fit:
                self.mean = np.mean(data, axis=(0, 1), keepdims=True)
                self.std = np.std(data, axis=(0, 1), keepdims=True) + 1e-8
            return (data - self.mean) / self.std
    
    def denormalize(self, data: np.ndarray, is_target: bool = False):
        """反归一化"""
        if not self.config.normalize:
            return data
        
        if is_target:
            return data * self.target_std + self.target_mean
        else:
            return data * self.std + self.mean
    
    def create_sequences(self, meteorological_data: np.ndarray,
                        historical_power: np.ndarray,
                        target_power: np.ndarray,
                        timestamps: Optional[np.ndarray] = None) -> Tuple:
        """创建时序样本"""
        n_samples = len(historical_power) - self.config.input_length - self.config.output_length + 1
        
        met_sequences = []
        power_sequences = []
        target_sequences = []
        time_sequences = []
        
        for i in range(n_samples):
            # 输入序列
            met_seq = meteorological_data[i:i+self.config.input_length]
            power_seq = historical_power[i:i+self.config.input_length]
            
            # 目标序列
            target_seq = target_power[
                i+self.config.input_length:i+self.config.input_length+self.config.output_length
            ]
            
            met_sequences.append(met_seq)
            power_sequences.append(power_seq)
            target_sequences.append(target_seq)
            
            # 时间特征
            if timestamps is not None:
                time_seq = timestamps[i:i+self.config.input_length]
                time_features = self.add_time_features(time_seq)
                time_sequences.append(time_features)
        
        met_sequences = np.array(met_sequences)
        power_sequences = np.array(power_sequences)
        target_sequences = np.array(target_sequences)
        
        if timestamps is not None:
            time_sequences = np.array(time_sequences)
            return met_sequences, power_sequences, target_sequences, time_sequences
        
        return met_sequences, power_sequences, target_sequences


class TCNTrainer:
    """TCN训练器"""
    def __init__(self, model: PVPowerPredictionTCN, config: TCNConfig, device: str = 'cuda'):
        self.model = model
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 优化器和损失函数
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        self.criterion = nn.MSELoss()
        
        # 早停机制
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        logger.info(f"训练器初始化完成，使用设备: {self.device}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """完整训练流程"""
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 早停检查
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                # 保存最佳模型
                self.save_checkpoint('best_model.pth')
            else:
                self.patience_counter += 1
            
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )
            
            # 早停
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"早停触发，在epoch {epoch+1}")
                break
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_loss': self.best_loss
        }
    
    def save_checkpoint(self, path: str):
        """保存模型检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss
        }
        torch.save(checkpoint, path)
        logger.info(f"模型已保存到 {path}")
    
    def load_checkpoint(self, path: str):
        """加载模型检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_loss = checkpoint['best_loss']
        logger.info(f"模型已从 {path} 加载")


class TCNPredictor:
    """TCN预测器"""
    def __init__(self, model: PVPowerPredictionTCN, 
                 preprocessor: DataPreprocessor,
                 device: str = 'cuda'):
        self.model = model
        self.preprocessor = preprocessor
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, meteorological_data: np.ndarray,
                historical_power: np.ndarray,
                timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """
        执行预测
        
        参数:
            meteorological_data: 气象数据 (seq_len, met_features)
            historical_power: 历史发电量 (seq_len, 1)
            timestamps: 时间戳 (seq_len,)
        
        返回:
            预测的发电量序列 (output_length,)
        """
        # 处理缺失值
        meteorological_data = self.preprocessor.handle_missing_values(meteorological_data)
        historical_power = self.preprocessor.handle_missing_values(historical_power)
        
        # 归一化
        meteorological_data = self.preprocessor.normalize(meteorological_data, fit=False)
        historical_power = self.preprocessor.normalize(
            historical_power.reshape(-1, 1), is_target=True, fit=False
        )
        
        # 构建输入
        features_list = [
            meteorological_data.reshape(1, -1, meteorological_data.shape[-1]),
            historical_power.reshape(1, -1, 1)
        ]
        
        if timestamps is not None:
            time_features = self.preprocessor.add_time_features(timestamps)
            features_list.append(time_features.reshape(1, -1, time_features.shape[-1]))
        
        input_tensor = torch.FloatTensor(
            np.concatenate(features_list, axis=-1)
        ).to(self.device)
        
        # 预测
        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = output.cpu().numpy()[0]
        
        # 反归一化
        prediction = self.preprocessor.denormalize(prediction, is_target=True)
        
        return prediction


# 使用示例
if __name__ == "__main__":
    # 配置
    config = TCNConfig(
        meteorological_features=8,
        historical_power_features=1,
        additional_features=4,
        num_channels=[32, 64, 128, 128],
        kernel_size=3,
        dropout=0.2,
        input_length=288,  # 24小时，5分钟采样
        output_length=144,  # 12小时预测
        batch_size=32,
        num_epochs=100
    )
    
    # 创建模型
    model = PVPowerPredictionTCN(config)
    print(f"模型参数数量: {model.count_parameters():,}")
    
    # 创建模拟数据
    n_samples = 1000
    met_data = np.random.randn(n_samples, config.input_length, config.meteorological_features)
    power_data = np.random.randn(n_samples, config.input_length, 1) * 100 + 300
    target_data = np.random.randn(n_samples, config.output_length) * 80 + 280
    time_features = np.random.randn(n_samples, config.input_length, config.additional_features)
    
    # 创建数据集
    dataset = PVDataset(met_data, power_data, target_data, time_features)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # 训练
    trainer = TCNTrainer(model, config, device='cpu')
    history = trainer.train(train_loader, val_loader)
    
    print(f"训练完成，最佳验证损失: {history['best_loss']:.6f}")
    
    # 预测示例
    preprocessor = DataPreprocessor(config)
    predictor = TCNPredictor(model, preprocessor, device='cpu')
    
    test_met = np.random.randn(config.input_length, config.meteorological_features)
    test_power = np.random.randn(config.input_length, 1) * 100 + 300
    
    prediction = predictor.predict(test_met, test_power)
    print(f"预测输出形状: {prediction.shape}")
    print(f"预测均值: {prediction.mean():.2f} kW")
