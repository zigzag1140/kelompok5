import torch
import torch.nn as nn

class Baseline1DCNN(nn.Module):
    def __init__(self, in_channels=90, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(self.features(x))


class ProposedCNNBiLSTM(nn.Module):
    def __init__(self, in_channels=90, hidden_dim=64, num_classes=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2)
        )
        self.bilstm = nn.LSTM(
            input_size=128, hidden_size=hidden_dim, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.2
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, num_classes)
        )
        
    def forward(self, x):
        x = self.cnn(x).permute(0, 2, 1)
        lstm_out, _ = self.bilstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.classifier(context)

if __name__ == "__main__":
    x_dummy = torch.randn(2, 90, 250)
    print("Baseline1DCNN Output Shape :", Baseline1DCNN()(x_dummy).shape)
    print("ProposedCNNBiLSTM Out Shape:", ProposedCNNBiLSTM()(x_dummy).shape)