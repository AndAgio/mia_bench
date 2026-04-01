import torch
import torch.nn as nn
import torch.nn.functional as F


class TabularMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[1024, 512, 256], dropout_rate=0.2):
        """
        Args:
            input_dim (int): The number of features in the dataset.
            num_classes (int): The number of output classes.
            hidden_dims (list): A list specifying the number of neurons in each hidden layer.
            dropout_rate (float): Dropout probability to control overfitting (crucial for MIA tuning).
        """
        super(TabularMLP, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        # Dynamically build hidden layers based on the hidden_dims list
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
            
        # Compile the hidden layers into a sequential block
        self.feature_extractor = nn.Sequential(*layers)
        
        # Final classification head
        self.classifier = nn.Linear(current_dim, num_classes)

    def forward(self, x):
        """Forward pass through the network."""
        # Pass input through the hidden layers
        features = self.feature_extractor(x)
        
        # Pass the extracted features to the final classifier layer
        logits = self.classifier(features)
        
        return logits


# ==========================================
# Example usage across datasets:
# ==========================================
if __name__ == "__main__":
    
    # 1. Purchase100 Configuration
    # (Typically 600 features, 100 classes)
    model_purchase = TabularMLP(
        input_dim=600, 
        num_classes=100, 
        hidden_dims=[1024, 512, 256]
    )
    print("Purchase100 Model:", model_purchase)
    
    # 2. Texas100 Configuration
    # (Typically 6169 features, 100 classes)
    model_texas = TabularMLP(
        input_dim=6169, 
        num_classes=100, 
        hidden_dims=[1024, 512, 256]
    )
    print("\nTexas100 Model:", model_texas)
    
    # 3. AGNews Configuration
    # (Typically 134410 features, 20 classes)
    model_agnews = TabularMLP(
        input_dim=134410, 
        num_classes=20, 
        hidden_dims=[1024, 512, 256]
    )
    print("\nAGNews Model:", model_agnews)
    
    # Test a dummy forward pass for the Texas model
    batch_size = 32
    dummy_input = torch.randn(batch_size, 6169)
    output_logits = model_texas(dummy_input)
    print(f"\nDummy forward pass output shape (Texas100): {output_logits.shape}") 
    # Expected: [32, 100]