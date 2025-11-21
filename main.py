import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

# Importación de módulos locales
from dataset import ASVspoofDataset       
from model import DeepFakeVoiceDetector   

# --- CONFIGURACIÓN ---
BASE_PATH_AUDIO = "./LA/ASVspoof2019_LA_train/flac"
PROTOCOL_PATH = "./LA/ASVspoof2019_LA_train/ASVspoof2019.LA.cm.train.trn.txt"
CHECKPOINT_PATH = "best_model_deepfake.pth"

# Hiperparámetros
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 15 
WEIGHT_DECAY = 1e-4

def audio_augmentations(y):
    """Aplica transformaciones aleatorias para robustez."""
    if np.random.rand() < 0.5:
        return y 
        
    aug_type = np.random.choice(['noise', 'gain'])

    if aug_type == 'noise':
        noise_amp = 0.005 * np.random.rand() * np.max(y)
        y = y + noise_amp * np.random.normal(size=len(y))
        
    elif aug_type == 'gain':
        gain_factor = 0.8 + (np.random.rand() * 0.4) 
        y = y * gain_factor

    return y

def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            running_loss += loss.item()
            
            probs = torch.sigmoid(outputs.squeeze())
            predictions = (probs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
    return running_loss / len(loader), 100 * correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Instanciar Datasets
 
    train_dataset_full = ASVspoofDataset(BASE_PATH_AUDIO, PROTOCOL_PATH, transform=audio_augmentations, is_train=True)
   
    val_dataset_full = ASVspoofDataset(BASE_PATH_AUDIO, PROTOCOL_PATH, transform=None, is_train=False)
    
    # 2. Split Manual Determinista (Evita Data Leakage)
    dataset_size = len(train_dataset_full)
    indices = list(range(dataset_size))
    split = int(np.floor(0.8 * dataset_size))
    
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_indices, val_indices = indices[:split], indices[split:]

    train_ds = Subset(train_dataset_full, train_indices)
    val_ds = Subset(val_dataset_full, val_indices)

    # 3. DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Muestras: {len(train_ds)} Train | {len(val_ds)} Val")

    # 4. Modelo y Optimizador
    model = DeepFakeVoiceDetector().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_acc = 0.0

    # 5. Ciclo de Entrenamiento
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Ep {epoch+1} [Batch {i}] Loss: {loss.item():.4f}")

        # Validación
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        print(f"Ep {epoch+1} Resumen -> Train Loss: {running_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"Checkpoint guardado: {val_acc:.2f}%")

    print(f"Entrenamiento finalizado. Mejor Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()