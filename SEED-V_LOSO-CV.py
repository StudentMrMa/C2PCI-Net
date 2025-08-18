
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from main import Fusion_Model
from load_database import dataset_loaders
from configurations import Config
from main import weights_init
from main import custom_weights_init
import random
import numpy as np
import time
from datetime import timedelta
from sklearn.metrics import accuracy_score
start_time = time.time()

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

set_seed() 

config = Config(dataset_name='SEED-V')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 10
train_loaders, test_loaders = dataset_loaders(config.dataset_name, batch_size=config.batch_size)

best_result = []

for p in range(config.num_subjects):
    train_loader = train_loaders[p]
    test_loader = test_loaders[p]

    model = Fusion_Model(config).to(device)
    model.apply(weights_init)
    model.apply(custom_weights_init)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_f1 = 0.0, 0.0
    # 
    fused_x_outputs = []
    all_labels_list = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for eeg_map, eeg_stat, peri, labels in train_loader:
            eeg_map, eeg_stat, peri, labels = eeg_map.to(device), eeg_stat.to(device), peri.to(device), labels.to(device)
            labels = labels.squeeze()
            optimizer.zero_grad()
            _, _, _, _, e = model(eeg_map, eeg_stat, peri)
            loss = criterion(e, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * eeg_map.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        test_acc, all_labels, all_predictions = 0.0, [], []
        with torch.no_grad():
            for eeg_map, eeg_stat, peri, labels in test_loader:
                eeg_map, eeg_stat, peri, labels = eeg_map.to(device), eeg_stat.to(device), peri.to(device), labels.to(device)
                x_s1, _, _, _, e = model(eeg_map, eeg_stat, peri)
                _, predictions = torch.max(e, 1)
                test_acc += torch.sum(predictions == labels.data).item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                # 收集fused_x输出
                fused_x_outputs.append(x_s1.cpu().numpy())

        
        test_acc = accuracy_score(all_labels, all_predictions)
        test_f1 = f1_score(all_labels, all_predictions, average='macro')
        print(f"Subject ID {p+1}, Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.2f}%, Test F1: {test_f1:.2f}")

        if test_acc > best_acc:
            best_acc = test_acc
            best_f1 = test_f1

    best_result.append([p + 1, best_acc, best_f1])

print("Best Test Results:")
for result in best_result:
    print(f"Subject {result[0]}: Acc: {result[1]:.2f}%, F1: {result[2]:.2f}")
results_df = pd.DataFrame(best_result, columns=['Subject ID', 'Max Accuracy', 'Max F1 Score'])
results_df.to_excel(".xlsx", index=False)

end_time = time.time()
elapsed_time = end_time - start_time
formatted_time = str(timedelta(seconds=elapsed_time))
print(f"Total runtime: {formatted_time}")


