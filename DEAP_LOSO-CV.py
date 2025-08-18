import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
from main import Fusion_Model
from load_database import dataset_loaders
from configurations import Config
from main import weights_init
from main import custom_weights_init
import random
import numpy as np
import time
from datetime import timedelta

start_time = time.time()

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    # 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

#
set_seed(42)

# 
config = Config(dataset_name='DEAP')

# 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 
num_epochs = 10

# 
train_loaders, test_loaders = dataset_loaders(config.dataset_name, batch_size=config.batch_size)

best_result = []

print(f"Number of train loaders: {len(train_loaders)}")
print(f"Number of test loaders: {len(test_loaders)}")


for p in range(config.num_subjects):
    train_loader = train_loaders[p]   # list index out of range(error)
    test_loader = test_loaders[p]

    best_acc, best_f1 = 0.0, 0.0
    last_epoch_acc, last_epoch_f1 = 0.0, 0.0

    model = Fusion_Model(config).to(config.device)

    # 
    model.apply(weights_init)
    model.apply(custom_weights_init)
    model.PSD_map_backbone.init_weights()
    model.hf_icma.init_weights()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    conv1_outputs = []

    # fused_x_outputs = []
    # all_labels_list = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for eeg_map, eeg_stat, peri, labels in train_loader:
            eeg_map, eeg_stat, peri, labels = eeg_map.to(device), eeg_stat.to(device), peri.to(device), labels.to(device)
            labels = labels[:, 0].long() if labels.dim() > 1 else labels.long() # choose valence or arousal

            optimizer.zero_grad()
            _, _, _, _, e = model(eeg_map, eeg_stat, peri)
            loss = criterion(e, labels)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * eeg_map.size(0)

        train_loss /= len(train_loader.dataset)

        # test model
        model.eval()
        test_acc, all_predictions, all_labels = 0.0, [], []

        with torch.no_grad():
            for eeg_map, eeg_stat, peri, labels in test_loader:
                eeg_map, eeg_stat, peri, labels = eeg_map.to(device), eeg_stat.to(device), peri.to(device), labels.to(device)
                labels = labels[:, 0].long() if labels.dim() > 1 else labels.long() # choose valence or arousal

                _, _, _, _, e = model(eeg_map, eeg_stat, peri)
                _, predictions = torch.max(e, 1)
                test_acc += torch.sum(predictions == labels.data).item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
 
        # test_acc /= len(test_loader.dataset)
        test_acc = accuracy_score(all_labels, all_predictions)
        test_f1 = f1_score(all_labels, all_predictions, average='macro')
        
        last_epoch_acc = test_acc
        last_epoch_f1 = test_f1
        
        print(f"Subject ID {p+1}, Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Acc: {test_acc*100:.2f}%, Test F1: {test_f1:.2f}")



    # Add the accuracy and F1 score of the last epoch to the best result list
    best_result.append([p + 1, last_epoch_acc * 100, last_epoch_f1])


print("Best Test Results:")
for result in best_result:
    print(f"Subject {result[0]}: Last Epoch Acc: {result[1]:.2f}%, Last Epoch F1: {result[2]:.2f}")


results_df = pd.DataFrame(best_result, columns=['Subject ID', 'Last Epoch Accuracy', 'Last Epoch F1 Score'])
results_df.to_excel(".xlsx", index=False)

end_time = time.time()
elapsed_time = end_time - start_time 
formatted_time = str(timedelta(seconds=elapsed_time)) 
print(f"Total runtime: {formatted_time}")




