import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import os
import random
import numpy as np
from networks.vae import Conv1DVAE
from data_loader import get_dataloaders
from config import Config

def loss_function(recon_x, x, mu, logvar, kl_weight):
    recon_loss = F.l1_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train():
    set_seed(Config.RANDOM_SEED)
    print(f"데이터 로딩 및 전처리({Config.SCALING_METHOD}) 중... 잠시만 기다려주세요.")
    train_loader, _, _ = get_dataloaders()
    
    total_batches = len(train_loader)
    print(f"데이터 로딩 완료! 총 배치 수: {total_batches} (Batch Size: {Config.BATCH_SIZE})")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 기기: {device}")
    
    model = Conv1DVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    model.train()
    global_step = 0
    
    print("본격적인 학습을 시작합니다...")
    for epoch in range(Config.EPOCHS):
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, _, mu, logvar = model(data)
            
            warm_up_steps = Config.WARM_UP_PERIOD * total_batches
            kl_weight = min(1.0, global_step / warm_up_steps) if warm_up_steps > 0 else 1.0
            
            loss, _, _ = loss_function(recon_batch, data, mu, logvar, kl_weight)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            global_step += 1
            
            # 500 배치마다 중간 진행 상황 출력
            if (batch_idx + 1) % 500 == 0:
                print(f"   -> [Epoch {epoch+1:02d}] Batch {batch_idx+1}/{total_batches} 진행 중...")
                
        avg_loss = total_loss / total_batches
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1:02d}/{Config.EPOCHS} | Avg Loss: {avg_loss:.4f} | KL Weight: {kl_weight:.4f} | Time: {epoch_time:.2f}s")
    
    tmp_save_path = Config.MODEL_SAVE_PATH + ".tmp"
    torch.save(model.state_dict(), tmp_save_path)
    os.replace(tmp_save_path, Config.MODEL_SAVE_PATH)
    print(f"훈련 종료 및 가중치 저장 완료: {Config.MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train()