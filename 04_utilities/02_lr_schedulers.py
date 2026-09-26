"""
22. Learning Rate Schedulers & Hyperparameter Tuning in PyTorch

This script demonstrates how to dynamically adjust learning rates during training 
to optimize model convergence. It covers standard PyTorch scheduler classes—including 
StepLR (decay after epochs), CosineAnnealingLR (cosine wave updates), and 
ReduceLROnPlateau (decaying when validation metrics stall)—and implements a basic 
comparison routine.

Learning Objectives:
1. Couple learning rate schedulers with optimizer parameters.
2. Step learning rate changes per-epoch or based on model loss performance.
3. Compare the optimization trajectories of different scheduler algorithms.
"""

# We import the core torch library.
import torch

# nn contains foundational neural network layers.
import torch.nn as nn

# lr_scheduler contains all standard learning rate adaptation strategies.
# Documentation: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
import torch.optim.lr_scheduler as lr_scheduler

def compare_schedulers():
    # Setup simple model and optimizer
    model = nn.Linear(10, 2)
    
    # ==========================================
    # 1. INITIALIZE SCHEDULERS
    # ==========================================
    print("--- 1. Initializing PyTorch LR Schedulers ---")
    
    # All schedulers must be coupled with an active optimizer instance
    opt1 = torch.optim.SGD(model.parameters(), lr=0.1)
    opt2 = torch.optim.SGD(model.parameters(), lr=0.1)
    opt3 = torch.optim.SGD(model.parameters(), lr=0.1)

    # 1.1 StepLR: Decays the learning rate by a multiplicative 'gamma' factor every 'step_size' epochs.
    # In this case: Multiply LR by 0.5 every 10 epochs.
    # Documentation: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
    scheduler_step = lr_scheduler.StepLR(opt1, step_size=10, gamma=0.5)

    # 1.2 CosineAnnealingLR: Gradually decays LR in a cosine-wave shape down to 'eta_min'.
    # Highly popular for state-of-the-art vision models. T_max represents cycle length.
    # Documentation: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    scheduler_cosine = lr_scheduler.CosineAnnealingLR(opt2, T_max=30, eta_min=0.001)

    # 1.3 ReduceLROnPlateau: Decays learning rate when a monitored validation metric (like validation loss)
    # stops improving for a specified 'patience' window of epochs.
    # Documentation: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
    scheduler_plateau = lr_scheduler.ReduceLROnPlateau(opt3, mode='min', factor=0.1, patience=5)

    # ==========================================
    # 2. RUN SIMULATED EPOCHS & RECORD LEARNING RATES
    # ==========================================
    print("\n--- 2. Simulating 30 Epochs of Tuning ---")
    
    step_lrs = []
    cosine_lrs = []
    plateau_lrs = []
    
    # We simulate a validation loss profile that drops quickly, then plateaus.
    simulated_losses = [1.0 - (0.03 * i) for i in range(15)] + [0.55 for _ in range(15)]

    for epoch in range(30):
        # 2.1 Track current learning rates from optimizer parameter groups
        step_lrs.append(opt1.param_groups[0]['lr'])
        cosine_lrs.append(opt2.param_groups[0]['lr'])
        plateau_lrs.append(opt3.param_groups[0]['lr'])

        # Simulate standard optimizer steps to satisfy PyTorch ordering conventions and silence warnings
        opt1.step()
        opt2.step()
        opt3.step()

        # 2.2 Advance Schedulers
        # StepLR and CosineAnnealingLR are stepped at the end of every epoch.
        scheduler_step.step()
        scheduler_cosine.step()
        
        # ReduceLROnPlateau requires the monitored validation metric passed to step().
        current_loss = simulated_losses[epoch]
        scheduler_plateau.step(current_loss)

    # ==========================================
    # 3. ANALYZE AND PRINT TRAJECTORIES
    # ==========================================
    print("\nScheduler Trajectory Results:")
    print(f"  Initial Learning Rates: StepLR = {step_lrs[0]:.4f} | Cosine = {cosine_lrs[0]:.4f} | Plateau = {plateau_lrs[0]:.4f}")
    
    # Print rates at mid-point (epoch 15) and final step (epoch 30)
    print(f"  Epoch 15 Rates:         StepLR = {step_lrs[14]:.4f} | Cosine = {cosine_lrs[14]:.4f} | Plateau = {plateau_lrs[14]:.4f}")
    print(f"  Epoch 30 Rates:         StepLR = {step_lrs[29]:.4f} | Cosine = {cosine_lrs[29]:.4f} | Plateau = {plateau_lrs[29]:.4f}")

    # Assertions to confirm correctness (rounded to 4 decimal places to prevent float-point representation mismatches)
    assert round(step_lrs[29], 4) == 0.0250, "StepLR decay value mismatch."
    assert round(plateau_lrs[29], 4) == 0.0010, "ReduceLROnPlateau decay value mismatch (expected 2 decays, final rate = 0.0010)."
    print("\nDynamic learning rate transitions verified successfully!")

if __name__ == "__main__":
    compare_schedulers()
