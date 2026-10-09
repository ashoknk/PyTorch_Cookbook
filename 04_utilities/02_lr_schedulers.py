"""
22. Learning Rate Schedulers & Hyperparameter Tuning in PyTorch

The main purpose of a Learning Rate Scheduler script is to show how to automatically adjust the step size 
(learning rate) of your optimizer as training progresses, 
rather than keeping it fixed at the same number the entire time.

1. StepLR
   - How It Adjusts Speed: Drops speed by a fixed multiplier after every N epochs 
     (e.g., cuts speed by half every 10 epochs).
   - Best Used When: You want simple, predictable, scheduled drops at fixed intervals.

2. ReduceLROnPlateau
   - How It Adjusts Speed: Monitors validation loss. If loss stops improving for a 
     few epochs, it automatically reduces speed.
   - Best Used When: You want the model to slow down only when it actually gets stuck.

3. CosineAnnealingLR
   - How It Adjusts Speed: Smoothly lowers (and optionally raises) speed following 
     a soft cosine wave curve.
   - Best Used When: You are training modern deep networks (like CNNs or Transformers) 
     for maximum accuracy.

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
    # Highly popular for state-of-the-art vision models.
    #   T_max=30: Smoothly reduces LR over a 30-epoch cycle.
    #             Tells the scheduler how long it should take to smoothly lower the learning rate
    #   eta_min=0.001: 
    #             Sets a floor so LR drops to 0.001 instead of stopping at 0.0
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
    
    """
    Simulate a validation loss profile that drops quickly, then plateaus. A scenario where training loss stops improving.
    - Phase 1: [1.0 - (0.03 * i) for i in range(15)]
        Creates a steadily decreasing loss list for epochs 0-14: 1.0, 0.97, 0.94, ..., 0.58.
    - Phase 2: [0.55 for _ in range(15)]
        Appends 0.55 fifteen times in a row for epochs 15-29: 0.55, 0.55, 0.55, ...
    """
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


"""
SCHEDULER OUTPUT INTERPRETATION

- StepLR: Decreases in fixed, step-like drops. Around Epoch 15, it halved from 
  0.1000 to 0.0500, and by Epoch 30, it dropped again to 0.0250.

- CosineAnnealingLR: Decreases following a smooth, continuous wave curve. At Epoch 15 
  (midway), it smoothly drifted down to 0.0557, and by Epoch 30 (nearing T_max=30), 
  it smoothly settled down near its floor (eta_min=0.0010) at 0.0013.

- ReduceLROnPlateau: Waits passively until loss stops improving. Because loss was 
  steadily decreasing during the first 15 epochs, it remained unchanged at 0.1000. 
  Once it detected flatlined loss in the second half, it stepped in and slashed 
  the rate down to 0.0010 by Epoch 30.

  ReduceLROnPlateau is the most adaptive for this specific simulated run. 
    It saved compute time by maintaining a high learning rate while the model was making progress, 
    then aggressively slowed down only when the loss flatlined.

  CosineAnnealingLR is the smoothest and often achieves the highest top-tier accuracy in 
    modern deep learning (like CNNs and Transformers) because it avoids sudden sharp drops.

  StepLR is the simplest, but it is rigid because it drops the rate strictly on schedule regardless of 
    whether the model actually needs to slow down or is still learning fast.
"""