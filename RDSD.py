import torch
import torch.nn as nn

class ReverseDiffusion:
    def __init__(self, model, beta_schedule, sigma_schedule, dt=0.01, device='cuda'):
        """
        Args:
        - model: Neural network model to predict the score function
        - beta_schedule: A schedule for noise variance \(\beta(t)\)
        - sigma_schedule: A schedule for noise magnitude \(\sigma(t)\)
        - dt: Time step for the reverse process
        - device: Device to run the model on
        """
        self.model = model
        self.beta_schedule = beta_schedule
        self.sigma_schedule = sigma_schedule
        self.dt = dt
        self.device = device

    def score_function(self, x, t):
        """
        Given an image and a time step t, predict the score function
        (the gradient of log probability of the data).

        Args:
        - x: Input tensor representing the noisy image
        - t: Time step

        Returns:
        - Gradient of log p_t(x), i.e., the score function
        """
        return self.model(x, t)  # model should return the score function

    def reverse_step(self, x, t):
        """
        Perform one step of the reverse process (denoising).

        Args:
        - x: Noisy image at time step t
        - t: Current time step

        Returns:
        - Updated image after one reverse diffusion step
        """
        # Compute the noise schedule parameters at time t
        sigma_t = self.sigma_schedule(t)
        beta_t = self.beta_schedule(t)

        # Compute the score function (gradient of log probability)
        score = self.score_function(x, t)

        # Apply Euler-Maruyama discretization to the reverse SDE
        dx = -0.5 * beta_t * x + sigma_t * score
        x_next = x + dx * self.dt  # Update the image

        return x_next

    def reverse_process(self, x0, timesteps):
        """
        Perform the full reverse diffusion process to denoise the image.

        Args:
        - x0: Original image (clean image)
        - timesteps: List of time steps from which to sample

        Returns:
        - Denoised image after running the reverse process
        """
        # Start with pure noise and iterate the reverse process
        x_t = x0  # Assuming we start with the noisy image
        for t in timesteps:
            x_t = self.reverse_step(x_t, t)

        return x_t

class ScoreNet(nn.Module):
    # 我们需要一个score function 吗？
    def __init__(self, channels=3):
        super(ScoreNet, self).__init__()
        # Simple convolutional neural network for score function prediction
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(128 * 28 * 28, 1)  # Assume image is 28x28

    def forward(self, x, t):
        # Predict the score function (gradient of log probability)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        score = self.fc(x)
        return score

