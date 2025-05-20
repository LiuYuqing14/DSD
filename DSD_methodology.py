import torch
from torch.distributions import Binomial, Multinomial

def stratified_time_sampling(T: float, N: int, device: str = 'cuda') -> torch.Tensor:
    """
    Sample time steps using stratified sampling across the interval [0, T].

    Parameters:
    T (float): Maximum time in the diffusion process.
    N (int): Number of time steps to sample.
    device (str): Device on which to run the calculations (e.g., 'cuda' or 'cpu').

    Returns:
    torch.Tensor: Tensor of sampled times (shape: (N,))
    """
    # 确定观测时间序列（离散时间？）
    # Create N uniform strata (subintervals) in the range [0, T]
    # Stratified intervals will have width T/N
    intervals = torch.linspace(0, T, N + 1, device=device)  # Dividing into N+1 points

    # Sample uniformly within each interval
    # We sample in the range [intervals[i], intervals[i+1]] for each interval i
    # Create random values between 0 and 1, then scale them to fit in the intervals
    stratified_samples = intervals[:-1] + (intervals[1:] - intervals[:-1]) * torch.rand(N, device=device)

    return stratified_samples

class DSDCorruptor:
    """Forward corruption process from Discrete Spatial Diffusion (Santos et al., 2025)."""
    def __init__(self, rate: float = 1.0, dt: float = 1.0,
                 boundary: str = "noflux", device="cuda"):
        """
        Args
        ----
        rate      Jump rate r (per particle, per second).
        dt        Time step Δt for the Poisson leap (set so r·Δt ≲ 0.1 for accuracy).
        boundary  'noflux' | 'periodic'
        """
        self.r, self.dt = rate, dt
        self.boundary = boundary
        self.device   = device

        # kernels for neighbour shifts (dx,dy)
        self.shifts = torch.tensor([[0,-1],[0,1],[-1,0],[1,0]], device=device)

    def _apply_boundary(self, x):
        if self.boundary == "noflux":
            return x  # nothing to do: rolled moves that cross edge fall outside tensor and we drop them
        elif self.boundary == "periodic":
            return x  # torch.roll already applies periodic wrap‑around
        else:
            raise ValueError("boundary must be 'noflux' or 'periodic'")

    def step(self, img_int: torch.Tensor) -> torch.Tensor:
        """
        One Poisson–leap of length Δt.

        img_int  (B, C, H, W)  *integer* tensor of particle counts
        returns  corrupted image with same shape & dtype
        """
        img_int = img_int.to(self.device) # convert image into (H, W, C)
        prob_move = 1.0 - torch.exp(-self.r * self.dt) # Poisson random

        # sample how many particles leave each pixel
        movers = Binomial(total_count=img_int, probs=prob_move).sample().to(img_int.dtype)  # (H, W, C)
        stay   = img_int - movers # separate the moving particles and left particles

        # split movers into 4 directions (uniform)
        multinom_probs = torch.full((4,), 0.25, device=self.device) # assume uniform movement
        dir_counts = Multinomial(total_count=movers.reshape(-1).to(torch.float),
                                 probs=multinom_probs).sample().to(img_int.dtype)
        # dir_counts count the number of
        dir_counts = dir_counts.reshape(4, *movers.shape)  # (4, H, W, C)

        # scatter to neighbours
        out = stay.clone() # clone
        for d, (dx, dy) in enumerate(self.shifts):
            # moves these particles to the appropriate neighboring
            shift = torch.roll(dir_counts[..., d], shifts=(dx, dy), dims=(-2, -1))
            if self.boundary == "noflux":
                # zero‑out flux that wrapped around
                if dx == -1: shift[..., -1, :] = 0                    # top edge
                if dx ==  1: shift[...,  0, :] = 0                    # bottom
                if dy == -1: shift[..., :, -1] = 0                    # left
                if dy ==  1: shift[..., :,  0] = 0                    # right
            out += shift

        return out # return updated particles

    def corrupt(self, img_int: torch.Tensor, t: float) -> torch.Tensor:
        n_steps = int(torch.round(torch.tensor(t / self.dt)).item())
        x = img_int
        for _ in range(n_steps):
            x = self.step(x)
        return x

    def stratified_corrupt(self, img_int: torch.Tensor, T: float, N: int) -> torch.Tensor:
        """
        Corrupt the image using stratified time sampling.

        Parameters:
        img_int (torch.Tensor): Input image with particle counts (shape: (H, W, C)).
        T (float): Maximum time for the corruption process.
        N (int): Number of time steps to sample.

        Returns:
        torch.Tensor: The corrupted image after applying the diffusion process.
        """
        # Sample stratified times
        stratified_times = stratified_time_sampling(T, N, device=self.device)

        # Simulate diffusion at each sampled time step
        x = img_int.clone()
        for t in stratified_times:
            # Here, `step()` would be a method of your corruption process
            # that simulates diffusion over time (you should have this implemented already).
            # Use the step method in 3.1
            x = DSDCorruptor.step(x)  # Apply the diffusion step for each time point

        return x

