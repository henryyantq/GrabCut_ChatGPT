import numpy as np
import scipy.signal

class GrabCut:
    def __init__(self, img):
        self.img = img.astype(np.float32) / 255.0
        self.mask = np.zeros(img.shape[:2], dtype=np.uint8)
        self.beta = None
        self.gamma = None
        self.fg = None
        self.bg = None

    def compute_beta(self):
        h, w = self.img.shape[:2]
        dy, dx = np.gradient(self.img)
        self.beta = 1 / (2 * np.mean(np.abs(dx))) + 1 / (2 * np.mean(np.abs(dy)))

    def compute_gamma(self):
        h, w = self.img.shape[:2]
        mask = self.mask.astype(np.bool)
        self.gamma = np.zeros((h, w), dtype=np.float32)
        self.gamma[mask] = -np.log(1 - self.fg[mask])
        self.gamma[~mask] = -np.log(1 - self.bg[~mask])

    def compute_fg_bg(self):
        h, w = self.img.shape[:2]
        mask = self.mask.astype(np.bool)
        kernel = np.ones((3, 3), dtype=np.float32)
        fg = np.zeros((h, w), dtype=np.float32)
        bg = np.zeros((h, w), dtype=np.float32)
        fg[mask], bg[mask] = 1.0, 0.0
        fg[~mask], bg[~mask] = 0.0, 1.0
        for i in range(5):
            fg = mask * self.img / (self.gamma + self.beta)
            bg = (1 - mask) * self.img / (self.gamma + self.beta)
            fg = scipy.signal.convolve2d(fg, kernel, mode='same', boundary='symm')
            bg = scipy.signal.convolve2d(bg, kernel, mode='same', boundary='symm')
        self.fg, self.bg = fg, bg

    def compute_mask(self, threshold=0.5):
        mask = np.zeros_like(self.mask, dtype=np.uint8)
        mask[self.fg > threshold] = 1
        mask[self.bg > threshold] = 0
        self.mask = mask

    def run(self, num_iterations=5, threshold=0.5):
        self.compute_beta()
        for i in range(num_iterations):
            self.compute_gamma()
            self.compute_fg_bg()
        self.compute_mask(threshold)
