

# Import core packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as tick

def compute_fourier_kernel(N: int) -> np.ndarray:
  """Computes the Fourier space kernel which represents the mapping between 
    convergence (kappa) and shear (gamma).
  Args:
    N (int): x,y dimension of image patch (assumes square images).
  Returns:
    𝒟 (np.ndarray): Fourier space Kaiser-Squires kernel, with shape = [N,N].
  """
  # Generate grid of Fourier domain
  kx = np.arange(N).astype(np.float64) - N/2
  ky, kx = np.meshgrid(kx, kx)
  k = kx**2+ky**2
  # Define Kaiser-Squires kernel
  𝒟 = np.zeros((N, N), dtype=np.complex128)
  𝒟 = np.where(k > 0, ((kx ** 2.0 - ky ** 2.0) + 1j * (2.0 * kx * ky))/k, 𝒟)
  # Apply inverse FFT shift 
  return np.fft.ifftshift(𝒟)

 
def forward_model(𝜅: np.ndarray, 𝒟: np.ndarray) -> np.ndarray:
  """Applies the forward mapping between convergence and shear through their 
    relationship in Fourier space.
  Args:
    𝜅 (np.ndarray): Convergence field, with shape [N,N].
    𝒟 (np.ndarray): Fourier space Kaiser-Squires kernel, with shape = [N,N].
  Returns:
    γ (np.ndarray): Shearing field, with shape [N,N].
  """
  𝓕𝜅 = np.fft.fft2(𝜅) # Perform 2D forward FFT
  𝓕γ = 𝓕𝜅 * 𝒟 # Map convergence onto shear
  return np.fft.ifft2(𝓕γ) # Perform 2D inverse FFT

 
def backward_model(γ: np.ndarray, 𝒟: np.ndarray) -> np.ndarray:
  """Applies the backward mapping between shear and convergence through their 
    relationship in Fourier space.
  Args:
    γ (np.ndarray): Shearing field, with shape [N,N].
    𝒟 (np.ndarray): Fourier space Kaiser-Squires kernel, with shape = [N,N].
  Returns:
    𝜅 (np.ndarray): Convergence field, with shape [N,N].
  """
  𝓕γ = np.fft.fft2(γ) # Perform 2D forward FFT
  𝓕𝜅 = 𝓕γ / 𝒟 # Map convergence onto shear
  𝓕𝜅 = np.nan_to_num(𝓕𝜅, nan=0, posinf=0, neginf=0) # Remove singularities
  return np.fft.ifft2(𝓕𝜅) # Perform 2D inverse FFT

def noise_maker(theta, ngrid, ngal, kappa):
    """Adds some random Gaussian noise to a mock weak lensing map.
    Args:
        theta (float): Opening angle in deg.
        ngrid (int): Number of grids.
        ngal (int): Number of galaxies.
        kappa (np.ndarray): Convergence map.
    
    Returns:
        gamma (np.ndarray): A synthetic representation of the shear field, gamma, with added noise.
    """
    D = compute_fourier_kernel(ngrid) #Fourier kernel
    sigma = 0.37/np.sqrt(((theta*60/ngrid)**2)*ngal)
    gamma = forward_model(kappa, D) + sigma*(np.random.randn(ngrid,ngrid) + 1j*np.random.randn(ngrid,ngrid))
    return gamma


def Gaussian_smoothing(kappa,m,n,theta,ngrid):
    """Applies Gaussian smoothing to a convergence map.

    This is done by taking Fourier transform of the convergence map, and a Gaussian,
    convolving them, and then applying an inverse Fourier transform to the result.

    Args:
        kappa (np.ndarray): Convergence map.
        m,n (int, int): The dimensions, in pixels, of kappa.
        theta (float): Opening angle in deg.
        ngrid (int): Number of grids.

    Returns:
        smoothed_kappa (np.ndarray): Returns a smoothed representation of the the convergence field.
    """
    kappa_f = np.fft.fft2(kappa) #Fourier transform of kappa
    kappa_f_shifted = np.fft.fftshift(kappa_f) #Changes the indexing of the Fourier coefficients
    
    Gaussian_filter = np.zeros((m,n))
    i = (25*ngrid)/(60*theta)
    sig_pix = i/(2*ngrid*np.sqrt(2*np.log(2)))

    s = int(m/2)
    t = int(n/2)
    x = np.arange(-s,s)
    y = np.arange(-t,t)
    xx,yy = np.meshgrid(x,y)

    const = 1/(np.sqrt(2*np.pi*sig_pix**2))
    exponential = np.exp(-(xx**2 + yy**2)*2*(sig_pix*np.pi)**2)
    Gaussian_filter = const*exponential

    smoothed_kappa_f = np.fft.ifftshift(kappa_f_shifted*Gaussian_filter)
    smoothed_kappa = np.fft.ifft2(smoothed_kappa_f)
    return smoothed_kappa

ng = 384
kappa = np.load("/share/gpu0/jjwhit/mass_map_dataset/kappa20_cropped/kappa_val/kappa_run_09907.npy")


cbar_font_size = 14
title_fonts = 26


fig = plt.figure(figsize=(12,10))
ax = plt.gca()
ax.set_title("Mock Convergence Map", fontsize=title_fonts)
mx,mn = np.max(kappa),np.min(kappa)
im = plt.imshow(kappa, cmap ="inferno", vmax=mx*0.7, vmin=mn, origin="lower")
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(im, cax=cax)
cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
cbar.ax.tick_params(labelsize=cbar_font_size)
ax.set_xticks([]);ax.set_yticks([])
plt.savefig('mock_c_map.png', bbox_inches='tight', dpi=200)


gamma = noise_maker(5.0, ng, 30, kappa)

fig = plt.figure(figsize=(12,10))
ax = plt.gca()
ax.set_title("Simulated Observation", fontsize=title_fonts)
max, min = np.max(gamma.real), np.min(gamma.real)
im = plt.imshow(gamma.real, cmap="inferno",vmax = max, vmin = min * 0.5, origin="lower")
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(im, cax=cax)
cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
cbar.ax.tick_params(labelsize=cbar_font_size)
ax.set_xticks([]);ax.set_yticks([])

plt.savefig('mock_observation.png', bbox_inches='tight', dpi=200)