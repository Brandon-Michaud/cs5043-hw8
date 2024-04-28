import tensorflow as tf

from tensorflow.keras.utils import plot_model
from tensorflow import keras

# Provided
from chesapeake_loader2 import *
from hw8_parser import *
from diffusion_tools import *


if __name__ == '__main__':
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()

    beta, alpha, gamma = compute_beta_alpha(args.n_steps, args.beta_start, args.beta_end)

    ds = create_single_dataset(base_dir=args.dataset,
                               full_sat=False,
                               partition='valid',
                               patch_size=args.image_size,
                               fold=args.fold,
                               cache_path=args.cache,
                               repeat=args.repeat,
                               shuffle=args.shuffle,
                               batch_size=args.batch,
                               prefetch=args.prefetch,
                               num_parallel_calls=args.num_parallel_calls)

    fname = 'results/diffusion_v3_model'
    model = keras.models.load_model(fname)

    for I, L in ds.take(1):
        print(I.shape, L.shape)

    # Inference with loaded I/L
    TS = list(range(args.n_steps))
    stepdata = list(zip(TS, beta, alpha, gamma))
    stepdata.reverse()
    print(TS)
    # Random noise
    Z = np.random.normal(loc=0, scale=1.0, size=I.shape)
    print("SHAPE:", Z.shape)
    one = np.ones(shape=Z.shape)
    zero = np.zeros(shape=Z.shape)
    Zs = []

    # Loop over timesteps
    for ts, b, a, g in stepdata:
        Zs.append(Z)

        # All examples get the same time index
        t_tensor = ts * np.ones(shape=(I.shape[0], 1))

        # Predict the noise
        delta = model.predict(x={'image_input': Z, 'time_input': t_tensor, 'label_input': L})

        print(ts, a, b)

        # Adjust the image
        Z = Z / np.sqrt(1 - b) - delta * b / (np.sqrt(1 - a) * np.sqrt(1 - b))

        if ts > 0:
            # Add exploratory noise
            noise = np.random.normal(loc=0, scale=1.0, size=I.shape)
            Z = Z + g * noise

    # Final step
    Zs.append(Z)

    i = 1
    cols = 7
    fig, axs = plt.subplots(args.n_steps // cols + 2, cols)

    cl = np.argmax(L[i, :, :, :], axis=-1)
    axs[0, 0].imshow(cl, vmax=6, vmin=0)
    axs[0, 1].imshow(I[i, :, :, :])

    for j in range(cols):
        axs[0, j].set_xticks([])
        axs[0, j].set_yticks([])

    for j, Z in enumerate(Zs):
        axs[j // cols + 1, j % cols].imshow(convert_image(Z[i, :, :, :]))
        axs[j // cols + 1, j % cols].set_xticks([])
        axs[j // cols + 1, j % cols].set_yticks([])

    fig.savefig('figures/test.png')
