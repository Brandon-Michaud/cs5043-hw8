import tensorflow as tf

from tensorflow.keras.utils import plot_model
from tensorflow import keras

# Provided
from chesapeake_loader4 import *
from hw8_parser import *
from diffusion_tools import *


if __name__ == '__main__':
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()

    # Turn off GPU?
    if not args.gpu or "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        tf.config.set_visible_devices([], 'GPU')
        print('NO VISIBLE DEVICES!!!!')

    # GPU check
    visible_devices = tf.config.get_visible_devices('GPU')
    n_visible_devices = len(visible_devices)
    print('GPUS:', visible_devices)
    if n_visible_devices > 0:
        for device in visible_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print('We have %d GPUs\n' % n_visible_devices)
    else:
        print('NO GPU')

    # Set number of threads, if it is specified
    if args.cpus_per_task is not None:
        tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task)
        tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task)

    # Noise schedule
    beta, alpha, gamma = compute_beta_alpha(args.n_steps, args.beta_start, args.beta_end)

    # Load dataset
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

    # Load model
    fname = 'results/diffusion_v6_model'
    model = keras.models.load_model(fname)

    # Take one batch of images and labels
    for I, L in ds.take(1):
        print(I.shape, L.shape)

    # Inference with loaded I/L
    TS = list(range(args.n_steps))
    stepdata = list(zip(TS, beta, alpha, gamma))
    stepdata.reverse()

    # Random noise
    Z = np.random.normal(loc=0, scale=1.0, size=I.shape)
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

        # Adjust the image
        Z = Z / np.sqrt(1 - b) - delta * b / (np.sqrt(1 - a) * np.sqrt(1 - b))

        if ts > 0:
            # Add exploratory noise
            noise = np.random.normal(loc=0, scale=1.0, size=I.shape)
            Z = Z + g * noise

    # Final step
    Zs.append(Z)

    # Dimensions of grid of de-noised images
    cols = 7
    rows = args.n_steps // cols + 2

    # De-noise 3 images
    for i in range(3):
        # Create grid of images
        fig, axs = plt.subplots(rows, cols)

        # Show labels and corresponding true image
        cl = np.argmax(L[i, :, :, :], axis=-1)
        axs[0, 0].imshow(cl, vmax=6, vmin=0)
        axs[0, 1].imshow(I[i, :, :, :])

        # Remove ticks on every subplot
        for j in range(cols):
            for k in range(rows):
                axs[k, j].set_xticks([])
                axs[k, j].set_yticks([])

        # Show de-noising steps
        for j, Z in enumerate(Zs):
            axs[j // cols + 1, j % cols].imshow(convert_image(Z[i, :, :, :]))

        # Save figure
        fig.savefig(f'figures/v6_steps_{i}.png')

    # Create gallery of de-noised of images
    rows = 25
    cols = 3
    fig, axs = plt.subplots(rows, cols, figsize=(16, 300))

    # Get only de-noised images
    final_Z = Zs[len(Zs) - 1]

    # Remove ticks on every subplot
    for i in range(rows):
        for j in range(cols):
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    # Show gallery of final images
    for i in range(rows):
        # Show label
        cl = np.argmax(L[i, :, :, :], axis=-1)
        axs[i, 0].imshow(cl, vmax=6, vmin=0)

        # Show real image
        axs[i, 1].imshow(I[i, :, :, :])

        # Show de-noised image
        axs[i, 2].imshow(convert_image(final_Z[i, :, :, :]))

    # Save figure
    fig.savefig(f'figures/v6_gallery.png')


