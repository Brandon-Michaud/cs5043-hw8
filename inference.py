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

    fname = 'results/diffusion_v6_model'
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

    cols = 7
    rows = args.n_steps // cols + 2

    for i in range(3):
        fig, axs = plt.subplots(rows, cols)

        cl = np.argmax(L[i, :, :, :], axis=-1)
        axs[0, 0].imshow(cl, vmax=6, vmin=0)
        axs[0, 1].imshow(I[i, :, :, :])

        for j in range(cols):
            for k in range(rows):
                axs[k, j].set_xticks([])
                axs[k, j].set_yticks([])

        for j, Z in enumerate(Zs):
            axs[j // cols + 1, j % cols].imshow(convert_image(Z[i, :, :, :]))
            axs[j // cols + 1, j % cols].set_xticks([])
            axs[j // cols + 1, j % cols].set_yticks([])

        fig.savefig(f'figures/steps_{i}.png')
