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

    beta, alpha, _ = compute_beta_alpha(args.n_steps, args.beta_start, args.beta_end)

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
