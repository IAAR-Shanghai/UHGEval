import warnings

import transformers

# Ignore warnings
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
