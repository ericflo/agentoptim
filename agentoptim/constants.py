"""Constants used throughout AgentOptim."""

# Maximum number of questions allowed per EvalSet
MAX_QUESTIONS_PER_EVALSET = 100

# Maximum number of EvalSets allowed
MAX_EVALSETS = 100

# Maximum number of evaluation runs to store
MAX_EVAL_RUNS = 500

# Maximum number of system message optimization runs to store
MAX_SYSOPT_RUNS = 500

# Maximum number of candidate system messages per optimization
MAX_CANDIDATES = 20

# Maximum length for system messages
MAX_SYSTEM_MESSAGE_LENGTH = 10000

# Default number of candidates to generate
DEFAULT_NUM_CANDIDATES = 5

# Default timeout values
DEFAULT_API_TIMEOUT_SECONDS = 120  # 2 minutes

# API base URLs
DEFAULT_LOCAL_API_BASE = "http://localhost:1234/v1"
DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"
DEFAULT_ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"

# Default models for different providers
DEFAULT_LOCAL_MODEL = "meta-llama-3.1-8b-instruct"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"

# Default generator model (should handle structured output well)
DEFAULT_GENERATOR_MODEL = "gpt-4o-mini"

# Default pagination settings
DEFAULT_PAGE_SIZE = 10
MAX_PAGE_SIZE = 100

# System message diversity levels
DIVERSITY_LEVELS = ["low", "medium", "high"]

# System message optimization domains
DEFAULT_DOMAINS = ["general", "customer_support", "technical", "creative", "educational"]